import argparse
from typing import Dict

import torch
import numpy as np
from gguf import *
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLConfig,
)

VISION = "clip.vision"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def to_gguf_name(name: str) -> str:
    og = name
    name = name.replace("text_model", "t").replace("visual", "v")
    name = name.replace("blocks", "blk").replace("embeddings.", "")
    name = name.replace("attn.", "attn_")

    # Handle new Qwen2.5 MLP structure
    if "mlp.gate_proj" in name:
        name = name.replace("mlp.gate_proj", "ffn_gate")
    elif "mlp.up_proj" in name:
        name = name.replace("mlp.up_proj", "ffn_up")
    elif "mlp.down_proj" in name:
        name = name.replace("mlp.down_proj", "ffn_down")
    else:
        name = name.replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up")

    name = name.replace("proj.", "out.")
    name = name.replace("norm1", "ln1").replace("norm2", "ln2")
    name = name.replace("merger.mlp", 'mm')

    # For RMSNorm, which doesn't have bias
    if "weight_g" in name:
        name = name.replace("weight_g", "weight")

    print(f"[to_gguf_name] {og} --> {name}")
    return name


def find_vision_tensors(model, dtype) -> Dict[str, np.ndarray]:
    visual = model.visual
    tensor_map = {}

    for name, ten in visual.state_dict().items():
        ten = ten.numpy()
        if 'qkv' in name:
            if ten.ndim == 2:  # weight
                c3, _ = ten.shape
            else:  # bias
                c3 = ten.shape[0]
            assert c3 % 3 == 0
            c = c3 // 3
            wq = ten[:c]
            wk = ten[c: c * 2]
            wv = ten[c * 2:]
            tensor_map[to_gguf_name(f"visual.{name}").replace("qkv", "q")] = wq
            tensor_map[to_gguf_name(f"visual.{name}").replace("qkv", "k")] = wk
            tensor_map[to_gguf_name(f"visual.{name}").replace("qkv", "v")] = wv
        elif 'merger' in name:
            if name.endswith("ln_q.weight_g"):
                tensor_map['v.post_ln.weight'] = ten
            elif name.endswith("ln_q.bias") and 'weight_g' not in name:
                tensor_map['v.post_ln.bias'] = ten
            else:
                # "merger.mlp.%d.weight/bias" --> "mm.%d.weight/bias"
                tensor_map[to_gguf_name(name)] = ten
        elif 'patch_embed.proj.weight' in name:
            # NOTE: split Conv3D into Conv2Ds
            c1, c2, kt, kh, kw = ten.shape
            assert kt == 2, "Current implementation only support temporal_patch_size of 2"
            tensor_map["v.patch_embd.weight"] = ten[:, :, 0, ...]
            tensor_map["v.patch_embd.weight.1"] = ten[:, :, 1, ...]
        else:
            tensor_map[to_gguf_name(f"visual.{name}")] = ten

    for new_name, ten in tensor_map.items():
        if ten.ndim <= 1 or new_name.endswith("_norm.weight"):
            tensor_map[new_name] = ten.astype(np.float32)
        else:
            tensor_map[new_name] = ten.astype(dtype)
    tensor_map["v.position_embd.weight"] = np.zeros([10, 10], dtype=np.float32)  # dummy tensor, just here as a placeholder
    return tensor_map


def main(args):
    if args.data_type == 'fp32':
        dtype = torch.float32
        np_dtype = np.float32
        ftype = 0
    elif args.data_type == 'fp16':
        dtype = torch.float32
        np_dtype = np.float16
        ftype = 1
    else:
        raise ValueError()

    local_model = False
    model_path = ""
    model_name = args.model_name
    print("model_name: ", model_name)

    # Load the model with the specific Qwen2.5 class
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cpu"
    )
    cfg = model.config
    vcfg = cfg.vision_config

    if os.path.isdir(model_name):
        local_model = True
        if model_name.endswith(os.sep):
            model_name = model_name[:-1]
        model_path = model_name
        model_name = os.path.basename(model_name)
    fname_out = f"{model_name.replace('/', '-').lower()}-vision.gguf"

    fout = GGUFWriter(path=fname_out, arch="clip")
    fout.add_description("image encoder for Qwen2.5VL")

    fout.add_file_type(ftype)
    fout.add_bool("clip.has_text_encoder", False)
    fout.add_bool("clip.has_vision_encoder", True)
    fout.add_bool("clip.has_qwen2vl_merger", True)
    fout.add_bool("clip.is_qwen2_5", True)  # Flag to identify Qwen2.5 models
    fout.add_string("clip.projector_type", "qwen2vl_merger")

    print(cfg.vision_config)
    # SiLU activation
    fout.add_bool("clip.use_silu", True)
    fout.add_bool("clip.use_gelu", False)

    tensor_map = find_vision_tensors(model, np_dtype)
    for name, data in tensor_map.items():
        fout.add_tensor(name, data)

    fout.add_uint32("clip.vision.patch_size", vcfg.patch_size)
    fout.add_uint32("clip.vision.image_size", 14 * 40)  # reasonable size divisible by (14*2)
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), vcfg.hidden_size)
    fout.add_uint32("clip.vision.projection_dim", vcfg.hidden_size)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), vcfg.num_heads)
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), vcfg.depth)
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), vcfg.intermediate_size)
    fout.add_name(model_name)

    # Load the processor using the specific Qwen2.5 processor class
    if local_model:
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    else:
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name)

    # Get the image mean and std values from the processor
    fout.add_array("clip.vision.image_mean", processor.image_processor.image_mean)
    fout.add_array("clip.vision.image_std", processor.image_processor.image_std)

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    print("save model as: ", fname_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs='?', default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_type", nargs='?', choices=['fp32', 'fp16'], default="fp32")
    args = parser.parse_args()
    main(args)
