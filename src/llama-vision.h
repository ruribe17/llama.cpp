#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-arch.h"

#include <vector>
#include <array>

enum clip_projector_type {
    CLIP_PROJECTOR_TYPE_UNKNOWN,
    CLIP_PROJECTOR_TYPE_MLP,
};

enum mm_patch_merge {
    MM_PATCH_MERGE_UNKNOWN,
    MM_PATCH_MERGE_FLAT,
    MM_PATCH_MERGE_SPATIAL_UNPAD,
};

struct clip_hparams {
    vision_arch arch = VISION_ARCH_UNKNOWN;

    uint32_t image_size;
    uint32_t patch_size;
    uint32_t hidden_size;
    uint32_t n_intermediate;
    uint32_t projection_dim;
    uint32_t n_head;
    uint32_t n_layer;
    uint32_t max_pos_embd;
    int32_t select_layer = 0;
    bool use_gelu = false;

    float eps;

    clip_projector_type proj_type = CLIP_PROJECTOR_TYPE_UNKNOWN;
    mm_patch_merge mm_patch_merge_type = MM_PATCH_MERGE_FLAT;

    std::array<float, 3> image_mean;
    std::array<float, 3> image_std;

    std::array<int32_t, 32> image_grid_pinpoints; // TODO: should this be array of (x, y) pairs?
    int32_t image_crop_resolution;
};

struct clip_layer {
    // attention
    struct ggml_tensor * k_w = nullptr;
    struct ggml_tensor * k_b = nullptr;
    struct ggml_tensor * q_w = nullptr;
    struct ggml_tensor * q_b = nullptr;
    struct ggml_tensor * v_w = nullptr;
    struct ggml_tensor * v_b = nullptr;

    struct ggml_tensor * output_w = nullptr;
    struct ggml_tensor * output_b = nullptr;

    // layernorm 1
    struct ggml_tensor * norm_in_w = nullptr;
    struct ggml_tensor * norm_in_b = nullptr;

    // ff
    struct ggml_tensor * ffn_up_w = nullptr;
    struct ggml_tensor * ffn_up_b = nullptr;

    struct ggml_tensor * ffn_down_w = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr;

    // layernorm 2
    struct ggml_tensor * norm_out_w = nullptr;
    struct ggml_tensor * norm_out_b = nullptr;
};

struct clip_vision_model {
    struct clip_hparams hparams;
    ggml_backend_buffer_type_t buft;

    // embeddings
    struct ggml_tensor * class_embedding     = nullptr;
    struct ggml_tensor * patch_embeddings    = nullptr;
    struct ggml_tensor * patch_bias          = nullptr;
    struct ggml_tensor * position_embeddings = nullptr;

    struct ggml_tensor * pre_norm_w = nullptr;
    struct ggml_tensor * pre_norm_b = nullptr;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_norm_w = nullptr;
    struct ggml_tensor * post_norm_b = nullptr;

    struct ggml_tensor * projection = nullptr;

    // LLaVA projection
    struct ggml_tensor * mm_1_w = nullptr;
    struct ggml_tensor * mm_1_b = nullptr;
    struct ggml_tensor * mm_2_w = nullptr;
    struct ggml_tensor * mm_2_b = nullptr;

    struct ggml_tensor * image_newline = nullptr;
};

struct clip_context {
    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_t sched = nullptr;
    struct ggml_context * ctx_ggml = nullptr;

    const clip_vision_model * model;

    // temporary output data, to be picked up by llama_decode()
    struct ggml_tensor * output;
};

// for now, this only contains:
// - the instruction for ggml_conv_2d to break the image into patches
// - the pre-processed image data in f32
struct llama_vision_patches {
    uint32_t px; // size of patch
    uint32_t py; // size of patch
    size_t n_px; // number of patches in x direction
    size_t n_py; // number of patches in y direction
    // RGB float32 image (NHWC)
    // Memory layout: RGBRGBRGB...
    std::vector<std::vector<float>> buf; // preprocessed image data
};

inline mm_patch_merge mm_patch_merge_from_name(std::string & name) {
    if (name == "flat") {
        return MM_PATCH_MERGE_FLAT;
    } else if (name == "spatial_unpad") {
        return MM_PATCH_MERGE_SPATIAL_UNPAD;
    }
    return MM_PATCH_MERGE_UNKNOWN;
}

inline clip_projector_type clip_projector_type_from_name(std::string & name) {
    if (name == "mlp") {
        return CLIP_PROJECTOR_TYPE_MLP;
    }
    return CLIP_PROJECTOR_TYPE_UNKNOWN;
}

// only for sanity check: must be equal to n_embd of language model
uint32_t clip_n_mmproj_embd(const clip_vision_model & clip_model);

struct ggml_tensor * llama_vision_get_output_tensor(llama_context * ctx);
