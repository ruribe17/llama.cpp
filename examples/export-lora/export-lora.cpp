#include "common.h"
#include "ggml.h"
#include "ggml-alloc.h"

#include <map>
#include <vector>
#include <string>
#include <thread>
#include <fstream>

static bool g_verbose = false;

static std::string get_kv_str(struct gguf_context * ctx_gguf, const std::string & key){
    int id = gguf_find_key(ctx_gguf, key.c_str());
    return id < 0 ? "" : std::string(gguf_get_val_str(ctx_gguf, id));
}

static float get_kv_f32(struct gguf_context * ctx_gguf, const std::string & key) {
    int id = gguf_find_key(ctx_gguf, key.c_str());
    return id < 0 ? 0.0f : gguf_get_val_f32(ctx_gguf, id);
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static struct gguf_context * load_gguf(std::string & fname, struct ggml_context ** ctx_ggml) {
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ ctx_ggml,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load input GGUF from " + fname);
    }
    return ctx_gguf;
}

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

struct file_input {
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_context * ctx_gguf = nullptr;
    std::ifstream f_in;
    std::map<std::string, ggml_tensor *> tensors;
    float alpha;
    float scale;

    file_input(std::string & fname, float scale): f_in(fname, std::ios::binary), scale(scale) {
        if (!f_in.is_open()) {
            throw std::runtime_error("failed to open input gguf from " + fname);
        }

        ctx_gguf = load_gguf(fname, &ctx_meta);
        alpha = get_kv_f32(ctx_gguf, "adapter.lora.alpha");
        printf("%s: loaded gguf from %s\n", __func__, fname.c_str());

        for (ggml_tensor * cur = ggml_get_first_tensor(ctx_meta); cur; cur = ggml_get_next_tensor(ctx_meta, cur)) {
            std::string name(cur->name);
            tensors[name] = cur;
            if (g_verbose) {
                printf("%s: %s\n", __func__, cur->name);
            }
        }
    }

    ggml_tensor * get_tensor(std::string name) {
        if (tensors.find(name) == tensors.end()) {
            return nullptr;
        }
        return tensors[name];
    }

    void read_tensor_data(std::string name, std::vector<uint8_t> & buf) {
        if (tensors.find(name) == tensors.end()) {
            throw std::runtime_error("cannot find tensor with name: " + name);
        }
        auto len = ggml_nbytes(tensors[name]);
        if (buf.size() < len) {
            buf.resize(len);
        }
        auto i_tensor_in = gguf_find_tensor(ctx_gguf, name.c_str()); // idx of tensor in the input file
        auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor_in);
        f_in.seekg(offset);
        f_in.read((char* )buf.data(), len);
    }

    ~file_input() {
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
    }
};

struct lora_merge_ctx {
    // input base model + adapters
    file_input base_model;
    std::vector<std::unique_ptr<file_input>> adapters;

    // for computing merged tensor
    int n_threads;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    std::vector<uint8_t> read_buf;

    // output file
    struct gguf_context * ctx_out;
    struct ggml_context * ctx_out_ggml;
    std::ofstream fout;

    lora_merge_ctx(
            std::string & base_fname,
            std::vector<std::tuple<std::string, float>> & lora_files,
            std::string & outfile,
            int n_threads) : base_model(base_fname, 0), n_threads(n_threads), fout(outfile, std::ios::binary) {
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors

        if (gguf_find_key(base_model.ctx_gguf, LLM_KV_SPLIT_COUNT) >= 0) {
            throw std::runtime_error("split model is not yet supported");
        }

        for (auto lora_inp : lora_files) {
            auto fname = std::get<0>(lora_inp);
            auto scale = std::get<1>(lora_inp);
            std::unique_ptr<file_input> adapter(new file_input(fname, scale));
            check_metadata_lora(adapter.get());
            adapters.push_back(std::move(adapter));
        }

        ctx_out = gguf_init_empty();
        struct ggml_init_params params = {
            /*.mem_size   =*/ gguf_get_n_tensors(base_model.ctx_gguf)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx_out_ggml = ggml_init(params);
        backend = ggml_backend_cpu_init();
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }

    void check_metadata_lora(file_input * adapter) {
        auto general_type = get_kv_str(adapter->ctx_gguf, "general.type");
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }

        auto adapter_type = get_kv_str(adapter->ctx_gguf, "adapter.type");
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        auto general_arch_base = get_kv_str(base_model.ctx_gguf, "general.architecture");
        auto general_arch_lora = get_kv_str(adapter->ctx_gguf,   "general.architecture");
        if (general_arch_base != general_arch_lora) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }
    }

    ggml_type get_out_tensor_type(struct ggml_tensor * t) {
        if (t->type == GGML_TYPE_F32) {
            return GGML_TYPE_F32;
        } else {
            return GGML_TYPE_F16;
        }
    }

    void run_merge() {
        // prepare metadata
        gguf_set_kv(ctx_out, base_model.ctx_gguf);
        // output is forced to f16 for now
        gguf_set_val_u32(ctx_out, "general.file_type", LLAMA_FTYPE_MOSTLY_F16);

        // if true, this tensor can be lora-merged. if false, we skip merging and just copy data to outfile
        std::vector<std::pair<struct ggml_tensor *, bool>> base_tensors;
        for (auto & it : base_model.tensors) {
            bool t_a = true;
            bool t_b = true;
            for (auto & adapter : adapters) {
                t_a &= nullptr != adapter->get_tensor(it.first + ".lora_a");
                t_b &= nullptr != adapter->get_tensor(it.first + ".lora_b");
            }
            auto base_tensor = it.second;
            struct ggml_tensor * out_tensor;
            if (!t_a && !t_b) {
                // only copy
                out_tensor = ggml_dup_tensor(ctx_out_ggml, base_tensor);
                ggml_set_name(out_tensor, base_tensor->name);
                base_tensors.push_back(std::make_pair(out_tensor, false));
            } else if (t_a && t_b) {
                // need merging
                out_tensor = ggml_dup_tensor(ctx_out_ggml, base_tensor);
                out_tensor->type = get_out_tensor_type(base_tensor);
                ggml_set_name(out_tensor, base_tensor->name);
                base_tensors.push_back(std::make_pair(out_tensor, true));
            } else {
                throw std::runtime_error("tensor " + it.first + " missing either lora_a or lora_b");
            }
            gguf_add_tensor(ctx_out, out_tensor);
        }

        // placeholder for the meta data
        {
            size_t meta_size = gguf_get_meta_size(ctx_out);
            zeros(fout, meta_size);
        }

        // process base model tensors
        size_t n_merged = 0;
        for (auto & it : base_tensors) {
            if (it.second) {
                merge_tensor(it.first);
                n_merged++;
            } else {
                copy_tensor(it.first);
            }
        }

        // write output metadata
        {
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
            gguf_get_meta_data(ctx_out, data.data());
            fout.seekp(0);
            fout.write((const char *)data.data(), data.size());
        }

        printf("%s : merged %ld tensors with lora adapters\n", __func__, n_merged);
        printf("%s : wrote %ld tensors to output file\n", __func__, base_tensors.size());
    }

    void copy_tensor(struct ggml_tensor * base) {
        printf("%s :  %s\n", __func__, base->name);
        size_t len = ggml_nbytes(base);
        base_model.read_tensor_data(base->name, read_buf);
        fout.write((char* )read_buf.data(), len);
        zeros(fout, GGML_PAD(len, GGUF_DEFAULT_ALIGNMENT) - len);
    }

    void merge_tensor(struct ggml_tensor * base) {
        std::string name_base(base->name);
        std::string name_lora_a = name_base + ".lora_a";
        std::string name_lora_b = name_base + ".lora_b";

        printf("%s : %s\n", __func__, base->name);

        // context for input tensor
        std::vector<struct ggml_tensor *> inp_a(adapters.size());
        std::vector<struct ggml_tensor *> inp_b(adapters.size());
        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead()*(1+adapters.size()*2),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx = ggml_init(params);

        // alloc tensors
        struct ggml_tensor * inp = ggml_dup_tensor(ctx, base);
        for (size_t i = 0; i < adapters.size(); ++i) {
            auto t_a = adapters[i]->get_tensor(name_lora_a);
            auto t_b = adapters[i]->get_tensor(name_lora_b);
            inp_a[i] = ggml_dup_tensor(ctx, t_a);
            inp_b[i] = ggml_dup_tensor(ctx, t_b);
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

        // load data to backend buffer
        base_model.read_tensor_data(name_base, read_buf);
        ggml_backend_tensor_set(inp, read_buf.data(), 0, ggml_nbytes(inp));
        for (size_t i = 0; i < adapters.size(); ++i) {
            adapters[i]->read_tensor_data(name_lora_a, read_buf);
            ggml_backend_tensor_set(inp_a[i], read_buf.data(), 0, ggml_nbytes(inp_a[i]));
            adapters[i]->read_tensor_data(name_lora_b, read_buf);
            ggml_backend_tensor_set(inp_b[i], read_buf.data(), 0, ggml_nbytes(inp_b[i]));
        }

        // build graph
        struct ggml_cgraph * gf;
        {
            static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
            static std::vector<uint8_t> buf(buf_size);
            struct ggml_init_params params0 = {
                /*.mem_size   =*/ buf_size,
                /*.mem_buffer =*/ buf.data(),
                /*.no_alloc   =*/ true,
            };
            struct ggml_context * ctx0 = ggml_init(params0);
            gf = ggml_new_graph(ctx0);
            struct ggml_tensor * cur = inp;
            for (size_t i = 0; i < adapters.size(); ++i) {
                struct ggml_tensor * a_T = ggml_cont(ctx0, ggml_transpose(ctx0, inp_a[i]));
                struct ggml_tensor * delta = ggml_mul_mat(ctx0, a_T, inp_b[i]);
                // scale
                const float alpha = adapters[i]->alpha;
                const float rank  = (float) inp_b[i]->ne[0];
                const float scale = alpha ? adapters[i]->scale * alpha / rank : adapters[i]->scale;
                delta = ggml_scale(ctx0, delta, scale);
                cur = ggml_add(ctx0, cur, delta);
            }
            cur = ggml_cast(ctx0, cur, get_out_tensor_type(base));
            ggml_build_forward_expand(gf, cur);
            ggml_free(ctx0);
        }

        // compute
        {
            ggml_gallocr_alloc_graph(allocr, gf);
            ggml_backend_cpu_set_n_threads(backend, n_threads);
            ggml_backend_graph_compute(backend, gf);
        }

        // write data to output file
        {
            auto result = gf->nodes[gf->n_nodes - 1];
            size_t len = ggml_nbytes(result);
            if (read_buf.size() < len) {
                read_buf.resize(len);
            }
            ggml_backend_tensor_get(result, read_buf.data(), 0, len);
            fout.write((char* )read_buf.data(), len);
            zeros(fout, GGML_PAD(len, GGUF_DEFAULT_ALIGNMENT) - len);
        }

        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
    }

    ~lora_merge_ctx() {
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        gguf_free(ctx_out);
        ggml_free(ctx_out_ggml);
    }
};

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    printf("\nexample usage:\n");
    printf("\n  %s -m base-model.gguf --lora lora-file.gguf -o merged-model-f16.gguf\n", argv[0]);
    printf("\nNOTE: output model is F16\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    g_verbose = (params.verbosity == 1);
    try {
        lora_merge_ctx ctx(params.model, params.lora_adapter, params.lora_outfile, params.n_threads);
        ctx.run_merge();
    } catch (const std::exception & err) {
        fprintf(stderr, "%s\n", err.what());
        exit(EXIT_FAILURE);
    }

    printf("done, output file is %s\n", params.lora_outfile.c_str());

    return 0;
}
