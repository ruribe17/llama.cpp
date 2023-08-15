#include "ggml.h"
#include "gguf-util.h"
#include "gguf-llama.h"

#include <cstdio>
#include <cinttypes>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

template<typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

bool gguf_ex_write(const std::string & fname) {
    struct gguf_context * ctx = gguf_init_empty();

    {
        gguf_set_val_u8  (ctx, "some.parameter.uint8",    0x12);
        gguf_set_val_i8  (ctx, "some.parameter.int8",    -0x13);
        gguf_set_val_u16 (ctx, "some.parameter.uint16",   0x1234);
        gguf_set_val_i16 (ctx, "some.parameter.int16",   -0x1235);
        gguf_set_val_u32 (ctx, "some.parameter.uint32",   0x12345678);
        gguf_set_val_i32 (ctx, "some.parameter.int32",   -0x12345679);
        gguf_set_val_f32 (ctx, "some.parameter.float32",  0.123456789f);
        gguf_set_val_bool(ctx, "some.parameter.bool",     true);
        gguf_set_val_str (ctx, "some.parameter.string",   "hello world");

        gguf_set_arr_data(ctx, "some.parameter.arr.i16", GGUF_TYPE_INT16,   std::vector<int16_t>{ 1, 2, 3, 4, }.data(), 4);
        gguf_set_arr_data(ctx, "some.parameter.arr.f32", GGUF_TYPE_FLOAT32, std::vector<float>{ 3.145f, 2.718f, 1.414f, }.data(), 3);
        gguf_set_arr_str (ctx, "some.parameter.arr.str",                    std::vector<const char *>{ "hello", "world", "!" }.data(), 3);
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_data = ggml_init(params);

    const int n_tensors = 10;

    // tensor infos
    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = "tensor_" + to_string(i);

        int64_t ne[GGML_MAX_DIMS] = { 1 };
        int32_t n_dims = rand() % GGML_MAX_DIMS + 1;

        for (int j = 0; j < n_dims; ++j) {
            ne[j] = rand() % 10 + 1;
        }

        struct ggml_tensor * cur = ggml_new_tensor(ctx_data, GGML_TYPE_F32, n_dims, ne);
        ggml_set_name(cur, name.c_str());

        {
            float * data = (float *) cur->data;
            for (int j = 0; j < ggml_nelements(cur); ++j) {
                data[j] = 100 + i;
            }
        }

        gguf_add_tensor(ctx, cur);
    }

    gguf_write_to_file(ctx, fname.c_str());

    fprintf(stdout, "%s: wrote file '%s;\n", __func__, fname.c_str());

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

// just read tensor info
bool gguf_ex_read_0(const std::string & fname) {
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    fprintf(stdout, "%s: version:      %d\n", __func__, gguf_get_version(ctx));
    fprintf(stdout, "%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    fprintf(stdout, "%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        fprintf(stdout, "%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            fprintf(stdout, "%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }

    // find kv string
    {
        const char * findkey = "some.parameter.string";

        const int keyidx = gguf_find_key(ctx, findkey);
        if (keyidx == -1) {
            fprintf(stdout, "%s: find key: %s not found.\n", __func__, findkey);
        } else {
            const char * key_value = gguf_get_val_str(ctx, keyidx);
            fprintf(stdout, "%s: find key: %s found, kv[%d] value = %s\n", __func__, findkey, keyidx, key_value);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        fprintf(stdout, "%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            fprintf(stdout, "%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
        }
    }

    gguf_free(ctx);

    return true;
}

// read and create ggml_context containing the tensors and their data
bool gguf_ex_read_1(const std::string & fname) {
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    fprintf(stdout, "%s: version:      %d\n", __func__, gguf_get_version(ctx));
    fprintf(stdout, "%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    fprintf(stdout, "%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        fprintf(stdout, "%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            fprintf(stdout, "%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        fprintf(stdout, "%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            fprintf(stdout, "%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
        }
    }

    // data
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        for (int i = 0; i < n_tensors; ++i) {
            fprintf(stdout, "%s: reading tensor %d data\n", __func__, i);

            const char * name = gguf_get_tensor_name(ctx, i);

            struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);

            fprintf(stdout, "%s: tensor[%d]: n_dims = %d, name = %s, data = %p\n",
                    __func__, i, cur->n_dims, cur->name, cur->data);

            // check data
            {
                const float * data = (const float *) cur->data;
                for (int j = 0; j < ggml_nelements(cur); ++j) {
                    if (data[j] != 100 + i) {
                        fprintf(stderr, "%s: tensor[%d]: data[%d] = %f\n", __func__, i, j, data[j]);
                        return false;
                    }
                }
            }
        }
    }

    fprintf(stdout, "%s: ctx_data size: %zu\n", __func__, ggml_get_mem_size(ctx_data));

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

// read just the tensor info and mmap the data in user code
bool gguf_ex_read_2(const std::string & fname) {
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    struct gguf_file file(fname.c_str(), "rb");
    gguf_mmap data_mmap(&file, 0, false);

    const int n_tensors = gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name   = gguf_get_tensor_name(ctx, i);
        const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);

        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);

        cur->data = static_cast<char *>(data_mmap.addr) + offset;

        // print first 10 elements
        const float * data = (const float *) cur->data;

        printf("%s data[:10] : ", name);
        for (int j = 0; j < MIN(10, ggml_nelements(cur)); ++j) {
            printf("%f ", data[j]);
        }
        printf("\n\n");
    }

    fprintf(stdout, "%s: ctx_data size: %zu\n", __func__, ggml_get_mem_size(ctx_data));

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stdout, "usage: %s data.gguf r|w\n", argv[0]);
        return -1;
    }

    const std::string fname(argv[1]);
    const std::string mode (argv[2]);

    GGML_ASSERT((mode == "r" || mode == "w" || mode == "q") && "mode must be r, w or q");

    if (mode == "w") {
        GGML_ASSERT(gguf_ex_write(fname) && "failed to write gguf file");
    } else if (mode == "r") {
        GGML_ASSERT(gguf_ex_read_0(fname) && "failed to read gguf file");
        GGML_ASSERT(gguf_ex_read_1(fname) && "failed to read gguf file");
        GGML_ASSERT(gguf_ex_read_2(fname) && "failed to read gguf file");
    } else if (mode == "q") {
        llama_model_quantize_params params = llama_model_quantize_default_params();
        llama_model_quantize(fname.c_str(), "quant.gguf", &params);
    }

    return 0;
}
