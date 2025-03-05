#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-kv-cache.h"
#include "llama-model.h"

#include <cassert>
#include <cmath>
#include <cstring>

static int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}

ggml_tensor * llm_graph_input_attn_i::get_kq_mask() {
    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

ggml_tensor * llm_graph_input_attn_i::get_kq_mask_swa() {
    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

ggml_tensor * llm_graph_input_attn_i::get_kq_mask_cross() {
    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);
    return nullptr;
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        const int64_t n_embd   = embd->ne[0];
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_token*ggml_element_size(pos));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (cur) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(cur->buffer));
        GGML_ASSERT(!ubatch->equal_seqs); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) cur->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[h*(n_tokens*n_tokens) + j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
                }
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (cur) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(cur->buffer));
        GGML_ASSERT(!ubatch->equal_seqs); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) cur->data;

        const int64_t n_kv = kv_self->n;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_kv; ++i) {
                    data[h*(n_kv*n_tokens) + j*n_kv + i] = llama_relative_position_bucket(kv_self->cells[i].pos, ubatch->pos[j], hparams.n_rel_attn_bkts, false);
                }
            }
        }
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    if (hparams.causal_attn || cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        //GGML_ASSERT(out_ids && "every model that can must skip unused outputs");

        if (!out_ids) {
            LLAMA_LOG_WARN("%s: 'out_ids' is not created\n", __func__);
        } else {
            const int64_t n_tokens = ubatch->n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
            int32_t * data = (int32_t *) out_ids->data;

            if (n_outputs == n_tokens) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[i] = i;
                }
            } else if (ubatch->output) {
                int32_t n_outputs = 0;
                for (int i = 0; i < n_tokens; ++i) {
                    if (ubatch->output[i]) {
                        data[n_outputs++] = i;
                    }
                }
                // the graph needs to have been passed the correct number of outputs
                GGML_ASSERT(n_outputs == n_outputs);
            } else if (n_outputs == 1) {
                // only keep last output
                data[0] = n_tokens - 1;
            } else {
                GGML_ASSERT(n_outputs == 0);
            }
        }
    }
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs       = ubatch->n_seqs;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens * n_tokens * ggml_element_size(mean));

        std::vector<uint64_t> sum(n_tokens, 0);

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch->seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch->equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == MEAN");

            sum[seq_id] += ubatch->n_seq_tokens;
        }

        std::vector<float> div(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const uint64_t s = sum[i];
            if (s > 0) {
                div[i] = 1.0f/float(s);
            }
        }

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch->seq_id[s][0];

            for (int i = 0; i < n_seq_tokens; ++i) {
                data[seq_id*n_tokens + s*n_seq_tokens + i] = div[seq_id];
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings && (
                cparams.pooling_type == LLAMA_POOLING_TYPE_CLS ||
                cparams.pooling_type == LLAMA_POOLING_TYPE_RANK)) {
        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs       = ubatch->n_seqs;

        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_tokens * ggml_element_size(cls));

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch->seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch->equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == CLS or RANK");

            for (int i = 0; i < n_seq_tokens; ++i) {
                const llama_pos pos = ubatch->pos[s*n_seq_tokens + i];

                if (pos == 0) {
                    data[seq_id] = s*n_seq_tokens + i;
                }
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_LAST) {
        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs       = ubatch->n_seqs;

        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_tokens * ggml_element_size(cls));

        std::vector<int> last_pos(n_tokens, -1);
        std::vector<int> last_row(n_tokens, -1);

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch->seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch->equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == LAST");

            for (int i = 0; i < n_seq_tokens; ++i) {
                const llama_pos pos = ubatch->pos[s*n_seq_tokens + i];

                if (pos >= last_pos[seq_id]) {
                    last_pos[seq_id] = pos;
                    last_row[seq_id] = s*n_seq_tokens + i;
                }
            }
        }

        for (int i = 0; i < n_tokens; ++i) {
            if (last_row[i] >= 0) {
                data[i] = last_row[i];
            }
        }
    }
}

void llm_graph_input_s_copy::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_kv = kv_self->n;

    if (cur) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cur->buffer));
        int32_t * data = (int32_t *) cur->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t  cell_id = i + kv_self->head;

            //////////////////////////////////////////////
            // TODO: this should not mutate the KV cache !
            llama_kv_cell & kv_cell = const_cast<class llama_kv_cache_recurrent *>(kv_self)->cells[i];

            // prevent out-of-bound sources
            if (kv_cell.src < 0 || (uint32_t) kv_cell.src >= kv_self->size) {
                kv_cell.src = cell_id;
            }

            data[i] = kv_cell.src;

            // TODO: do not mutate the KV cache
            // ensure copy only happens once
            if (kv_cell.src != (int32_t) cell_id) {
                kv_cell.src = cell_id;
            }
        }
    }
}

void llm_graph_input_s_mask::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_kv = kv_self->n;

    if (cur) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cur->buffer));
        float * data = (float *) cur->data;

        // clear unused states
        for (int i = 0; i < n_kv; ++i) {
            const uint32_t  cell_id = i + kv_self->head;

            //////////////////////////////////////////////
            // TODO: this should not mutate the KV cache !
            llama_kv_cell & kv_cell = const_cast<class llama_kv_cache_recurrent *>(kv_self)->cells[i];

            data[i] = (float) (kv_cell.src >= 0);

            // only clear once
            if (kv_cell.src < 0) {
                kv_cell.src = cell_id;
            }
        }
    }
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cur && cross->t_embd) {
        assert(cur->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cur, cross->v_embd, 0, ggml_nbytes(cur));
    }
}

void llm_graph_input_attn_base::set_input(const llama_ubatch * ubatch) {
    if (kq_mask) {
        if (cparams.causal_attn) {
            const int64_t n_kv         = ubatch->n_tokens;
            const int64_t n_tokens     = ubatch->n_tokens;
            const int64_t n_seq_tokens = ubatch->n_seq_tokens;
            const int64_t n_seqs       = ubatch->n_seqs;

            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
            float * data = (float *) kq_mask->data;

            for (int h = 0; h < 1; ++h) {
                for (int s1 = 0; s1 < n_seqs; ++s1) {
                    const llama_seq_id seq_id = ubatch->seq_id[s1][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const int32_t tj = s1*n_seq_tokens + j;

                        for (int s0 = 0; s0 < n_seqs; ++s0) {
                            for (int i = 0; i < n_seq_tokens; ++i) {
                                const int32_t ti = s0*n_seq_tokens + i;
                                float f = -INFINITY;

                                for (int s = 0; s < ubatch->n_seq_id[s0]; ++s) {
                                    if (ubatch->seq_id[s0][s] == seq_id && ubatch->pos[ti] <= ubatch->pos[tj]) {
                                        if (hparams.use_alibi) {
                                            f = -std::abs(ubatch->pos[ti] - ubatch->pos[tj]);
                                        } else {
                                            f = 0.0f;
                                        }
                                        break;
                                    }
                                }

                                data[h*(n_kv*n_tokens) + tj*n_kv + ti] = f;
                            }
                        }
                    }
                }
            }
        } else {
            const int64_t n_tokens     = ubatch->n_tokens;
            const int64_t n_seq_tokens = ubatch->n_seq_tokens;
            const int64_t n_seqs       = ubatch->n_seqs;
            const int64_t n_stride     = ubatch->n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));

            float * data = (float *) kq_mask->data;

            for (int h = 0; h < 1; ++h) {
                for (int s1 = 0; s1 < n_seqs; ++s1) {
                    const llama_seq_id seq_id = ubatch->seq_id[s1][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const int32_t tj = s1*n_seq_tokens + j;

                        for (int s0 = 0; s0 < n_seqs; ++s0) {
                            for (int i = 0; i < n_seq_tokens; ++i) {
                                const int32_t ti = s0*n_seq_tokens + i;
                                float f = -INFINITY;

                                for (int s = 0; s < ubatch->n_seq_id[s0]; ++s) {
                                    if (ubatch->seq_id[s0][s] == seq_id) {
                                        if (hparams.use_alibi) {
                                            f = -std::abs(ubatch->pos[ti] - ubatch->pos[tj]);
                                        } else {
                                            f = 0.0f;
                                        }
                                        break;
                                    }
                                }

                                data[h*(n_tokens*n_tokens) + tj*n_stride + ti] = f;
                            }
                        }

                        for (int i = n_tokens; i < n_stride; ++i) {
                            data[h*(n_tokens*n_tokens) + tj*n_stride + i] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

void llm_graph_input_attn_kv_self::set_input(const llama_ubatch * ubatch) {
    if (self_kq_mask || self_kq_mask_swa) {
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        if (cparams.causal_attn) {
            const int64_t n_kv         = kv_self->n;
            const int64_t n_tokens     = ubatch->n_tokens;
            const int64_t n_seq_tokens = ubatch->n_seq_tokens;
            const int64_t n_seqs       = ubatch->n_seqs;

            float * data     = nullptr;
            float * data_swa = nullptr;

            if (self_kq_mask) {
                GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
                data = (float *) self_kq_mask->data;
            }

            if (self_kq_mask_swa) {
                GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
                data_swa = (float *) self_kq_mask_swa->data;
            }

            // For causal attention, use only the previous KV cells
            // of the correct sequence for each token of the ubatch.
            // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
            for (int h = 0; h < 1; ++h) {
                for (int s = 0; s < n_seqs; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[s][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const llama_pos pos = ubatch->pos[s*n_seq_tokens + j];

                        for (int i = 0; i < n_kv; ++i) {
                            float f;
                            if (!kv_self->cells[i].has_seq_id(seq_id) || kv_self->cells[i].pos > pos) {
                                f = -INFINITY;
                            } else {
                                if (hparams.use_alibi) {
                                    f = -std::abs(kv_self->cells[i].pos - pos);
                                } else {
                                    f = 0.0f;
                                }
                            }

                            if (data) {
                                data[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }

                            // may need to cut off old tokens for sliding window
                            if (data_swa) {
                                if (pos - kv_self->cells[i].pos >= (int32_t)hparams.n_swa) {
                                    f = -INFINITY;
                                }
                                data_swa[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }
                        }
                    }
                }

                if (data) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }

                if (data_swa) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data_swa[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }
            }
        } else {
            const int64_t n_tokens     = ubatch->n_tokens;
            const int64_t n_seq_tokens = ubatch->n_seq_tokens;
            const int64_t n_seqs       = ubatch->n_seqs;
            // when using kv cache, the mask needs to match the kv cache size
            const int64_t n_stride     = n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));

            float * data = (float *) self_kq_mask->data;

            for (int h = 0; h < 1; ++h) {
                for (int s1 = 0; s1 < n_seqs; ++s1) {
                    const llama_seq_id seq_id = ubatch->seq_id[s1][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const int32_t tj = s1*n_seq_tokens + j;

                        for (int s0 = 0; s0 < n_seqs; ++s0) {
                            for (int i = 0; i < n_seq_tokens; ++i) {
                                const int32_t ti = s0*n_seq_tokens + i;
                                float f = -INFINITY;

                                for (int s = 0; s < ubatch->n_seq_id[s0]; ++s) {
                                    if (ubatch->seq_id[s0][s] == seq_id) {
                                        if (hparams.use_alibi) {
                                            f = -std::abs(ubatch->pos[ti] - ubatch->pos[tj]);
                                        } else {
                                            f = 0.0f;
                                        }
                                        break;
                                    }
                                }

                                data[h*(n_tokens*n_tokens) + tj*n_stride + ti] = f;
                            }
                        }

                        for (int i = n_tokens; i < n_stride; ++i) {
                            data[h*(n_tokens*n_tokens) + tj*n_stride + i] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

void llm_graph_input_attn_dec::set_input(const llama_ubatch * ubatch) {
    inp_kv_self->set_input(ubatch);

    if (cross_kq_mask) {
        const int64_t n_enc    = cross_kq_mask->ne[0];
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
        GGML_ASSERT(!ubatch->equal_seqs); // TODO: use ubatch->n_seqs instead of failing

        float * data = (float *) cross_kq_mask->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_enc; ++i) {
                    float f = -INFINITY;
                    for (int s = 0; s < ubatch->n_seq_id[j]; ++s) {
                        const llama_seq_id seq_id = ubatch->seq_id[j][s];
                        if (cross->seq_ids_enc[i].find(seq_id) != cross->seq_ids_enc[i].end()) {
                            f = 0.0f;
                        }
                    }
                    data[h*(n_enc*n_tokens) + j*n_enc + i] = f;
                }
            }

            for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                for (int j = 0; j < n_enc; ++j) {
                    data[h*(n_enc*n_tokens) + i*n_enc + j] = -INFINITY;
                }
            }
        }
    }
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    model            (params.model),
    arch             (model.arch),
    hparams          (model.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer),
    n_rot            (hparams.n_rot),
    n_ctx            (cparams.n_ctx),
    n_ctx_per_seq    (cparams.n_ctx / cparams.n_seq_max),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    ctx0             (params.ctx),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    backends         (params.backends),
    cvec             (params.cvec),
    loras            (params.loras),
    memory           (params.memory),
    cross            (params.cross),
    res              (std::make_unique<llm_graph_result>()) {
    }

int64_t llm_graph_context::n_pos_per_token() const {
    return arch == LLM_ARCH_QWEN2VL ? 4 : 1;
}

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }

    if (!cparams.offload_kqv) {
        if (strcmp(name, "kqv_merged_cont") == 0) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
    // FIXME: fix in ggml_backend_sched
    const bool full_offload = model.params.n_gpu_layers > (int) model.hparams.n_layer;
    if (ubatch.n_tokens < 32 || full_offload) {
        if (il != -1 && strcmp(name, "norm") == 0) {
            const auto & dev_layer = model.dev_layer(il);
            for (const auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == dev_layer) {
                    if (ggml_backend_supports_op(backend.get(), cur)) {
                        ggml_backend_sched_set_tensor_backend(sched, cur, backend.get());
                    }
                }
            }
        }
    }
}

ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                // Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
                int64_t split_point = cur->ne[0] / 2;
                ggml_tensor * x0 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, split_point, cur->ne[1], cur->nb[1], 0));
                ggml_tensor * x1 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, split_point, cur->ne[1], cur->nb[1], split_point * ggml_element_size(cur)));

                x0 = ggml_silu(ctx0, x0);
                cb(cur, "ffn_silu", il);

                cur = ggml_mul(ctx0, x0, x1);
                cb(cur, "ffn_mul", il);
            } break;
    }

    if (type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
                bool   scale_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il) const {
    int64_t n_embd = cur->ne[0];
    int64_t n_tokens = cur->ne[1];

    ggml_tensor * logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    ggml_tensor * weights = ggml_get_rows(ctx0,
            ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(up, "ffn_moe_up", il);

    ggml_tensor * gate = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(gate, "ffn_moe_gate", il);

    switch (type_op) {
        case LLM_FFN_SILU:
            {
                gate = ggml_silu(ctx0, gate);
                cb(gate, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            {
                gate = ggml_gelu(ctx0, gate);
                cb(gate, "ffn_moe_gelu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    ggml_tensor * par = ggml_mul(ctx0, up, gate); // [n_ff, n_expert_used, n_tokens]
    cb(par, "ffn_moe_gate_par", il);

    ggml_tensor * experts = build_lora_mm_id(down_exps, par, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    experts = ggml_mul(ctx0, experts, weights);

    // aggregate experts
    ggml_tensor * moe_out = nullptr;
    for (int i = 0; i < n_expert_used; ++i) {
        ggml_tensor * cur_expert = ggml_view_2d(ctx0, experts, n_embd, n_tokens,
                experts->nb[2], i*experts->nb[1]);

        if (i == 0) {
            moe_out = cur_expert;
        } else {
            moe_out = ggml_add(ctx0, moe_out, cur_expert);
        }
    }

    if (n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd = hparams.n_embd;

    auto inp = std::make_shared<llm_graph_input_embd>();

    auto & cur = inp->cur;

    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        //cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }
    } else {
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ubatch.n_tokens);
        cur = inp->embd;
        ggml_set_input(inp->embd);
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    //cb(cur, "inp_embd", -1);

    cb(inp->cur, "inp_embd", -1);

    res->add_input(inp);

    return inp->cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_shared<llm_graph_input_pos>(n_pos_per_token());

    inp->pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens*n_pos_per_token());
    ggml_set_input(inp->pos);

    res->add_input(inp);

    return inp->pos;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    auto inp = std::make_shared<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    inp->out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(inp->out_ids);

    res->add_input(inp);

    return inp->out_ids;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_shared<llm_graph_input_mean>(cparams);

    inp->mean = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_input(inp->mean);

    res->add_input(inp);

    return inp->mean;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_shared<llm_graph_input_cls>(cparams);

    inp->cls = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->cls);

    res->add_input(inp);

    return inp->cls;
}

ggml_tensor * llm_graph_context::build_inp_s_copy() const {
    const llama_kv_cache_recurrent * kv_self = static_cast<const llama_kv_cache_recurrent *>(memory);

    auto inp = std::make_shared<llm_graph_input_s_copy>(kv_self);

    const auto n_kv = kv_self->n;

    inp->cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
    //cb(inp.cur, "inp_s_copy", -1);
    ggml_set_input(inp->cur);

    res->add_input(inp);

    return inp->cur;
}

ggml_tensor * llm_graph_context::build_inp_s_mask() const {
    const llama_kv_cache_recurrent * kv_self = static_cast<const llama_kv_cache_recurrent *>(memory);

    auto inp = std::make_shared<llm_graph_input_s_mask>(kv_self);

    const auto n_kv = kv_self->n;

    inp->cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_kv);
    //cb(inp->cur, "inp_s_mask", -1);
    ggml_set_input(inp->cur);

    res->add_input(inp);

    return inp->cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_shared<llm_graph_input_cross_embd>(cross);

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    inp->cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return inp->cur;
    //}

    const auto n_embd = cross->t_embd ? cross->t_embd->ne[0] : hparams.n_embd;
    const auto n_enc  = cross->t_embd ? cross->t_embd->ne[1] : hparams.n_ctx_train;

    inp->cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(inp->cur);

    res->add_input(inp);

    return inp->cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_shared<llm_graph_input_pos_bucket>(hparams);

    inp->cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(inp->cur);

    res->add_input(inp);

    return inp->cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const llama_kv_cache_unified * kv_self = static_cast<const llama_kv_cache_unified *>(memory);

    auto inp = std::make_shared<llm_graph_input_pos_bucket_kv>(hparams, kv_self);

    const auto n_kv = kv_self->n;

    inp->cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(inp->cur);

    res->add_input(inp);

    return inp->cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_permute(ctx0, pos_bias, 2, 0, 1, 3);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_cont(ctx0, pos_bias);
    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_cgraph * gf,
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
             bool      v_trans,
             float     kq_scale) const {
  //const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
  //const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

  //const int64_t n_head    = hparams.n_head(il);
  //const int64_t n_head_kv = hparams.n_head_kv(il);

  //const auto & n_embd_head_k = hparams.n_embd_head_k;
  //const auto & n_embd_head_v = hparams.n_embd_head_v;

    const auto n_embd_head_v = v_trans ? v->ne[1] : v->ne[0];

    const auto n_tokens = q->ne[1];
    const auto n_head   = q->ne[2];
    const auto n_kv     = k->ne[1];

    ggml_tensor * cur;

    // TODO: replace hardcoded padding with ggml-provided padding
    if (cparams.flash_attn && (n_kv % 256 == 0) && kq_b == nullptr) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);

        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, n_embd_head_v*n_head, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplyer of 0.08838834764831845
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, 0.08838834764831845f/30.0f));
            kq = ggml_scale(ctx0, kq, 30);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            kq = ggml_tanh (ctx0, kq);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);

        ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_head_v*n_head, n_tokens);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_base_ptr llm_graph_context::build_attn_inp_base(
                      bool   causal,
                      bool   swa) const {
    auto inp = std::make_shared<llm_graph_input_attn_base>(hparams, cparams);

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    GGML_UNUSED(causal);
    GGML_UNUSED(swa);

    inp->kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    //cb(inp_kq_mask, "KQ_mask", -1);
    ggml_set_input(inp->kq_mask);

    inp->kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->kq_mask, GGML_TYPE_F16) : inp->kq_mask;

    res->add_input(inp);

    return inp;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_base * inp,
        ggml_cgraph * gf,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    //cb(k, "k", il);

    ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);
    //cb(k, "v", il);

    ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_b, kq_mask, false, kq_scale);

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_kv_self_ptr llm_graph_context::build_attn_inp_kv_self(
                bool   causal,
                bool   swa) const {
    const llama_kv_cache_unified * kv_self = static_cast<const llama_kv_cache_unified *>(memory);

    auto inp = std::make_shared<llm_graph_input_attn_kv_self>(hparams, cparams, kv_self);

    const auto n_kv = kv_self->n;

    inp->self_kq_mask = causal
        ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    //cb(inp->self_kq_mask, "KQ_mask", -1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;

    if (swa) {
        GGML_ASSERT(hparams.n_swa > 0);

        inp->self_kq_mask_swa = causal
            ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
            : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        //cb(inp->self_kq_mask_swa, "KQ_mask_swa", -1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    }

    res->add_input(inp);

    return inp;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_self * inp,
        ggml_cgraph * gf,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const llama_kv_cache_unified * kv_self = static_cast<const llama_kv_cache_unified *>(memory);
    const auto & n_ctx = cparams.n_ctx;

    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    const auto n_tokens = q_cur->ne[2];

    const bool v_trans = !cparams.flash_attn;

    // store to KV cache
    {
        GGML_ASSERT(!kv_self->recurrent);

        const auto kv_head = kv_self->head;

        GGML_ASSERT(kv_self->size == n_ctx);

        ggml_tensor * k_cache_view = ggml_view_1d(ctx0, kv_self->k_l[il], n_tokens*n_embd_k_gqa, ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa)*kv_head);
        //cb(k_cache_view, "k_cache_view", il);

        // note: storing RoPE-ed version of K in the KV cache
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, k_cur, k_cache_view));

        v_cur = ggml_reshape_2d(ctx0, v_cur, n_embd_v_gqa, n_tokens);

        ggml_tensor * v_cache_view = nullptr;

        if (!v_trans) {
            v_cache_view = ggml_view_1d(ctx0, kv_self->v_l[il], n_tokens*n_embd_v_gqa, ggml_row_size(kv_self->v_l[il]->type, n_embd_v_gqa)*kv_head);
        } else {
            // note: the V cache is transposed when not using flash attention
            v_cache_view = ggml_view_2d(ctx0, kv_self->v_l[il], n_tokens, n_embd_v_gqa,
                    (  n_ctx)*ggml_element_size(kv_self->v_l[il]),
                    (kv_head)*ggml_element_size(kv_self->v_l[il]));

            v_cur = ggml_transpose(ctx0, v_cur);
        }
        //cb(v_cache_view, "v_cache_view", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, v_cur, v_cache_view));
    }

    // TODO: improve
    bool is_sliding = false;

    switch (arch) {
        case LLM_ARCH_COHERE2:
            {
                const int32_t sliding_window_pattern = 4;
                is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);
            } break;
        case LLM_ARCH_GEMMA2:
            {
                const int32_t sliding_window_pattern = 2;
                is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);
            } break;
        case LLM_ARCH_GEMMA3:
            {
                const int32_t sliding_window_pattern = 6;
                is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);
            } break;
        case LLM_ARCH_PHI3:
            {
                is_sliding = hparams.n_swa > 0;
            } break;
        default:
            {
                is_sliding = false;
            }
    };

    const auto & kq_mask = is_sliding ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    const auto n_kv = kv_self->n;

    const int64_t n_head_kv = hparams.n_head_kv(il);

    const auto & n_embd_head_k = hparams.n_embd_head_k;
    const auto & n_embd_head_v = hparams.n_embd_head_v;

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k =
        ggml_view_3d(ctx0, kv_self->k_l[il],
                n_embd_head_k, n_kv, n_head_kv,
                ggml_row_size(kv_self->k_l[il]->type, n_embd_k_gqa),
                ggml_row_size(kv_self->k_l[il]->type, n_embd_head_k),
                0);
    //cb(k, "k", il);

    ggml_tensor * v = !v_trans ?
        ggml_view_3d(ctx0, kv_self->v_l[il],
                n_embd_head_v, n_kv, n_head_kv,
                ggml_row_size(kv_self->v_l[il]->type, n_embd_v_gqa),
                ggml_row_size(kv_self->v_l[il]->type, n_embd_head_v),
                0) :
        ggml_view_3d(ctx0, kv_self->v_l[il],
                n_kv, n_embd_head_v, n_head_kv,
                ggml_element_size(kv_self->v_l[il])*n_ctx,
                ggml_element_size(kv_self->v_l[il])*n_ctx*n_embd_head_v,
                0);

    ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_b, kq_mask, v_trans, kq_scale);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_dec_ptr llm_graph_context::build_attn_inp_dec(
                    bool   causal,
                    bool   swa) const {
    auto inp_kv_self = build_attn_inp_kv_self(causal, swa);

    auto inp = std::make_shared<llm_graph_input_attn_dec>(std::move(inp_kv_self), cross);

    const int32_t n_enc = cross->t_embd ? cross->t_embd->ne[1] : hparams.n_ctx_train;

    inp->cross_kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_enc, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->cross_kq_mask, GGML_TYPE_F16) : inp->cross_kq_mask;

    res->add_input(inp);

    return inp;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_dec * inp,
        ggml_cgraph * gf,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    //cb(k, "k", il);

    ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);
    //cb(k, "v", il);

    ggml_tensor * cur = build_attn_mha(gf, q, k, v, kq_b, kq_mask, false, kq_scale);

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_copy_mask_state(
         ggml_cgraph * gf,
         ggml_tensor * s,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
             int32_t   n_state,
             int32_t   n_seqs) const {
    const llama_kv_cache_recurrent * kv_self = static_cast<const llama_kv_cache_recurrent *>(memory);

    const auto n_kv    = kv_self->n;
    const auto kv_head = kv_self->head;

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, n_state, kv_self->size);

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between kv_head and kv_head + n_kv
    // this shrinks the tensors's ne[1] to n_kv
    states = ggml_get_rows(ctx0, states, state_copy);

    // clear states of sequences which are starting at the beginning of this batch
    // FIXME: zero-out NANs?
    states = ggml_mul(ctx0, states, state_mask);

    // copy states which won't be changed further (between n_seqs and n_kv)
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            ggml_view_1d(ctx0, states, n_state*(n_kv - n_seqs), (n_seqs          )*n_state*ggml_element_size(states)),
            ggml_view_1d(ctx0, s,      n_state*(n_kv - n_seqs), (kv_head + n_seqs)*n_state*ggml_element_size(s))));

    // the part of the states that will be used and modified
    return ggml_view_2d(ctx0, states, n_state, n_seqs, states->nb[1], 0);
}

// TODO: split
ggml_tensor * llm_graph_context::build_mamba_layer(
         ggml_cgraph * gf,
         ggml_tensor * cur,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il) const {
    const llama_kv_cache_recurrent * kv_self = static_cast<const llama_kv_cache_recurrent *>(memory);

    const auto kv_head = kv_self->head;

    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;
    const int64_t n_seqs  = ubatch.n_seqs;
    // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
    const bool ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;
    // Use the same RMS norm as the final layer norm
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs);
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * conv_states_all = kv_self->k_l[il];
    ggml_tensor * ssm_states_all  = kv_self->v_l[il];

    // (ab)using the KV cache to store the states
    ggml_tensor * conv = build_copy_mask_state(
            gf, conv_states_all, state_copy, state_mask,
            hparams.n_embd_k_s(), n_seqs);
    conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);
    ggml_tensor * ssm = build_copy_mask_state(
            gf, ssm_states_all, state_copy, state_mask,
            hparams.n_embd_v_s(), n_seqs);
    ssm = ggml_reshape_3d(ctx0, ssm, d_state, d_inner, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * xz = build_lora_mm(model.layers[il].ssm_in, cur);
    // split the above in two
    // => {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * x = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
    ggml_tensor * z = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner*ggml_element_size(xz));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);

        // copy last (d_conv - 1) columns back into the state cache
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens*(conv_x->nb[0]));

        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, last_conv,
                ggml_view_1d(ctx0, conv_states_all,
                    (d_conv - 1)*(d_inner)*(n_seqs),
                    kv_head*(d_conv - 1)*(d_inner)*ggml_element_size(conv_states_all))));

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        x = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);

        // bias
        x = ggml_add(ctx0, x, model.layers[il].ssm_conv1d_b);

        x = ggml_silu(ctx0, x);
    }

    // ssm
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        ggml_tensor * x_db = build_lora_mm(model.layers[il].ssm_x, x);
        // split
        ggml_tensor * dt = ggml_view_3d(ctx0, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
        ggml_tensor * B  = ggml_view_3d(ctx0, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*dt_rank);
        ggml_tensor * C  = ggml_view_3d(ctx0, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*(dt_rank+d_state));

        // Some Mamba variants (e.g. FalconMamba) apply RMS norm in B, C & Dt layers
        if (ssm_dt_b_c_rms) {
            dt = ggml_rms_norm(ctx0, dt, norm_rms_eps);
            B = ggml_rms_norm(ctx0, B, norm_rms_eps);
            C = ggml_rms_norm(ctx0, C, norm_rms_eps);
        }

        // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = build_lora_mm(model.layers[il].ssm_dt, dt);
        dt = ggml_add(ctx0, dt, model.layers[il].ssm_dt_b);

        // Custom operator to optimize the parallel associative scan
        // as described in the Annex D of the Mamba paper.
        // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
        ggml_tensor * y_ssm = ggml_ssm_scan(ctx0, ssm, x, dt, model.layers[il].ssm_a, B, C);

        // store last states
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0,
                ggml_view_1d(ctx0, y_ssm, d_state*d_inner*n_seqs, x->nb[3]),
                ggml_view_1d(ctx0, ssm_states_all, d_state*d_inner*n_seqs, kv_head*d_state*d_inner*ggml_element_size(ssm_states_all))));

        ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[1], x->nb[2], 0);

        // TODO: skip computing output earlier for unused tokens

        // {d_inner, n_seq_tokens, n_seqs} * {d_inner} => {d_inner, n_seq_tokens, n_seqs}
        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
        y = ggml_mul(ctx0, y, ggml_silu(ctx0, ggml_cont(ctx0, z)));

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = build_lora_mm(model.layers[il].ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
    //cb(cur, "mamba_out", il);

    return cur;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
         ggml_cgraph * gf,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il) const {
    const llama_kv_cache_recurrent * kv_self = static_cast<const llama_kv_cache_recurrent *>(memory);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = kv_self->k_l[il];

    ggml_tensor * token_shift = build_copy_mask_state(
            gf, token_shift_all, state_copy, state_mask,
            hparams.n_embd_k_s(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const llama_kv_cache_recurrent * kv_self = static_cast<const llama_kv_cache_recurrent *>(memory);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = kv_self->head;

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, kv_self->k_l[il], hparams.n_embd_k_s() * n_seqs, hparams.n_embd_k_s() * kv_head * ggml_element_size(kv_self->k_l[il]))
    );
}

ggml_tensor * llm_graph_context::build_rwkv6_time_mix(
         ggml_cgraph * gf,
         ggml_tensor * cur,
         ggml_tensor * x_prev,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il) const {
    const llama_kv_cache_recurrent * kv_self = static_cast<const llama_kv_cache_recurrent *>(memory);

    const auto n_tokens = ubatch.n_tokens;
    const auto n_seqs = ubatch.n_seqs;
    const auto n_embd = hparams.n_embd;
    const auto head_size = hparams.wkv_head_size;
    const auto n_head = n_embd / head_size;
    const auto n_head_kv = hparams.n_head_kv(il);

    const auto kv_head = kv_self->head;

    const auto & layer = model.layers[il];

    bool is_qrwkv = layer.time_mix_first == nullptr;

    ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
    ggml_tensor * xxx = ggml_add(ctx0, ggml_mul(ctx0, sx, layer.time_mix_lerp_x), cur);

    xxx = ggml_reshape_4d(
        ctx0,
        ggml_tanh(
            ctx0,
            ggml_mul_mat(ctx0, layer.time_mix_w1, xxx)
        ),
        layer.time_mix_w1->ne[1] / 5, 1, 5, n_tokens
    );

    xxx = ggml_cont(ctx0, ggml_permute(ctx0, xxx, 0, 1, 3, 2));

    xxx = ggml_mul_mat(
        ctx0,
        ggml_reshape_4d(
            ctx0,
            layer.time_mix_w2,
            layer.time_mix_w2->ne[0], layer.time_mix_w2->ne[1], 1, 5
        ),
        xxx
    );

    ggml_tensor *xw, *xk, *xv, *xr, *xg;
    if (layer.time_mix_lerp_fused) {
        // fusing these weights makes some performance improvement
        sx  = ggml_reshape_3d(ctx0, sx,  n_embd, 1, n_tokens);
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        xxx = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xxx, layer.time_mix_lerp_fused), sx), cur);
        xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
    } else {
        // for backward compatibility
        xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));

        xw = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xw, layer.time_mix_lerp_w), sx), cur);
        xk = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xk, layer.time_mix_lerp_k), sx), cur);
        xv = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xv, layer.time_mix_lerp_v), sx), cur);
        xr = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xr, layer.time_mix_lerp_r), sx), cur);
        xg = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xg, layer.time_mix_lerp_g), sx), cur);
    }

    ggml_tensor * r = build_lora_mm(layer.time_mix_receptance, xr);
    ggml_tensor * k = build_lora_mm(layer.time_mix_key,        xk);
    ggml_tensor * v = build_lora_mm(layer.time_mix_value,      xv);
    if (layer.time_mix_receptance_b) {
        r = ggml_add(ctx0, r, layer.time_mix_receptance_b);
    }
    if (layer.time_mix_key_b) {
        k = ggml_add(ctx0, k, layer.time_mix_key_b);
    }
    if (layer.time_mix_value_b) {
        v = ggml_add(ctx0, v, layer.time_mix_value_b);
    }

    ggml_tensor * g = build_lora_mm(layer.time_mix_gate, xg);
    if (is_qrwkv) {
        g = ggml_sigmoid(ctx0, g);
    } else {
        g = ggml_silu(ctx0, g);
    }

    if (n_head_kv != 0 && n_head_kv != n_head) {
        GGML_ASSERT(n_head % n_head_kv == 0);
        k = ggml_reshape_4d(ctx0, k, head_size, 1, n_head_kv, n_tokens);
        v = ggml_reshape_4d(ctx0, v, head_size, 1, n_head_kv, n_tokens);
        ggml_tensor * tmp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_size, n_head / n_head_kv, n_head_kv, n_tokens);
        k = ggml_repeat(ctx0, k, tmp);
        v = ggml_repeat(ctx0, v, tmp);
    }

    k = ggml_reshape_3d(ctx0, k, head_size, n_head, n_tokens);
    v = ggml_reshape_3d(ctx0, v, head_size, n_head, n_tokens);
    r = ggml_reshape_3d(ctx0, r, head_size, n_head, n_tokens);

    ggml_tensor * w = ggml_mul_mat(
        ctx0,
        layer.time_mix_decay_w2,
        ggml_tanh(
            ctx0,
            ggml_mul_mat(ctx0, layer.time_mix_decay_w1, xw)
        )
    );

    w = ggml_add(ctx0, w, layer.time_mix_decay);
    w = ggml_exp(ctx0, ggml_neg(ctx0, ggml_exp(ctx0, w)));
    w = ggml_reshape_3d(ctx0, w, head_size, n_head, n_tokens);

    if (is_qrwkv) {
        // k = k * (1 - w)
        k = ggml_sub(ctx0, k, ggml_mul(ctx0, k, w));
    }

    ggml_tensor * wkv_state = build_copy_mask_state(
            gf, kv_self->v_l[il], state_copy, state_mask,
            hparams.n_embd_v_s(), n_seqs);

    ggml_tensor * wkv_output;
    if (is_qrwkv) {
        wkv_output = ggml_gated_linear_attn(ctx0, k, v, r, w, wkv_state, pow(head_size, -0.5f));
    } else {
        wkv_output = ggml_rwkv_wkv6(ctx0, k, v, r, layer.time_mix_first, w, wkv_state);
    }
    cur = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
    wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

    ggml_build_forward_expand(
        gf,
        ggml_cpy(
            ctx0,
            wkv_state,
            ggml_view_1d(
                ctx0,
                kv_self->v_l[il],
                hparams.n_embd_v_s() * n_seqs,
                hparams.n_embd_v_s() * kv_head * ggml_element_size(kv_self->v_l[il])
            )
        )
    );

    if (!is_qrwkv) {
        // group norm with head_count groups
        cur = ggml_reshape_3d(ctx0, cur, n_embd / n_head, n_head, n_tokens);
        cur = ggml_norm(ctx0, cur, 64e-5f);

        // Convert back to regular vectors.
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    } else {
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    }

    cur = ggml_mul(ctx0, cur, g);
    cur = build_lora_mm(layer.time_mix_output, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_rwkv_channel_mix(
    const llama_layer * layer,
    ggml_tensor * cur,
    ggml_tensor * x_prev,
    llm_arch arch) const {
    ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
    switch (arch) {
        case LLM_ARCH_RWKV6:
        {
            ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);
            ggml_tensor * xr = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_r), cur);

            ggml_tensor * r = ggml_sigmoid(ctx0, build_lora_mm(layer->channel_mix_receptance, xr));
            ggml_tensor * k = ggml_sqr(
                    ctx0,
                    ggml_relu(
                        ctx0,
                        build_lora_mm(layer->channel_mix_key, xk)
                        )
                );
            cur = ggml_mul(ctx0, r, build_lora_mm(layer->channel_mix_value, k));
        } break;
        default:
            GGML_ABORT("fatal error");
    }

    return cur;
}

void llm_graph_context::build_pooling(
        ggml_cgraph * gf,
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                inp = ggml_get_rows(ctx0, inp, inp_cls);

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                GGML_ASSERT(cls   != nullptr);
                GGML_ASSERT(cls_b != nullptr);

                cur = ggml_add (ctx0, ggml_mul_mat(ctx0, cls, inp), cls_b);
                cur = ggml_tanh(ctx0, cur);

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                if (cls_out) {
                    GGML_ASSERT(cls_out_b != nullptr);

                    cur = ggml_add (ctx0, ggml_mul_mat(ctx0, cls_out, cur), cls_out_b);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

