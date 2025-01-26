#ifdef LLAMA_USE_LLGUIDANCE

#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include "llguidance.h"

struct llama_sampler_llg {
    const llama_vocab * vocab;
    std::string grammar_kind;
    std::string grammar_data;
    LlgTokenizer *tokenizer;
    LlgConstraint *grammar;
    LlgMaskResult llg_res;
    bool has_llg_res;
};

static LlgConstraint *llama_sampler_llg_new(LlgTokenizer *tokenizer,
        const char * grammar_kind, const char * grammar_data) {
    LlgConstraintInit cinit;
    llg_constraint_init_set_defaults(&cinit, tokenizer);
    auto c = llg_new_constraint_any(&cinit, grammar_kind, grammar_data);
    if (llg_get_error(c)) {
        LOG_ERR("llg error: %s\n", llg_get_error(c));
        llg_free_constraint(c);
        return nullptr;
    }
    return c;
}

static const char * llama_sampler_llg_name(const llama_sampler * /*smpl*/) {
    return "llguidance";
}

static void llama_sampler_llg_accept_impl(llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        LlgCommitResult res;
        llg_commit_token(ctx->grammar, token, &res);
        ctx->has_llg_res = false;
    }
}

static void llama_sampler_llg_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        if (!ctx->has_llg_res) {
            if (llg_compute_mask(ctx->grammar, &ctx->llg_res) == 0) {
                ctx->has_llg_res = true;
            } else {
                LOG_ERR("llg error: %s\n", llg_get_error(ctx->grammar));
                llg_free_constraint(ctx->grammar);
                ctx->grammar = nullptr;
            }
        }
        if (ctx->has_llg_res) {
            if (ctx->llg_res.is_stop) {
                for (size_t i = 0; i < cur_p->size; ++i) {
                    if (!llama_vocab_is_eog(ctx->vocab, cur_p->data[i].id)) {
                        cur_p->data[i].logit = -INFINITY;
                    }
                }
            } else {
                const uint32_t *mask = ctx->llg_res.sample_mask;
                for (size_t i = 0; i < cur_p->size; ++i) {
                    auto token = cur_p->data[i].id;
                    if ((mask[token / 32] & (1 << (token % 32))) == 0) {
                        cur_p->data[i].logit = -INFINITY;
                    }
                }
            }
        }
    }
}

static void llama_sampler_llg_reset(llama_sampler * smpl) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (!ctx->grammar) {
        return;
    }

    auto * grammar_new = llama_sampler_llg_new(ctx->tokenizer, ctx->grammar_kind.c_str(), ctx->grammar_data.c_str());
    llg_free_constraint(ctx->grammar);
    ctx->grammar = grammar_new;
    ctx->has_llg_res = false;
}

static llama_sampler * llama_sampler_llg_clone(const llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_llg *) smpl->ctx;

    auto * result = llama_sampler_init_llg(ctx->vocab, nullptr, nullptr);

    // copy the state
    {
        auto * result_ctx = (llama_sampler_llg *) result->ctx;

        if (ctx->grammar) {
            result_ctx->grammar_kind = ctx->grammar_kind;
            result_ctx->grammar_data = ctx->grammar_data;
            result_ctx->grammar = llg_clone_constraint(ctx->grammar);
            result_ctx->tokenizer = llg_clone_tokenizer(ctx->tokenizer);
        }
    }

    return result;
}

static void llama_sampler_llg_free(llama_sampler * smpl) {
    const auto * ctx = (llama_sampler_llg *) smpl->ctx;

    if (ctx->grammar) {
        llg_free_constraint(ctx->grammar);
        llg_free_tokenizer(ctx->tokenizer);
    }

    delete ctx;
}

static llama_sampler_i llama_sampler_llg_i = {
    /* .name   = */ llama_sampler_llg_name,
    /* .accept = */ llama_sampler_llg_accept_impl,
    /* .apply  = */ llama_sampler_llg_apply,
    /* .reset  = */ llama_sampler_llg_reset,
    /* .clone  = */ llama_sampler_llg_clone,
    /* .free   = */ llama_sampler_llg_free,
};


static size_t llama_sampler_llg_tokenize_fn(const void *user_data,
                                const uint8_t *bytes,
                                size_t bytes_len,
                                uint32_t *output_tokens,
                                size_t output_tokens_len)
{
    const llama_vocab *vocab = (const llama_vocab *)user_data;
    int r = llama_tokenize(vocab, (const char *) bytes, bytes_len, 
        (int32_t*)output_tokens, output_tokens_len, false, true);
    if (r < 0)
        return -r;
    return r;
}

static LlgTokenizer *llama_sampler_llg_new_tokenizer(const llama_vocab * vocab) {
    // TODO store the tokenizer in the vocab somehow
    static const llama_vocab *vocab_cache;
    static LlgTokenizer *tokenizer_cache;

    if (vocab_cache == vocab) {
        return llg_clone_tokenizer(tokenizer_cache);
    }

    auto tok_eos = llama_vocab_eot(vocab);
    if (tok_eos == LLAMA_TOKEN_NULL)
        tok_eos = llama_vocab_eos(vocab);

    size_t vocab_size = llama_vocab_n_tokens(vocab);

    auto token_lens = new uint32_t[vocab_size];
    // we typically have ~7 bytes per token; let's go on the safe side here
    auto token_bytes_size = vocab_size * 16 + 1024 * 1024;
    auto token_bytes = new uint8_t[token_bytes_size];

    size_t offset = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        size_t max_token = 1024;
        if (token_bytes_size - offset < max_token) {
            GGML_ABORT("token_bytes buffer too small\n");
        }

        llama_token token = i;
        auto dp = (char *) token_bytes + offset;
        auto size = llama_detokenize(vocab, &token, 1, dp, max_token, false, false);
        if (size < 0) {
            GGML_ABORT("llama_detokenize failed\n");
        }
        if (size == 0) {
            size = llama_detokenize(vocab, &token, 1, dp + 1, max_token - 1, false, true);
            if (size < 0) {
                GGML_ABORT("llama_detokenize failed\n");
            }
            if (size != 0) {
                *dp = '\xff'; // special token prefix marker
                size += 1;
            }
        }

        token_lens[i] = size;
        offset += size;
    }


    LlgTokenizerInit tinit = {
        /* .vocab_size                         = */ (uint32_t)vocab_size,
        /* .tok_eos                            = */ (uint32_t)tok_eos,
        /* .token_lens                         = */ token_lens,
        /* .token_bytes                        = */ token_bytes,
        /* .tokenizer_json                     = */ nullptr,
        /* .tokenize_assumes_string            = */ false,
        /* .tokenize_fn                        = */ llama_sampler_llg_tokenize_fn,
        /* .use_approximate_greedy_tokenize_fn = */ false,
        /* .tokenize_user_data                 = */ vocab,
    };

    char error_buffer[1024];
    LlgTokenizer *tokenizer = llg_new_tokenizer(&tinit, error_buffer, sizeof(error_buffer));

    delete[] token_bytes;
    delete[] token_lens;

    if (tokenizer == nullptr) {
        LOG_ERR("llg tokenizer error: %s\n", error_buffer);
        return tokenizer;
    }

    if (tokenizer_cache) {
        llg_free_tokenizer(tokenizer_cache);
    }
    vocab_cache = vocab;
    tokenizer_cache = tokenizer;

    return tokenizer;
}

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab, 
        const char * grammar_kind, const char * grammar_data) {
    auto * ctx = new llama_sampler_llg;

    if (grammar_kind != nullptr && grammar_kind[0] != '\0') {
        auto tokenizer = llama_sampler_llg_new_tokenizer(vocab);
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ grammar_kind,
            /* .grammar_data = */ grammar_data,
            /* .tokenizer    = */ tokenizer,
            /* .grammar      = */ llama_sampler_llg_new(tokenizer, grammar_kind, grammar_data),
            /* .llg_res      = */ {},
            /* .has_llg_res  = */ false,
        };
    } else {
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ {},
            /* .grammar_data = */ {},
            /* .tokenizer    = */ nullptr,
            /* .grammar      = */ nullptr,
            /* .llg_res      = */ {},
            /* .has_llg_res  = */ false,
        };
    }

    return new llama_sampler {
        /* .iface = */ &llama_sampler_llg_i,
        /* .ctx   = */ ctx,
    };
}

#endif // LLAMA_USE_LLGUIDANCE
