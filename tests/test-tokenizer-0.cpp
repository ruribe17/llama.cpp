#include "llama.h"
#include "common.h"

#include <cstdio>
#include <string>
#include <map>
#include <vector>

static std::string llama_detokenize(llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        result += llama_token_to_str(ctx, tokens[i]);
    }
    return result;
}

static const std::map<std::string, std::vector<llama_token>> & k_tests() {
    static std::map<std::string, std::vector<llama_token>> _k_tests = {
        { "" ,                      { }, },
        { " ",                      {    259, }, },
        { "  ",                     {   1678, }, },
        { "   ",                    {    268, }, },
        { "\t",                     {  29871,     12, }, },
        { "\n",                     {  29871,     13, }, },
        { "\t\n",                   {  29871,     12,     13, }, },
        { "Hello world",            {  15043,   3186, }, },
        { " Hello world",           {  29871,  15043,   3186, }, },
        { "Hello World",            {  15043,   2787, }, },
        { " Hello World",           {  29871,  15043,   2787, }, },
        { " Hello World!",          {  29871,  15043,   2787,  29991, }, },
        { "Hello, world!",          {  15043,  29892,   3186,  29991, }, },
        { " Hello, world!",         {  29871,  15043,  29892,   3186,  29991, }, },
        { " this is 🦙.cpp",        {  29871,    445,    338,  29871,    243,    162,    169,    156,  29889,   8223, }, },
        { "w048 7tuijk dsdfhu",     {    281,  29900,  29946,  29947,  29871,  29955,   9161,  13535,  18031,   2176,   6905, }, },
        { "нещо на Български",      {   1538,   4851,    665,   1386,  29713,   1305, }, },
        { "កាន់តែពិសេសអាចខលចេញ",
                                    {  29871,  31849,  31324,  31934,    228,    162,    142,    228,    161,
                                         146,    228,    162,    133,    228,    161,    153,    228,    161,    186,
                                       31708,    228,    162,    132,  31708,    228,    161,    165,  31324,    228,
                                         161,    136,    228,    161,    132,    228,    161,    158,    228,    161,
                                         136,    228,    162,    132,    228,    161,    140, }, },
        { "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)",
                                    {  29871,    243,    162,    157,    131,    313,   8945,  29897,  29871,
                                         243,    162,    155,    185,  30722,    243,    162,    143,    174,  30598,
                                         313,  20787,    953,   3848,    275,  16125,    630,  29897,  29871,  31681,
                                         313,   6194,    953,  29877,   2397,    393,    756,    967,   1914,   5993,  29897, }, },
        { "Hello",                  {  15043, }, },
        { " Hello",                 {  29871,  15043, }, },
        { "  Hello",                {    259,  15043, }, },
        { "   Hello",               {   1678,  15043, }, },
        { "    Hello",              {    268,  15043, }, },
        { "    Hello\n    Hello",   {    268,  15043,     13,   1678,  15043, }, },
    };

    return _k_tests;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vocab-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    fprintf(stderr, "%s : reading vocab from: '%s'\n", __func__, fname.c_str());

    llama_model * model;
    llama_context * ctx;

    llama_backend_init(false);

    // load the vocab
    {
        auto lparams = llama_context_default_params();

        lparams.vocab_only = true;

        model = llama_load_model_from_file(fname.c_str(), lparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            return 1;
        }

        ctx = llama_new_context_with_model(model, lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            llama_free_model(model);
            return 1;
        }
    }

    const int n_vocab = llama_n_vocab(ctx);

    if (n_vocab != 32000) {
        fprintf(stderr, "%s : expected 32000 tokens, got %d\n", __func__, n_vocab);
        llama_free_model(model);
        llama_free(ctx);
        return 2;
    }

    bool success = true;

    for (const auto & test_kv : k_tests()) {
        const std::vector<llama_token> res_bos   = llama_tokenize(ctx, test_kv.first, true);
        const std::vector<llama_token> res_nobos = llama_tokenize(ctx, test_kv.first, false);

        fprintf(stderr, "%s : '%s' tokenized to '%s'\n", __func__, test_kv.first.c_str(), llama_detokenize(ctx, res_bos).c_str());

        bool correct = res_nobos.size() == test_kv.second.size() && res_bos.size() == res_nobos.size() + 1 && res_bos[0] == 1;

        for (int i = 0; i < (int) res_nobos.size() && correct; ++i) {
            if (test_kv.second[i] != res_bos[i + 1]) {
                correct = false;
            }
            if (test_kv.second[i] != res_nobos[i]) {
                correct = false;
            }
        }

        if (!correct) {
            fprintf(stderr, "%s : failed test:    '%s'\n", __func__, test_kv.first.c_str());
            fprintf(stderr, "%s : detokenized to: '%s' instead of '%s'\n", __func__,
                llama_detokenize(ctx, res_nobos).c_str(),
                llama_detokenize(ctx, test_kv.second).c_str());
            fprintf(stderr, "%s : expected tokens: ", __func__);
            for (const auto & t : test_kv.second) {
                fprintf(stderr, "%6d, ", t);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "%s : got tokens:      ", __func__);
            for (const auto & t : res_nobos) {
                fprintf(stderr, "%6d, ", t);
            }
            fprintf(stderr, "\n");

            success = false;
        }
    }

    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    return success ? 0 : 3;
}
