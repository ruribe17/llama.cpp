#include "llama.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

enum outetts_version {
    OUTETTS_V0_2,
    OUTETTS_V0_3,
};

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

static void save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate) {
    std::ofstream file(fname, std::ios::binary);
    if (!file) {
        printf("%s: Failed to open file '%s' for writing", __func__, fname.c_str());
        return;
    }

    wav_header header;
    header.sample_rate = sample_rate;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = data.size() * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto & sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    file.close();
}

static outetts_version get_tts_version(llama_model *model, json speaker = json::object()) {
    if (speaker.contains("version")) {
        std::string version = speaker["version"].get<std::string>();
        if (version == "0.2") {
            return OUTETTS_V0_2;
        } else if (version == "0.3") {
            return OUTETTS_V0_3;
        } else {
            printf("%s: Unsupported speaker version '%s'\n", __func__, version.c_str());
        }
    }

    // Also could get version from model itself
    const char *chat_template = llama_model_chat_template(model, nullptr);
    if (chat_template && std::string(chat_template) == "outetts-0.3") {
        return OUTETTS_V0_3;
    }

    // Use 0.2 as the default version
    return OUTETTS_V0_2;
}

static std::string audio_text_from_speaker(json speaker, const outetts_version tts_version = OUTETTS_V0_2) {
    std::string audio_text = "<|text_start|>";

    if (tts_version == OUTETTS_V0_2 || tts_version == OUTETTS_V0_3) {
        std::string separator = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
        for (const auto &word : speaker["words"]) {
            audio_text += word["word"].get<std::string>() + separator;
        }
    }

    return audio_text;
}

static std::string audio_data_from_speaker(json speaker, const outetts_version tts_version = OUTETTS_V0_2) {
    std::string audio_data = "<|audio_start|>\n";

    if (tts_version == OUTETTS_V0_2 || tts_version == OUTETTS_V0_3) {
        std::string code_start = (tts_version == OUTETTS_V0_3) ? "" : "<|code_start|>";
        std::string code_end = (tts_version == OUTETTS_V0_3) ? "<|space|>" : "<|code_end|>";
        for (const auto &word : speaker["words"]) {
            std::string word_text = word["word"].get<std::string>();
            double duration = word["duration"].get<double>();
            std::vector<int> codes = word["codes"].get<std::vector<int>>();

            // Create the audio output entry
            std::ostringstream word_entry;
            word_entry << word_text << "<|t_" << std::fixed << std::setprecision(2)
                       << duration << "|>" + code_start;
            for (const auto &Code : codes) {
                word_entry << "<|" << Code << "|>";
            }
            word_entry << code_end << "\n";
            audio_data += word_entry.str();
        }
    }

    return audio_data;
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -mv vocoder.gguf -v en_male_1.json -p \"Hello!\"\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string prompt;
    std::string model_path;
    std::string vocoder_path;
    json speaker;

    // parse command line arguments
    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-mv") == 0) {
                if (i + 1 < argc) {
                    vocoder_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                if (i + 1 < argc) {
                    std::ifstream file(argv[++i]);
                    if (!file) {
                        fprintf(stderr, "%s: Failed to open file '%s' for reading\n", __func__, argv[i]);
                        return 1;
                    }
                    speaker = json::parse(file);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-p") == 0) {
                if (i + 1 < argc) {
                    prompt = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty() || vocoder_path.empty() || speaker.empty()) {
        print_usage(argc, argv);
        return 1;
    }
}