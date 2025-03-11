#include "translator.hpp"
#include <regex>

Translator::Translator()
{
    curl_global_init(CURL_GLOBAL_ALL);
}

Translator::~Translator()
{
    curl_global_cleanup();
}

bool Translator::is_korean(const std::string &text)
{
    for (unsigned char c : text)
    {
        if (c >= 0xE0 && c <= 0xFA)
        { // UTF-8 한글 범위
            return true;
        }
    }
    return false;
}

size_t Translator::WriteCallback(void *contents, size_t size, size_t nmemb, std::string *userp)
{
    size_t total_size = size * nmemb;
    userp->append((char *)contents, total_size);
    return total_size;
}

std::string Translator::translate_korean_to_english(const std::string &text)
{
    if (!is_korean(text))
    {
        return text;
    }

    CURL *curl = curl_easy_init();
    if (!curl)
    {
        return "{}";
    }

    std::string readBuffer;
    char *encodedText = curl_easy_escape(curl, text.c_str(), text.length());
    std::string url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=ko&tl=en&dt=t&q=" + std::string(encodedText);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(curl);
    curl_free(encodedText);

    if (res != CURLE_OK)
    {
        std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return "{}";
    }

    curl_easy_cleanup(curl);

    std::cout << "API Response: " << readBuffer << std::endl;

    try
    {
        auto json_response = nlohmann::json::parse(readBuffer);

        std::string translated_text;
        for (const auto &sentence : json_response[0])
        {
            translated_text += sentence[0].get<std::string>(); // 번역된 텍스트만 추출
        }

        return translated_text;
    }
    catch (const std::exception &e)
    {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return "{}";
    }
}
