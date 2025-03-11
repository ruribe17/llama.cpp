#ifndef TRANSLATOR_HPP
#define TRANSLATOR_HPP

#include <curl/curl.h>
#include <iostream>
#include <json.hpp>
#include <string>

class Translator
{
public:
    Translator();
    ~Translator();

    bool is_korean(const std::string &text);
    std::string translate_korean_to_english(const std::string &text);

private:
    static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *userp);
};

#endif // TRANSLATOR_HPP
