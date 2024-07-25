#if defined(_MSC_VER)
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "unicode.h"
#include "unicode-data.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <locale>
#include <codecvt>

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

//static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cp) {
//    std::vector<uint16_t> result;
//    if (/* 0x0000 <= cp && */ cp <= 0xffff) {
//        result.emplace_back(cp);
//        return result;
//    }
//    if (0x10000 <= cp && cp <= 0x10ffff) {
//        result.emplace_back(0xd800 | ((cp - 0x10000) >> 10));
//        result.emplace_back(0xdc00 | ((cp - 0x10000) & 0x03ff));
//        return result;
//    }
//    throw std::invalid_argument("failed to convert codepoint to utf16");
//}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

//static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
//    assert(offset < utf16.size());
//    if (((utf16[0] >> 10) << 10) != 0xd800) {
//        auto result = utf16[offset + 0];
//        offset += 1;
//        return result;
//    }
//
//    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//        throw std::invalid_argument("invalid character");
//    }
//
//    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
//    offset += 2;
//    return result;
//}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(unicode_cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

static inline std::wstring unicode_wstring_from_utf8(const std::string & s) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.from_bytes(s);
}

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    for (const auto & word : bpe_words) {
        std::string text_utf;
        auto utf_word =  unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        std::string encoded_token;
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        static const codepoint_categ SENTINEL = codepoint_categ::MASK + 1;
        auto _get_categ = [&] (const size_t pos) -> codepoint_categ {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_category(cpts[pos]) : SENTINEL;
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto categ = _get_categ(pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos+1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            auto categ2 = (cpt == ' ' ? _get_categ(pos+1) : categ);
            // regex: <space>?\p{L}+
            if (categ2.is_L()) {
                pos += (cpt == ' ');
                while (categ2.is_L()) {
                    categ2 = _get_categ(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if (categ2.is_N()) {
                pos += (cpt == ' ');
                while (categ2.is_N()) {
                    categ2 = _get_categ(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if (!(categ2.is_whitespace() | categ2.is_L() | categ2.is_N()) && categ2 != SENTINEL) {
                pos += (cpt == ' ');
                while (!(categ2.is_whitespace() | categ2.is_L() | categ2.is_N()) && categ2 != SENTINEL) {
                    categ2 = _get_categ(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (_get_categ(pos+num_whitespaces).is_whitespace()) {
                num_whitespaces++;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        static const codepoint_categ SENTINEL = codepoint_categ::MASK + 1;
        auto _get_categ = [&] (const size_t pos) -> codepoint_categ {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_category(cpts[pos]) : SENTINEL;
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto categ = _get_categ(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || categ.is_N())) {
                if (categ.is_L() || _get_categ(pos+1).is_L()) {  // one or more letters
                    pos++;
                    while (_get_categ(pos).is_L()) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (categ.is_N()) {
                size_t ini = pos;
                while (_get_categ(pos).is_N()) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto categ2 = (cpt == ' ' ? _get_categ(pos+1) : categ);
            if (!(categ2.is_whitespace() | categ2.is_L() | categ2.is_N()) && categ2 != SENTINEL) {
                pos += (cpt == ' ');
                while (!(categ2.is_whitespace() | categ2.is_L() | categ2.is_N()) && categ2 != SENTINEL) {
                    categ2 = _get_categ(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_categ(pos+num_whitespaces).is_whitespace()) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// use std::wregex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::wstring & wtext, const std::wstring & regex_expr, const std::vector<size_t> & offsets) {
    std::wregex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::wcregex_iterator it(wtext.data() + start, wtext.data() + start + offset, expr);
        std::wcregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::wcmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// use std::regex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::regex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::cregex_iterator it(text.data() + start, text.data() + start + offset, expr);
        std::cregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::cmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {

        bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
    }

    return bpe_offsets;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cp) {
    std::string result;

    if (/* 0x00 <= cp && */ cp <= 0x7f) {
        result.push_back(cp);
        return result;
    }
    if (0x80 <= cp && cp <= 0x7ff) {
        result.push_back(0xc0 | ((cp >> 6) & 0x1f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x800 <= cp && cp <= 0xffff) {
        result.push_back(0xe0 | ((cp >> 12) & 0x0f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x10000 <= cp && cp <= 0x10ffff) {
        result.push_back(0xf0 | ((cp >> 18) & 0x07));
        result.push_back(0x80 | ((cp >> 12) & 0x3f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.cbegin(), unicode_ranges_nfd.cend(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        result.push_back(unicode_cpt_from_utf8(utf8, offset));
    }
    return result;
}

codepoint_categ unicode_cpt_category(const uint32_t cp) {
    static const std::vector<codepoint_categ> cpt_categs = [] {
        std::vector<codepoint_categ> cpt_categs(MAX_CODEPOINTS, codepoint_categ::UNDEF);
        uint32_t cpt = 0;
        for (uint16_t rle : unicode_rle_codepoints_categs) {
            const uint32_t index = rle & 31;
            const uint32_t count = rle >> 5;
            const auto categ = codepoint_categ::from_index(index);
            //printf( "Codepoints 0x%05X to 0x%05X categ %s\n", cpt, cpt + count, categ.c_str());
            for (uint32_t i = 0; i <= count; ++i) {
                cpt_categs[cpt++] = categ;
            }
        }
        assert (cpt == MAX_CODEPOINTS);

        for (auto cpt : unicode_vec_whitespace) {
            cpt_categs[cpt].set_flag(codepoint_categ::WHITESPACE);
        }

        for (auto p : unicode_map_lowercase) {
            cpt_categs[cpt].set_flag(codepoint_categ::LOWERCASE);
        }

        for (auto p : unicode_map_uppercase) {
            cpt_categs[cpt].set_flag(codepoint_categ::UPPERCASE);
        }

        //for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        //    cpt_categs[cpt].set_flag(codepoint_categ::NORM_NFD);
        //}

        return cpt_categs;
    }();
    return cp < cpt_categs.size() ? cpt_categs[cp] : codepoint_categ{};
}

codepoint_categ unicode_cpt_category(const std::string & utf8) {
    if (utf8.empty()) {
        return codepoint_categ{};  // undefined
    }
    size_t offset = 0;
    return unicode_cpt_category(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

uint32_t unicode_tolower(uint32_t cp) {
    auto it = unicode_map_lowercase.find(cp);
    return it == unicode_map_lowercase.end() ? cp : it->second;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs) {
    //TODO: update and add more comments
    // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
    // ref: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2081479935

    // 0xDB80 to 0xDBFF: Private Use High Surrogate (128 range values)
    static const uint32_t COLLAPSE_CPT_RANGE_FIRST = 0xDB80;
    static const uint32_t COLLAPSE_CPT_RANGE_LAST  = 0xDBFF;
    auto category_to_collapsed_cpt = [] (const codepoint_categ categ) {
        const uint16_t subindex = categ.get_subcategory() >> 7;  // subcategory stored in 3 bits
        switch(categ.get_category()) {                           // category fits in other 3 bits
            case codepoint_categ::UNDEF: return COLLAPSE_CPT_RANGE_FIRST + ((0 << 3) | subindex);
            case codepoint_categ::C:     return COLLAPSE_CPT_RANGE_FIRST + ((1 << 3) | subindex);
            case codepoint_categ::L:     return COLLAPSE_CPT_RANGE_FIRST + ((2 << 3) | subindex);
            case codepoint_categ::M:     return COLLAPSE_CPT_RANGE_FIRST + ((3 << 3) | subindex);
            case codepoint_categ::N:     return COLLAPSE_CPT_RANGE_FIRST + ((4 << 3) | subindex);
            case codepoint_categ::P:     return COLLAPSE_CPT_RANGE_FIRST + ((5 << 3) | subindex);
            case codepoint_categ::S:     return COLLAPSE_CPT_RANGE_FIRST + ((6 << 3) | subindex);
            case codepoint_categ::Z:     return COLLAPSE_CPT_RANGE_FIRST + ((7 << 3) | subindex);
            default:  assert (false);    return COLLAPSE_CPT_RANGE_FIRST;
        }
    };
    auto category_to_collapsed_range = [&] (const codepoint_categ categ) {
        // \p{Ll} --> \p{Ll} to \p{Ll}  // has subcategory ? yes
        // \p{Lu} --> \p{Lu} to \p{Lu}  // has subcategory ? yes
        // \p{L}  --> \p{Ll} to \p{Lu}  // has subcategory ? no
        assert ((COLLAPSE_CPT_RANGE_FIRST & 0b111) == 0);
        const uint32_t collapsed = category_to_collapsed_cpt(categ);
        const uint32_t range = (collapsed & 0b111) ? 0 : 0b111;  // has subcategory ?
        return std::pair<uint32_t, uint32_t>(collapsed, collapsed + range);
    };

    const auto cpts = unicode_cpts_from_utf8(text);

    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        std::vector<std::pair<uint32_t, uint32_t>> regex_expr_ranges;        // start codepoint, last codepoint
        std::vector<std::pair<uint32_t, codepoint_categ>> regex_expr_categs; // offset, codepoint category
        std::map<uint16_t, std::wstring> map_categ_wregex;                   // categ --> regex utf32 string
        std::wstring wregex_collapsed;
        std::wstring wtext_collapsed;
        bool inside_square = false;
        bool is_cpt_range  = false;

        // common ranges: \w \d
        regex_expr_ranges.emplace_back('a', 'z');
        regex_expr_ranges.emplace_back('A', 'Z');
        regex_expr_ranges.emplace_back('0', '9');
        regex_expr_ranges.emplace_back('_', '_');
        // common ranges: \s
        for (uint32_t cpt : unicode_vec_whitespace) {
            const auto categ_prev = unicode_cpt_category(regex_expr_ranges.back().second);
            const auto categ_last = unicode_cpt_category(cpt);
            if (categ_prev == categ_last && regex_expr_ranges.back().second + 1 == cpt) {
                regex_expr_ranges.back().second = cpt;
            } else {
                regex_expr_ranges.emplace_back(cpt, cpt);
            }
        }

        // std::wregex \s does not match non-ASCII whitespaces
        static const codepoint_categ categ_whitespace(codepoint_categ::MASK + 1);  // UNDEF category, subcategory 1
        std::wstring & wregex_whitespaces = map_categ_wregex[categ_whitespace.get_subcategory()];
        wregex_whitespaces += L"\\s";
        for (uint32_t cpt : unicode_vec_whitespace) {
            if (cpt >= 0x80) {  // non-ASCII whitespaces
                if (wregex_whitespaces.back() + 1 == cpt) {
                    if (*(wregex_whitespaces.end() - 2) == '-') {
                        wregex_whitespaces.back() = cpt;
                    } else {
                        wregex_whitespaces += '-';
                        wregex_whitespaces += cpt;
                    }
                } else {
                    wregex_whitespaces += cpt;
                }
            }
        }

        const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);

        for (size_t i = 0; i < cpts_regex.size(); ++i) {
            uint32_t cpt = cpts_regex[i];

            if (inside_square) {
                switch(cpt) {
                    case '^':
                        if (cpts_regex[i - 1] != '[') {
                            break;
                        }
                        continue;
                    case ']':
                        inside_square = false;
                        continue;
                    case '-':
                        is_cpt_range = true;
                        continue;
                }
            } else {
                switch(cpt) {
                    case '^':
                        if (i > 0) {
                            break;
                        }
                        continue;
                    case '$':
                        if (i + 1 < cpts_regex.size()) {
                            break;
                        }
                        continue;
                    case '[':
                        inside_square = true;
                        continue;
                    case '{':
                        while (cpt && cpt != '}') {
                            cpt = cpts_regex[++i];
                        }
                        continue;
                    case '}':
                    case ']':
                        assert (false);
                    case '(':
                        if (cpts_regex[i + 1] == '?') {  // (?: (?i: (?= (?! (?<= (?<!
                            if (cpts_regex[i + 2] == ':') {
                                i += 2;
                            } else if (cpts_regex[i + 2] == 'i') {
                                i += 3;
                                assert (cpts_regex[i] == ':');
                            } else {
                                i += 2 + (cpts_regex[i + 2] == '<');
                                assert (cpts_regex[i] == '=' || cpts_regex[i] == '!');
                            }
                        }
                        continue;
                    case ')':
                    case '|':
                    case '.':
                    case '?':
                    case '+':
                    case '*':
                        continue;
                }
            }

            if (cpt == '\\' && cpts_regex[i + 1] == 'p' && cpts_regex[i + 2] == '{') {
                assert (cpts_regex[i + 3] && cpts_regex[i + 4]);
                codepoint_categ categ = {};
                if (cpts_regex[i + 4] == '}') {
                    categ = codepoint_categ::from_chars((char)cpts_regex[i + 3]);
                } else {
                    categ = codepoint_categ::from_chars((char)cpts_regex[i + 3], (char)cpts_regex[i + 4]);
                    assert (cpts_regex[i + 5] == '}');
                }
                categ.set_flag(codepoint_categ::WHITESPACE, inside_square);  //NOTE: reusing flag 'WHITESPACE' to store 'inside square brackets'
                regex_expr_categs.emplace_back(i, categ);
                i += cpts_regex[i + 4] == '}' ? 4 : 5;
                continue;
            }

            if (cpt == '\\') {
                if (cpts_regex[i + 1] == 's' || cpts_regex[i + 1] == 'S') {  // \s \S
                    regex_expr_categs.emplace_back(i, categ_whitespace);
                    //NOTE: reusing flag 'WHITESPACE' to store 'inside square brackets'
                    regex_expr_categs.back().second.set_flag(codepoint_categ::WHITESPACE, inside_square);
                    i += 1;
                    continue;
                }
            }

            if (cpt == '\\') {  
                switch (cpts_regex[i + 1]) {
                    case 's':  ++i;  continue;  // \s whitespaces
                    case 'w':  ++i;  continue;  // \w words
                    case 'd':  ++i;  continue;  // \d digits
                    case 'S':  ++i;  continue;  // \S no whitespaces
                    case 'W':  ++i;  continue;  // \W no words
                    case 'D':  ++i;  continue;  // \D no digits
                    case 't':  ++i;  cpt = '\t';  break;
                    case 'r':  ++i;  cpt = '\r';  break;
                    case 'n':  ++i;  cpt = '\n';  break;
                    case 'x':  assert (false);    break;  //TODO: hex values
                    case 'u':  assert (false);    break;  //TODO: unicode values
                    case 'U':  assert (false);    break;  //TODO: unicode values
                    default:  // escaped character
                        assert (!is_cpt_range);
                        cpt = cpts_regex[++i];
                        assert (cpt < 0x80);
                    break;
                }
            }

            assert (cpt < COLLAPSE_CPT_RANGE_FIRST || COLLAPSE_CPT_RANGE_LAST < cpt);

            if (is_cpt_range) {
                is_cpt_range = false;
                regex_expr_ranges.back().second = cpt;
            } else {
                regex_expr_ranges.emplace_back(cpt, cpt);
            }
        }

        // assign collapsed codepoint to each category regex \p{...}
        for (auto offset_categ : regex_expr_categs) {
            const uint16_t subcateg = offset_categ.second.get_subcategory();
            auto it = map_categ_wregex.find(subcateg);
            if (it == map_categ_wregex.end()) {
                const auto collapsed_range = category_to_collapsed_range(offset_categ.second);
                map_categ_wregex[subcateg] = (wchar_t) collapsed_range.first;
                if (collapsed_range.first < collapsed_range.second) {
                    map_categ_wregex[subcateg] += (wchar_t) '-';
                    map_categ_wregex[subcateg] += (wchar_t) collapsed_range.second;
                }
            }
        }

        // copy found regex ranges to each category regex
        uint32_t regex_expr_ranges_uniques = 0;
        std::pair<uint32_t, uint32_t> prev_range = {0, -1};
        std::sort(regex_expr_ranges.begin(), regex_expr_ranges.end());
        for (auto range : regex_expr_ranges) {
            range.first = std::max(range.first, prev_range.second + 1);  // prevent overlapping  //TODO: as error?
            if (range.first > range.second) {  // skip overlapping and repetitions
                continue;
            }
            codepoint_categ categ = unicode_cpt_category(range.first);
            assert (categ == unicode_cpt_category(range.second));
            auto it0 = map_categ_wregex.find(categ.get_category());
            auto it1 = map_categ_wregex.find(categ.get_subcategory());
            for (const auto & it : {it0, it1}) {
                if (it != map_categ_wregex.end()) {
                    it->second += (wchar_t) range.first;
                    if (range.first < range.second) {
                        it->second += (wchar_t) '-';
                        it->second += (wchar_t) range.second;
                    } 
                }
            }
            prev_range = range;
            regex_expr_ranges[regex_expr_ranges_uniques++] = range;
        }
        regex_expr_ranges.resize(regex_expr_ranges_uniques);

        // replace categories with respective collapsed codepoint and ranges
        uint32_t i = 0;
        wregex_collapsed.reserve(regex_expr.size());
        for (auto offset_categ : regex_expr_categs) {
            while (i < offset_categ.first) {  // copy original regex until reaching the category
                wregex_collapsed += (wchar_t) cpts_regex[i];
                i++;
            }
            assert (cpts_regex[i] == '\\');
            const uint32_t cpt_next = cpts_regex[i + 1];
            const bool is_negated = cpt_next < 'a';  // is uppercase
            if (cpt_next == 'p' || cpt_next == 'P') {
                assert (cpts_regex[i + 2] == '{' && cpts_regex[i + 3]);
                i += cpts_regex[i + 4] == '}' ? 5 : 6;
                assert (cpts_regex[i - 1] == '}');
            } else {
                assert (cpt_next == 's' || cpt_next == 'w' || cpt_next == 'd' ||  // \s \w \d
                        cpt_next == 'S' || cpt_next == 'W' || cpt_next == 'D');   // \S \W \D
                i += 2;
            }
            const codepoint_categ categ = offset_categ.second;
            auto it = map_categ_wregex.find(categ.get_subcategory());
            assert (it != map_categ_wregex.end());
            if (it != map_categ_wregex.end()) {
                if (categ.is_whitespace()) {  // inside square brackets  //NOTE: reusing flag WHITESPACE
                    assert (is_negated == false);
                    wregex_collapsed += it->second;
                } else if(it->second.size() == 1 && !is_negated) {
                    wregex_collapsed += it->second;
                } else {
                    wregex_collapsed += '[';
                    if (is_negated) {
                        wregex_collapsed += '^';
                    }
                    wregex_collapsed += it->second;
                    wregex_collapsed += ']';
                }
            }
        }
        while (i < (uint32_t)cpts_regex.size()) {
            wregex_collapsed += cpts_regex[i];
            i++;
        }

        // collapse text codepoints not included in 'regex_expr_ranges'
        wtext_collapsed.reserve(cpts.size());
        for (uint32_t cpt : cpts) {
            const codepoint_categ categ = unicode_cpt_category(cpt);
            auto it = std::lower_bound(regex_expr_ranges.begin(), regex_expr_ranges.end(), cpt,
                [] (const std::pair<uint32_t, uint32_t> range, const uint32_t cpt) {
                    return range.second < cpt;
                }
            );
            if (it == regex_expr_ranges.end() || cpt < it->first || it->second < cpt) {
                cpt = category_to_collapsed_cpt(categ);  // not found, collapse to category codepoint
            }
            wtext_collapsed += (wchar_t) cpt;
        }

        bpe_offsets = unicode_regex_split_stl(wtext_collapsed, wregex_collapsed, bpe_offsets);
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    return unicode_byte_encoding_process(bpe_words);
}
