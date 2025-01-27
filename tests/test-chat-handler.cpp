#include "chat-handler.hpp"
#include "chat-template.hpp"
#include "llama-grammar.h"
#include "unicode.h"

#include <fstream>
#include <iostream>
#include <string>
#include <json.hpp>

using json = nlohmann::ordered_json;

template <class T>
static void assert_equals(const T & expected, const T & actual) {
    if (expected != actual) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

static std::string read_file(const std::string &path) {
  std::cout << "# Reading: " << path << std::endl << std::flush;
  std::ifstream fs(path, std::ios_base::binary);
  if (!fs.is_open()) {
    fs = std::ifstream("../" + path, std::ios_base::binary);
    if (!fs.is_open()) {
      throw std::runtime_error("Failed to open file: " + path);
    }
  }
  fs.seekg(0, std::ios_base::end);
  auto size = fs.tellg();
  fs.seekg(0);
  std::string out;
  out.resize(static_cast<size_t>(size));
  fs.read(&out[0], static_cast<std::streamsize>(size));
  return out;
}

static std::unique_ptr<llama_grammar> build_grammar(const std::string & grammar_str) {
    return std::unique_ptr<llama_grammar>(llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", nullptr, 0, nullptr, 0));
}

// TODO: extract to common helper (copied from test-grammar-integration.cpp)
static bool match_string(const std::string & input, llama_grammar * grammar) {
    const auto cpts = unicode_cpts_from_utf8(input);

    auto & stacks_cur = llama_grammar_get_stacks(grammar);

    for (const auto & cpt : cpts) {
        llama_grammar_accept(grammar, cpt);

        if (stacks_cur.empty()) {
            // no stacks means that the grammar failed to match at this point
            return false;
        }
    }

    for (const auto & stack : stacks_cur) {
        if (stack.empty()) {
            // An empty stack means that the grammar has been completed
            return true;
        }
    }

    return false;
}

// Dumps `{"a": 1}` as `"{\"a\": 1}"`, unlike nlohmann::json::dump which would dump it as `"{\"a\":1}"`.
static std::string dump(const json & j) {
  return minja::Value(j).dump(-1, /* to_json= */ true);
}

static void assert_msg_equals(const common_chat_msg & result, const std::string & expected_content, const json & expected_tool_calls) {
    assert_equals(expected_content, result.content);
    auto tool_calls = json::array();
    for (const auto & tc : result.tool_calls) {
      auto arguments = tc.arguments;
      try {
        arguments = dump(json::parse(arguments));
      } catch (const std::exception & e) {
        // ignore
      }
      auto tool_call = json {
        {"type", "function"},
        {"function", {
          {"arguments", arguments},
          {"name", tc.name},
        }},
      };
      if (!tc.id.empty()) {
        tool_call["id"] = tc.id;
      }
      tool_calls.push_back(tool_call);
    }
    // Reparse / dump w/ non-ordered JSON variant.
    auto expected = nlohmann::json::parse(expected_tool_calls.dump()).dump();
    auto actual = nlohmann::json::parse(tool_calls.dump()).dump();
    assert_equals(expected, actual);
}

const auto special_function_tool = json::parse(R"({
  "type": "function",
  "function": {
    "name": "special_function",
    "description": "I'm special",
    "parameters": {
      "type": "object",
      "properties": {
        "arg1": {
          "type": "integer",
          "description": "The arg."
        }
      },
      "required": ["arg1"]
    }
  }
})");
const auto python_tool = json::parse(R"({
  "type": "function",
  "function": {
    "name": "python",
    "description": "an ipython interpreter",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "Python code to execute."
        }
      },
      "required": ["code"]
    }
  }
})");
const json tools = {special_function_tool, python_tool};

// static void test_parsing() {
//     json request = {
//       {"tools", tools}
//     };

//     const auto fooBarCall = json {
//       {"type", "function"},
//       {"function", {
//         {"name", "foo"},
//         {"arguments", dump({
//           {"bar", 1}
//         })},
//       }}
//     };

//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_GENERIC, tools,
//       "{\"tool_call\": {\"name\": \"foo\", \"arguments\": {\"bar\": 1}}}",
//       "",
//       json::array({fooBarCall}));
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_GENERIC, tools,
//       "{\"tool_calls\": [{\"name\": \"foo\", \"arguments\": {\"bar\": 1}}]}",
//       "",
//       json::array({fooBarCall}));

//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_HERMES_2_PRO, tools,
//       "<tool_call>{\"name\": \"foo\", \"arguments\": {\"bar\": 1}}</tool_call>",
//       "",
//       json::array({fooBarCall}));

//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3, tools,
//       ">>>python\n{\"code\": \"print('Hello, world!')\"}",
//       "",
//       json {{
//         {"type", "function"},
//         {"function", {
//           {"name", "python"},
//           {"arguments", dump({
//             {"code", "print('Hello, world!')"}
//           })}
//         }}
//       }});
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3, tools,
//       ">>>special_function\n{\"arg1\": 1}\n ",
//       "",
//       json {{
//         {"type", "function"},
//         {"function", {
//           {"name", "special_function"},
//           {"arguments", dump({
//             {"arg1", 1}
//           })}
//         }}
//       }});

//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3_1, tools,
//       "Hell<function=foo>{\"arg1\": 1}</function>o, world<function=bar>{\"arg2\": 2}</function>!",
//       "Hello, world!",
//       json {
//         {
//           {"type", "function"},
//           {"function", {
//             {"name", "foo"},
//             {"arguments", dump({
//               {"arg1", 1}
//             })}
//           }}
//         },
//         {
//           {"type", "function"},
//           {"function", {
//             {"name", "bar"},
//             {"arguments", dump({
//               {"arg2", 2}
//             })}
//           }}
//         },
//       });
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3_1, tools,
//       "<function=test>{ } </function> ",
//       " ",
//       json {{
//         {"type", "function"},
//         {"function", {
//           {"name", "test"},
//           {"arguments", "{}"}
//         }}
//       }});

//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "<|python_tag|>this could be anything",
//       "",
//       json {{
//         {"type", "function"},
//         {"function", {
//           {"name", "python"},
//           {"arguments", "this could be anything"},
//         }}
//       }});
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "I'm thinking<|python_tag|>",
//       "I'm thinking",
//       json {{
//         {"type", "function"},
//         {"function", {
//           {"name", "python"},
//           {"arguments", ""},
//         }}
//       }});
//     auto special_function_call = json {
//         {"type", "function"},
//         {"function", {
//           {"arguments", dump({{"arg1", 1}})},
//           {"name", "special_function"},
//         }},
//     };
//     auto special_function_call_with_id = json::parse(special_function_call.dump());
//     special_function_call_with_id["id"] = "123456789";

//     auto no_function_call = json::array();

//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\"name\": \"python\", \"parameters\": {\"code\": \"print('Hey')\"}}",
//       "",
//       json::array({{
//         {"type", "function"},
//         {"function", {
//           {"arguments", dump({{"code", "print('Hey')"}})},
//           {"name", "python"},
//         }}
//       }}));
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
//       "",
//       json::array({special_function_call}));
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\n  \"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
//       "",
//       json::array({special_function_call}));
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\n\t\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
//       "",
//       json::array({special_function_call}));
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\n    \"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
//       "",
//       json::array({special_function_call}));
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\"type\": \"function\", \"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
//       "",
//       json::array({special_function_call}));

//     // No match: function unknown
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
//       "{\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
//       no_function_call);
//     // No match: bad indentation
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\n\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
//       "{\n\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
//       no_function_call);
//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1, tools,
//       "{\n \"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
//       "{\n \"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
//       no_function_call);

//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_MISTRAL_NEMO, tools,
//       "Bleh[TOOL_CALLS][{\"arguments\": {\"arg1\": 1}, \"name\": \"special_function\", \"id\": \"123456789\"}]",
//       "Bleh",
//       json::array({special_function_call_with_id}));

//     test_parse_tool_call(common_tool_call_style::COMMON_TOOL_CALL_STYLE_FIRE_FUNCTION_V2, tools,
//       "Bleh functools[{\"arguments\": {\"arg1\": 1}, \"name\": \"special_function\"}]",
//       "Bleh",
//       json::array({special_function_call}));
// }

// static void test_tool_call_style(const std::string & template_file, common_tool_call_style expected) {
//     const common_chat_template tmpl(read_file(template_file), "<s>", "</s>");
//     auto tool_call_style = common_tool_call_style_detect(tmpl);
//     std::cout << "# Testing tool call style of: " << template_file << std::endl << std::flush;
//     assert_equals(expected, tool_call_style);
// }

// static void test_tool_call_style_detection() {
//     test_tool_call_style("tests/chat/templates/meetkai-functionary-medium-v3.1.jinja", COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3_1);
//     test_tool_call_style("tests/chat/templates/meetkai-functionary-medium-v3.2.jinja", COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3);
//     test_tool_call_style("tests/chat/templates/fireworks-ai-llama-3-firefunction-v2.jinja", COMMON_TOOL_CALL_STYLE_FIRE_FUNCTION_V2);
//     test_tool_call_style("tests/chat/templates/meta-llama-Meta-Llama-3.1-8B-Instruct.jinja", COMMON_TOOL_CALL_STYLE_LLAMA_3_1);
//     test_tool_call_style("tests/chat/templates/meta-llama-Llama-3.2-3B-Instruct.jinja", COMMON_TOOL_CALL_STYLE_LLAMA_3_2);
//     test_tool_call_style("tests/chat/templates/Qwen-Qwen2.5-7B-Instruct.jinja", COMMON_TOOL_CALL_STYLE_HERMES_2_PRO);
//     test_tool_call_style("tests/chat/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja", COMMON_TOOL_CALL_STYLE_HERMES_2_PRO);
//     test_tool_call_style("tests/chat/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja", COMMON_TOOL_CALL_STYLE_HERMES_2_PRO);
//     test_tool_call_style("tests/chat/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja", COMMON_TOOL_CALL_STYLE_COMMAND_R_PLUS);
//     test_tool_call_style("tests/chat/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja", COMMON_TOOL_CALL_STYLE_MISTRAL_NEMO);
//     test_tool_call_style("tests/chat/templates/google-gemma-7b-it.jinja", COMMON_TOOL_CALL_STYLE_GENERIC);
// }

static std::string get_message_prompt_delta(const common_chat_template & tmpl, const std::vector<std::string> & end_tokens, const json & user_message, const json & delta_message, const json & tools) {
  fprintf(stderr, "Template source: %s\n", tmpl.source().c_str());
  fprintf(stderr, "Delta message: %s\n", delta_message.dump(2).c_str());

  common_chat_params params;
  params.parallel_tool_calls = true;
  params.messages = json::array();
  params.messages.push_back(user_message);
  params.tools = tools;
  std::string prefix = common_chat_init(tmpl, params).prompt;
  params.messages.push_back(delta_message);
  params.add_generation_prompt = false;
  std::string full = common_chat_init(tmpl, params).prompt;

  // Check full starts with prefix
  if (full.find(prefix) != 0) {
    throw std::runtime_error("Full message does not start with prefix");
  }

  if (full == prefix) {
    throw std::runtime_error("Full message is the same as the prefix");
  }

  auto delta = full.substr(prefix.size());

  // Strip end tokens
  for (const auto & end_token : end_tokens) {
    // rfind to find the last occurrence
    auto pos = delta.rfind(end_token);
    if (pos != std::string::npos) {
      delta = delta.substr(0, pos);
      break;
    }
  }
  return delta;
}

static void test_template(const common_chat_template & tmpl, const std::vector<std::string> & end_tokens, const json & tool_calling_message, const json & tools, bool skip_grammar_test = false) {
  // auto tool_call_style = common_tool_call_style_detect(tmpl);
  auto & tool_calls = tool_calling_message.at("tool_calls");

  // Format the message: apply the template to 1 user message w/ add_generation_prompt=true, then w/ the extra message w/ add_generation_prompt=false,
  // get the diff and try and parse it w/ the grammar.
  auto user_message = json {
      {"role", "user"},
      {"content", "Hello, world!"}
  };

  common_chat_params params;
  params.parallel_tool_calls = true;
  params.messages = json {user_message, tool_calling_message};
  params.tools = tools;
  auto chat_data = common_chat_init(tmpl, params);
  fprintf(stderr, "PROMPT: %s\n", chat_data.prompt.get<std::string>().c_str());
  auto grammar = build_grammar(chat_data.grammar);
  if (!grammar) {
    throw std::runtime_error("Failed to build grammar");
  }

  if (!skip_grammar_test) {
    auto full_delta = get_message_prompt_delta(tmpl, end_tokens, user_message, tool_calling_message, tools);
    std::cout << "Full delta:\n```\n" << full_delta << "\n```" << std::endl;

    const auto msg = chat_data.parser->parse_final(full_delta);
    assert_msg_equals(msg, "", tool_calls);

    auto content_less_delta = get_message_prompt_delta(tmpl, end_tokens, user_message, {
      {"role", "assistant"},
      {"content", {}},
      {"tool_calls", tool_calls}
    }, tools);
    if (!match_string(content_less_delta, grammar.get())) {
      throw std::runtime_error("Failed to match content-less delta against grammar:\n\nContent-less delta: " + content_less_delta + "\n\nGrammar: " + chat_data.grammar);
    }
  }
}

static void test_grammars() {
  auto tool_call_message = json {
    {"role", "assistant"},
    {"content", {}},
    {"tool_calls", json {{
      {"type", "function"},
      {"function", {
        {"name", "special_function"},
        {"arguments", "{\"arg1\": 1}"}
      }},
    }}}
  };
  auto tool_call_message_with_id = json::parse(tool_call_message.dump());
  tool_call_message_with_id["tool_calls"][0]["id"] = "123456789";

  auto python_tool_call_message = json {
    {"role", "assistant"},
    {"content", {}},
    {"tool_calls", json {{
      {"type", "function"},
      {"function", {
        {"name", "python"},
        {"arguments", "print('hey')"}
      }},
    }}}
  };

  {
    const common_chat_template tmpl(read_file("tests/chat/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja"), "<s>", "</s>");
    test_template(tmpl, { "</s>" }, tool_call_message_with_id, tools, /* skip_grammar_test= */ true);
  }
  // {
  //   const common_chat_template tmpl(read_file("tests/chat/templates/Qwen-Qwen2.5-7B-Instruct.jinja"), "<s>", "</s>");
  //   // assert_equals(tmpl.requires_object_arguments_, true);
  //   test_template(tmpl, { "<|im_end|>" }, tool_call_message, tools);
  //   test_template(tmpl, { "<|im_end|>" }, python_tool_call_message, tools);
  // }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<|im_end|>" }, tool_call_message, tools);
  }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<|im_end|>" }, tool_call_message, tools);
  }
  // {
  //   const common_chat_template tmpl(read_file("tests/chat/templates/meta-llama-Meta-Llama-3.1-8B-Instruct.jinja"), "<s>", "</s>");
  //   test_template(tmpl, { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
  // }
  // {
  //   const common_chat_template tmpl(read_file("tests/chat/templates/meta-llama-Meta-Llama-3.1-8B-Instruct.jinja"), "<s>", "</s>");
  //   test_template(tmpl, { "<|eom_id|>", "<|eot_id|>" }, python_tool_call_message, tools);
  // }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/meta-llama-Llama-3.2-3B-Instruct.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
  }
  // {
  //   const common_chat_template tmpl(read_file("tests/chat/templates/meta-llama-Llama-3.3-70B-Instruct.jinja"), "<s>", "</s>");
  //   test_template(tmpl, { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
  // }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/meetkai-functionary-medium-v3.1.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
  }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/meetkai-functionary-medium-v3.2.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
  }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/fireworks-ai-llama-3-firefunction-v2.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<|eot_id|>" }, tool_call_message, tools);
  }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/google-gemma-2-2b-it.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<end_of_turn>" }, tool_call_message_with_id, tools);
  }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/microsoft-Phi-3.5-mini-instruct.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<|end|>" }, tool_call_message_with_id, tools);
  }
  {
    const common_chat_template tmpl(read_file("tests/chat/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja"), "<s>", "</s>");
    test_template(tmpl, { "<｜end▁of▁sentence｜>" }, tool_call_message, tools);
  }
}

int main() {
    // test_tool_call_style_detection();
    // test_parsing();
    test_grammars();

    std::cout << "\n[tool-call] All tests passed!" << std::endl;
    return 0;
}
