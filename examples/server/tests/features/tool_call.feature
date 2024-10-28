@llama.cpp
@server
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   BOS token is 1
    And   42 as server seed
    And   8192 KV cache size
    And   32 as batch size
    And   1 slots
    And   prometheus compatible metrics exposed
    And   jinja templates are enabled


  Scenario Outline: Template <template_name> + tinystories model w/ required tool_choice yields <tool_name> tool call
    Given a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a test chat template file named <template_name>
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   <n_predict> max tokens to predict
    And   a user prompt write a hello world in python
    And   a tool choice required
    And   tools <tools>
    And   parallel tool calls is <parallel_tool_calls>
    And   an OAI compatible chat completions request with no api error
    Then  tool <tool_name> is called with arguments <tool_arguments>

    Examples: Prompts
      | template_name                         | n_predict | tool_name | tool_arguments         | tools | parallel_tool_calls |
      | meetkai-functionary-medium-v3.1       | 128       | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       | disabled |
      | meetkai-functionary-medium-v3.1       | 128       | ipython   | {"code": "Yes, you can."} | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] | disabled |
      | meetkai-functionary-medium-v3.2       | 128       | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       | disabled |
      | meetkai-functionary-medium-v3.2       | 128       | ipython   | {"code": "Yes,"}       | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] | disabled |
      | meta-llama-Meta-Llama-3.1-8B-Instruct | 64        | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       | disabled |
      | meta-llama-Meta-Llama-3.1-8B-Instruct | 64        | ipython   | {"code": "it and realed at the otter. Asked Dave Dasty, Daisy is a big, shiny blue. As"}    | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] | disabled |
      | meta-llama-Llama-3.2-3B-Instruct      | 64        | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       | disabled |
      | meta-llama-Llama-3.2-3B-Instruct      | 64        | ipython   | {"code": "Yes,"}    | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] | disabled |
      | mistralai-Mistral-Nemo-Instruct-2407  | 128       | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       | disabled |
      | mistralai-Mistral-Nemo-Instruct-2407  | 128       | ipython   | {"code": "It's a small cable."}    | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] | disabled |


  Scenario Outline: Template <template_name> + tinystories model yields no tool call
    Given a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a test chat template file named <template_name>
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   <n_predict> max tokens to predict
    And   a user prompt write a hello world in python
    And   tools [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called

    Examples: Prompts
      | template_name                         | n_predict |
      | meta-llama-Meta-Llama-3.1-8B-Instruct | 64        |
      | meetkai-functionary-medium-v3.1       | 128       |
      | meetkai-functionary-medium-v3.2       | 128       |


  Scenario: Tool call template + tinystories and no tool won't call any tool
    Given a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a test chat template file named meta-llama-Meta-Llama-3.1-8B-Instruct
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   16 max tokens to predict
    And   a user prompt write a hello world in python
    And   tools []
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called


  @slow
  Scenario Outline: Python hello world w/ <hf_repo> + python tool yields tool call
    Given a model file <hf_file> from HF repo <hf_repo>
    And   a test chat template file named <template_override>
    And   no warmup
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   256 max tokens to predict
    And   a user prompt write a hello world in python
    And   python tool
    And   parallel tool calls is disabled
    And   an OAI compatible chat completions request with no api error
    Then  tool <tool_name> is called with arguments <tool_arguments>

    Examples: Prompts
      | tool_name | tool_arguments                       | hf_repo                                              | hf_file                                 | template_override                             |
      | ipython   | {"code": "print('Hello, World!')"}   | bartowski/Phi-3.5-mini-instruct-GGUF                 | Phi-3.5-mini-instruct-Q4_K_M.gguf       |                                               |
      | ipython   | {"code": "print('Hello, World!')"}   | NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF            | Hermes-2-Pro-Llama-3-8B-Q8_0.gguf       | NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use |
      | ipython   | {"code": "print('Hello, World!')"} | bartowski/Mistral-Nemo-Instruct-2407-GGUF            | Mistral-Nemo-Instruct-2407-Q8_0.gguf    | mistralai-Mistral-Nemo-Instruct-2407          |
      | ipython   | {"code": "print('Hello, World!'}"}   | lmstudio-community/Llama-3.2-1B-Instruct-GGUF        | Llama-3.2-1B-Instruct-Q4_K_M.gguf       | meta-llama-Llama-3.2-3B-Instruct              |
      | ipython   | {"code": "print("}                   | lmstudio-community/Llama-3.2-3B-Instruct-GGUF        | Llama-3.2-3B-Instruct-Q6_K.gguf         | meta-llama-Llama-3.2-3B-Instruct              |
      | ipython   | {"code": "print("}                   | lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF   | Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf  |                                               |
      # | ipython   | {"code": "print("}                   | lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF  | Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf |                                               |
      # | ipython   | {"code": "print('Hello, world!')"}   | bartowski/gemma-2-2b-it-GGUF                         | gemma-2-2b-it-Q4_K_M.gguf               |                                               |
      # | ipython   | {"code": "print('Hello, World!')"}   | meetkai/functionary-small-v3.2-GGUF                  | functionary-small-v3.2.Q4_0.gguf        | meetkai-functionary-medium-v3.2               |


  @slow
  Scenario Outline: Python hello world w/o tools yields no tool call
    Given a model file Phi-3.5-mini-instruct-Q4_K_M.gguf from HF repo bartowski/Phi-3.5-mini-instruct-GGUF
    And   no warmup
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   256 max tokens to predict
    And   a user prompt write a hello world in python
    And   parallel tool calls is disabled
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called
