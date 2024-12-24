import pytest
import time
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated,return_tokens", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False, False),
    ("Write a joke about AI from a very long prompt which will not be truncated", 256, "(princesses|everyone|kids|Anna|forest)+", 46, 64, False, True),
])
def test_completion(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool, return_tokens: bool):
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
        "return_tokens": return_tokens,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_prompt
    assert res.body["timings"]["predicted_n"] == n_predicted
    assert res.body["truncated"] == truncated
    assert type(res.body["has_new_line"]) == bool
    assert match_regex(re_content, res.body["content"])
    if return_tokens:
        assert len(res.body["tokens"]) > 0
        assert all(type(tok) == int for tok in res.body["tokens"])
    else:
        assert res.body["tokens"] == []


@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False),
    ("Write a joke about AI from a very long prompt which will not be truncated", 256, "(princesses|everyone|kids|Anna|forest)+", 46, 64, False),
])
def test_completion_stream(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool):
    global server
    server.start()
    res = server.make_stream_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
        "stream": True,
    })
    content = ""
    for data in res:
        assert "stop" in data and type(data["stop"]) == bool
        if data["stop"]:
            assert data["timings"]["prompt_n"] == n_prompt
            assert data["timings"]["predicted_n"] == n_predicted
            assert data["truncated"] == truncated
            assert data["stop_type"] == "limit"
            assert type(data["has_new_line"]) == bool
            assert "generation_settings" in data
            assert server.n_predict is not None
            assert data["generation_settings"]["n_predict"] == min(n_predict, server.n_predict)
            assert data["generation_settings"]["seed"] == server.seed
            assert match_regex(re_content, content)
        else:
            assert len(data["tokens"]) > 0
            assert all(type(tok) == int for tok in data["tokens"])
            content += data["content"]


def test_completion_stream_vs_non_stream():
    global server
    server.start()
    res_stream = server.make_stream_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "I believe the meaning of life is",
        "stream": True,
    })
    res_non_stream = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "I believe the meaning of life is",
    })
    content_stream = ""
    for data in res_stream:
        content_stream += data["content"]
    assert content_stream == res_non_stream.body["content"]


@pytest.mark.parametrize("n_slots", [1, 2])
def test_consistent_result_same_seed(n_slots: int):
    global server
    server.n_slots = n_slots
    server.start()
    last_res = None
    for _ in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
        if last_res is not None:
            assert res.body["content"] == last_res.body["content"]
        last_res = res


@pytest.mark.parametrize("n_slots", [1, 2])
def test_different_result_different_seed(n_slots: int):
    global server
    server.n_slots = n_slots
    server.start()
    last_res = None
    for seed in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": seed,
            "temperature": 1.0,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
        if last_res is not None:
            assert res.body["content"] != last_res.body["content"]
        last_res = res

# TODO figure why it don't work with temperature = 1
# @pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("n_batch", [16, 32])
@pytest.mark.parametrize("temperature", [0.0])
def test_consistent_result_different_batch_size(n_batch: int, temperature: float):
    global server
    server.n_batch = n_batch
    server.start()
    last_res = None
    for _ in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": 42,
            "temperature": temperature,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
        if last_res is not None:
            assert res.body["content"] == last_res.body["content"]
        last_res = res


@pytest.mark.skip(reason="This test fails on linux, need to be fixed")
def test_cache_vs_nocache_prompt():
    global server
    server.start()
    res_cache = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": True,
    })
    res_no_cache = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": False,
    })
    assert res_cache.body["content"] == res_no_cache.body["content"]


def test_completion_with_tokens_input():
    global server
    server.temperature = 0.0
    server.start()
    prompt_str = "I believe the meaning of life is"
    res = server.make_request("POST", "/tokenize", data={
        "content": prompt_str,
        "add_special": True,
    })
    assert res.status_code == 200
    tokens = res.body["tokens"]

    # single completion
    res = server.make_request("POST", "/completion", data={
        "prompt": tokens,
    })
    assert res.status_code == 200
    assert type(res.body["content"]) == str

    # batch completion
    res = server.make_request("POST", "/completion", data={
        "prompt": [tokens, tokens],
    })
    assert res.status_code == 200
    assert type(res.body) == list
    assert len(res.body) == 2
    assert res.body[0]["content"] == res.body[1]["content"]

    # mixed string and tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": [tokens, prompt_str],
    })
    assert res.status_code == 200
    assert type(res.body) == list
    assert len(res.body) == 2
    assert res.body[0]["content"] == res.body[1]["content"]

    # mixed string and tokens in one sequence
    res = server.make_request("POST", "/completion", data={
        "prompt": [1, 2, 3, 4, 5, 6, prompt_str, 7, 8, 9, 10, prompt_str],
    })
    assert res.status_code == 200
    assert type(res.body["content"]) == str


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 3),
    (2, 2),
    (2, 4),
    (4, 2), # some slots must be idle
    (4, 6),
])
def test_completion_parallel_slots(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.temperature = 0.0
    server.start()

    PROMPTS = [
        ("Write a very long book.", "(very|special|big)+"),
        ("Write another a poem.", "(small|house)+"),
        ("What is LLM?", "(Dad|said)+"),
        ("The sky is blue and I love it.", "(climb|leaf)+"),
        ("Write another very long music lyrics.", "(friends|step|sky)+"),
        ("Write a very long joke.", "(cat|Whiskers)+"),
    ]
    def check_slots_status():
        should_all_slots_busy = n_requests >= n_slots
        time.sleep(0.1)
        res = server.make_request("GET", "/slots")
        n_busy = sum([1 for slot in res.body if slot["is_processing"]])
        if should_all_slots_busy:
            assert n_busy == n_slots
        else:
            assert n_busy <= n_slots

    tasks = []
    for i in range(n_requests):
        prompt, re_content = PROMPTS[i % len(PROMPTS)]
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": prompt,
            "seed": 42,
            "temperature": 1.0,
        })))
    tasks.append((check_slots_status, ()))
    results = parallel_function_calls(tasks)

    # check results
    for i in range(n_requests):
        prompt, re_content = PROMPTS[i % len(PROMPTS)]
        res = results[i]
        assert res.status_code == 200
        assert type(res.body["content"]) == str
        assert len(res.body["content"]) > 10
        # FIXME: the result is not deterministic when using other slot than slot 0
        # assert match_regex(re_content, res.body["content"])


def test_n_probs():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_probs": 10,
        "temperature": 0.0,
        "n_predict": 5,
    })
    assert res.status_code == 200
    assert "completion_probabilities" in res.body
    assert len(res.body["completion_probabilities"]) == 5
    for tok in res.body["completion_probabilities"]:
        assert "id" in tok and tok["id"] > 0
        assert "token" in tok and type(tok["token"]) == str
        assert "logprob" in tok and tok["logprob"] <= 0.0
        assert "bytes" in tok and type(tok["bytes"]) == list
        assert len(tok["top_logprobs"]) == 10
        for prob in tok["top_logprobs"]:
            assert "id" in prob and prob["id"] > 0
            assert "token" in prob and type(prob["token"]) == str
            assert "logprob" in prob and prob["logprob"] <= 0.0
            assert "bytes" in prob and type(prob["bytes"]) == list


def test_n_probs_stream():
    global server
    server.start()
    res = server.make_stream_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_probs": 10,
        "temperature": 0.0,
        "n_predict": 5,
        "stream": True,
    })
    for data in res:
        if data["stop"] == False:
            assert "completion_probabilities" in data
            assert len(data["completion_probabilities"]) == 1
            for tok in data["completion_probabilities"]:
                assert "id" in tok and tok["id"] > 0
                assert "token" in tok and type(tok["token"]) == str
                assert "logprob" in tok and tok["logprob"] <= 0.0
                assert "bytes" in tok and type(tok["bytes"]) == list
                assert len(tok["top_logprobs"]) == 10
                for prob in tok["top_logprobs"]:
                    assert "id" in prob and prob["id"] > 0
                    assert "token" in prob and type(prob["token"]) == str
                    assert "logprob" in prob and prob["logprob"] <= 0.0
                    assert "bytes" in prob and type(prob["bytes"]) == list


def test_n_probs_post_sampling():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_probs": 10,
        "temperature": 0.0,
        "n_predict": 5,
        "post_sampling_probs": True,
    })
    assert res.status_code == 200
    assert "completion_probabilities" in res.body
    assert len(res.body["completion_probabilities"]) == 5
    for tok in res.body["completion_probabilities"]:
        assert "id" in tok and tok["id"] > 0
        assert "token" in tok and type(tok["token"]) == str
        assert "prob" in tok and 0.0 < tok["prob"] <= 1.0
        assert "bytes" in tok and type(tok["bytes"]) == list
        assert len(tok["top_probs"]) == 10
        for prob in tok["top_probs"]:
            assert "id" in prob and prob["id"] > 0
            assert "token" in prob and type(prob["token"]) == str
            assert "prob" in prob and 0.0 <= prob["prob"] <= 1.0
            assert "bytes" in prob and type(prob["bytes"]) == list
        # because the test model usually output token with either 100% or 0% probability, we need to check all the top_probs
        assert any(prob["prob"] == 1.0 for prob in tok["top_probs"])
