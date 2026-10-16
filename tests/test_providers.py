import sys
from types import SimpleNamespace

import pytest

from nl_sql_agent.providers import (
    BackendError,
    BackendUnavailable,
    GroqBackend,
    HuggingFaceBackend,
    MockBackend,
    OllamaBackend,
    WatsonxBackend,
    backend_from_env,
)
from tests.fakes import FakeGroqClient, FakeWatsonxFactory, all_backends

SYSTEM, PROMPT = "sys-prompt", "user-prompt"
EXPECTED_MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": PROMPT},
]


def test_mock_returns_scripted_responses_in_order_and_records_calls():
    backend = MockBackend(["first", "second"])
    assert backend.complete(PROMPT, system=SYSTEM) == "first"
    assert backend.complete(PROMPT, system=SYSTEM, tier="large") == "second"
    assert [call.model for call in backend.calls] == ["mock-small", "mock-large"]
    assert backend.calls[0].prompt == PROMPT
    assert backend.calls[0].system == SYSTEM


def test_mock_raises_backend_error_when_script_is_exhausted():
    backend = MockBackend([])
    with pytest.raises(BackendError, match="no scripted response"):
        backend.complete(PROMPT, system=SYSTEM)


def test_mock_can_simulate_a_provider_outage():
    backend = MockBackend(fail_with="429 rate limited")
    with pytest.raises(BackendError, match="429"):
        backend.complete(PROMPT, system=SYSTEM)


@pytest.mark.parametrize("name", ["watsonx", "groq", "huggingface", "ollama"])
def test_every_backend_sends_system_and_user_messages_and_returns_text(name):
    backend = all_backends("hello")[name]
    assert backend.complete(PROMPT, system=SYSTEM) == "hello"
    client = (
        getattr(backend, "_client", None) or backend._factory.created["ibm/granite-3-8b-instruct"]
    )
    assert client.kwargs[0]["messages"] == EXPECTED_MESSAGES


@pytest.mark.parametrize("name", ["watsonx", "groq", "huggingface", "ollama"])
def test_large_tier_selects_the_large_model(name):
    backend = all_backends("ok")[name]
    backend.complete(PROMPT, system=SYSTEM, tier="large")
    if name == "watsonx":
        assert list(backend._factory.created) == [backend.models["large"]]
    else:
        assert backend._client.kwargs[0]["model"] == backend.models["large"]


def test_model_ids_can_be_overridden_per_tier():
    backend = GroqBackend(FakeGroqClient("ok"), models={"small": "my-model"})
    backend.complete(PROMPT, system=SYSTEM)
    assert backend._client.kwargs[0]["model"] == "my-model"
    assert backend.models["large"] == GroqBackend.default_models["large"]


def test_sdk_exceptions_are_mapped_to_backend_error():
    backend = GroqBackend(FakeGroqClient(error=TimeoutError("read timed out")))
    with pytest.raises(BackendError, match="groq: TimeoutError"):
        backend.complete(PROMPT, system=SYSTEM)


def test_empty_provider_response_is_a_backend_error():
    backend = GroqBackend(FakeGroqClient(text=None))
    with pytest.raises(BackendError, match="non-text"):
        backend.complete(PROMPT, system=SYSTEM)


def test_watsonx_builds_each_model_once_and_reuses_it():
    factory = FakeWatsonxFactory("ok")
    backend = WatsonxBackend(factory)
    backend.complete(PROMPT, system=SYSTEM)
    first = factory.created[backend.models["small"]]
    backend.complete(PROMPT, system=SYSTEM)
    assert factory.created[backend.models["small"]] is first
    assert len(first.kwargs) == 2


@pytest.mark.parametrize(
    ("cls", "variable"),
    [
        (WatsonxBackend, "WATSONX_APIKEY"),
        (GroqBackend, "GROQ_API_KEY"),
        (HuggingFaceBackend, "HF_TOKEN"),
    ],
)
def test_missing_credentials_raise_backend_unavailable(cls, variable):
    with pytest.raises(BackendUnavailable, match=variable):
        cls.from_env({})


def test_missing_sdk_raises_backend_unavailable_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "groq", None)
    with pytest.raises(BackendUnavailable, match=r"nl-sql-duckdb-agent\[groq\]"):
        GroqBackend.from_env({"GROQ_API_KEY": "test-key"})


def test_from_env_builds_a_client_and_reads_model_overrides(monkeypatch):
    fake_sdk = SimpleNamespace(Groq=lambda api_key: FakeGroqClient("ok"))
    monkeypatch.setitem(sys.modules, "groq", fake_sdk)
    backend = GroqBackend.from_env({"GROQ_API_KEY": "test-key", "NLSQL_MODEL_LARGE": "big"})
    assert backend.models["large"] == "big"
    assert backend.complete(PROMPT, system=SYSTEM) == "ok"


def test_ollama_needs_no_credentials(monkeypatch):
    hosts = []
    fake_sdk = SimpleNamespace(Client=lambda host: hosts.append(host) or object())
    monkeypatch.setitem(sys.modules, "ollama", fake_sdk)
    OllamaBackend.from_env({})
    assert hosts == ["http://localhost:11434"]


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        backend_from_env("openai-but-not-really", {})


def test_watsonx_from_env_wires_credentials_project_and_model(monkeypatch):
    built = []

    class FakeModelInference:
        def __init__(self, *, model_id, credentials, project_id):
            built.append({"model_id": model_id, "credentials": credentials, "project": project_id})

        def chat(self, messages, params):
            return {"choices": [{"message": {"content": "ok"}}]}

    sdk = SimpleNamespace(Credentials=lambda *, url, api_key: {"url": url, "api_key": api_key})
    monkeypatch.setitem(sys.modules, "ibm_watsonx_ai", sdk)
    monkeypatch.setitem(
        sys.modules,
        "ibm_watsonx_ai.foundation_models",
        SimpleNamespace(ModelInference=FakeModelInference),
    )
    env = {"WATSONX_APIKEY": "test-key", "WATSONX_PROJECT_ID": "proj-1"}
    backend = WatsonxBackend.from_env(env)
    assert built == []  # models are built lazily, on first use
    assert backend.complete(PROMPT, system=SYSTEM) == "ok"
    assert built == [
        {
            "model_id": "ibm/granite-3-8b-instruct",
            "credentials": {"url": "https://us-south.ml.cloud.ibm.com", "api_key": "test-key"},
            "project": "proj-1",
        }
    ]


def test_huggingface_from_env_passes_the_token(monkeypatch):
    tokens = []
    sdk = SimpleNamespace(InferenceClient=lambda *, token: tokens.append(token) or object())
    monkeypatch.setitem(sys.modules, "huggingface_hub", sdk)
    HuggingFaceBackend.from_env({"HF_TOKEN": "hf_test"})
    assert tokens == ["hf_test"]


def test_completion_budget_fits_a_multi_line_query():
    from nl_sql_agent.providers.base import MAX_TOKENS

    backend = GroqBackend(FakeGroqClient("ok"))
    backend.complete(PROMPT, system=SYSTEM)
    assert backend._client.kwargs[0]["max_tokens"] == MAX_TOKENS >= 512
