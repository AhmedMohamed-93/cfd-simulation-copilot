"""Tests for the provider-agnostic LLM client (Hugging Face, Ollama, and
Mistral are all supported; the active one is set via LLM_PROVIDER).

All provider SDK calls are mocked, so these tests never require a Hugging
Face token, a running Ollama server, or any network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from config import settings
from src.common.llm_client import LLMClient, _extract_ollama_content


def test_llm_client_defaults_to_configured_provider():
    """LLMClient with no explicit provider falls back to settings.LLM_PROVIDER.

    Deliberately reads the expected provider/model from `settings` rather
    than hardcoding a literal: which provider is the configured default is
    itself a project setting (see .env / .env.example), not something this
    test should need updating every time that default changes.
    """
    client = LLMClient()
    assert client.provider == settings.LLM_PROVIDER
    assert client.model == settings.LLM_MODEL


def test_llm_client_huggingface_complete_dispatches_to_inference_client():
    """complete() calls InferenceClient.chat_completion with the given messages."""
    with patch("huggingface_hub.InferenceClient") as mock_client_cls:
        mock_instance = MagicMock()
        fake_message = MagicMock()
        fake_message.content = "hello from huggingface"
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        mock_instance.chat_completion.return_value = fake_response
        mock_client_cls.return_value = mock_instance

        client = LLMClient(provider="huggingface", api_key="test-token")
        result = client.complete(messages=[{"role": "user", "content": "hi"}], temperature=0.2)

    assert result == "hello from huggingface"
    mock_client_cls.assert_called_once_with(
        model="mistralai/Mistral-7B-Instruct-v0.2", token="test-token"
    )
    _, kwargs = mock_instance.chat_completion.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_llm_client_huggingface_json_mode_sets_response_format():
    """json_mode=True passes response_format={'type': 'json_object'} through."""
    with patch("huggingface_hub.InferenceClient") as mock_client_cls:
        mock_instance = MagicMock()
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content="{}"))]
        mock_instance.chat_completion.return_value = fake_response
        mock_client_cls.return_value = mock_instance

        client = LLMClient(provider="huggingface", api_key="test-token")
        client.complete(messages=[{"role": "user", "content": "hi"}], json_mode=True)

    _, kwargs = mock_instance.chat_completion.call_args
    assert kwargs["response_format"] == {"type": "json_object"}


def test_llm_client_huggingface_defaults_max_tokens_when_unset():
    """max_tokens defaults to a positive cap when not explicitly provided."""
    with patch("huggingface_hub.InferenceClient") as mock_client_cls:
        mock_instance = MagicMock()
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_instance.chat_completion.return_value = fake_response
        mock_client_cls.return_value = mock_instance

        client = LLMClient(provider="huggingface", api_key="test-token")
        client.complete(messages=[{"role": "user", "content": "hi"}])

    _, kwargs = mock_instance.chat_completion.call_args
    assert kwargs["max_tokens"] == 1024


def test_llm_client_ollama_complete_dispatches_to_client_chat():
    """complete() calls ollama.Client.chat with the given messages and options.

    Passes an explicit model= override so this test's expectation doesn't
    depend on whatever OLLAMA_MODEL happens to be configured in the
    environment it runs in.
    """
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {"message": {"content": "hello from ollama"}}
        mock_client_cls.return_value = mock_instance

        client = LLMClient(provider="ollama", model="llama3.1:8b")
        result = client.complete(messages=[{"role": "user", "content": "hi"}], temperature=0.2)

    assert result == "hello from ollama"
    mock_instance.chat.assert_called_once()
    _, kwargs = mock_instance.chat.call_args
    assert kwargs["model"] == "llama3.1:8b"
    assert kwargs["options"]["temperature"] == 0.2


def test_llm_client_ollama_json_mode_sets_format_json():
    """json_mode=True passes format='json' through to the Ollama chat call."""
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {"message": {"content": "{}"}}
        mock_client_cls.return_value = mock_instance

        client = LLMClient(provider="ollama")
        client.complete(messages=[{"role": "user", "content": "hi"}], json_mode=True)

    _, kwargs = mock_instance.chat.call_args
    assert kwargs["format"] == "json"


def test_llm_client_ollama_max_tokens_maps_to_num_predict():
    """max_tokens is translated to Ollama's num_predict option."""
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {"message": {"content": "ok"}}
        mock_client_cls.return_value = mock_instance

        client = LLMClient(provider="ollama")
        client.complete(messages=[{"role": "user", "content": "hi"}], max_tokens=10)

    _, kwargs = mock_instance.chat.call_args
    assert kwargs["options"]["num_predict"] == 10


def test_extract_ollama_content_handles_dict_response():
    """_extract_ollama_content parses a plain dict-style Ollama response."""
    response = {"message": {"content": "dict-style content"}}
    assert _extract_ollama_content(response) == "dict-style content"


def test_extract_ollama_content_handles_object_response():
    """_extract_ollama_content parses an attribute-style ChatResponse object."""
    message = MagicMock()
    message.content = "object-style content"
    response = MagicMock()
    response.message = message
    assert _extract_ollama_content(response) == "object-style content"


def test_llm_client_mistral_provider_uses_mistral_model_and_key():
    """LLMClient(provider='mistral') resolves the Mistral model/key from settings."""
    client = LLMClient(provider="mistral", api_key="test-key")
    assert client.provider == "mistral"
    assert client.model == "mistral-large-latest"
    assert client._api_key == "test-key"


def test_llm_client_mistral_complete_dispatches_to_mistral_sdk():
    """complete() on the mistral provider builds a Mistral client and reads its response."""
    client = LLMClient(provider="mistral", api_key="test-key")

    fake_message = MagicMock()
    fake_message.content = "hello from mistral"
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    mock_mistral_instance = MagicMock()
    mock_mistral_instance.chat.complete.return_value = fake_response

    with patch("mistralai.Mistral", return_value=mock_mistral_instance):
        result = client.complete(messages=[{"role": "user", "content": "hi"}])

    assert result == "hello from mistral"
