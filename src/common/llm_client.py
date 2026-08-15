"""Provider-agnostic chat LLM client.

Dispatches to the Hugging Face Inference API by default (free with an HF
account, no local RAM/GPU required; see LLM_PROVIDER="huggingface"). Two
alternative providers are also supported: a local Ollama server
(LLM_PROVIDER="ollama", free but requires enough local RAM to load the
model) and the Mistral API (LLM_PROVIDER="mistral", production/paid). Every
caller only ever talks to :class:`LLMClient`, so switching providers is a
one-line config change with no other code affected.

Each provider's SDK is imported lazily, inside its own branch, so an
installation only needs the one package matching its configured provider.
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


def _extract_ollama_content(response: Any) -> str:
    """Pull the assistant message text out of an Ollama chat response.

    Handles both the attribute-style ``ChatResponse`` object returned by
    recent ``ollama`` client versions and the plain dict returned by older
    ones.

    Args:
        response: The raw response returned by ``ollama.Client.chat``.

    Returns:
        The assistant message content as a string.
    """
    message = response["message"] if isinstance(response, dict) else response.message
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "") or ""


class LLMClient:
    """A minimal chat-completion client that works against HF, Ollama, or Mistral.

    Attributes:
        provider: The active backend: "huggingface", "ollama", or "mistral".
        model: The model identifier used for completions.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the client for the configured (or overridden) provider.

        Args:
            provider: "huggingface", "ollama", or "mistral". Defaults to
                settings.LLM_PROVIDER.
            model: Model identifier override. Defaults to the provider's
                configured model.
            api_key: API token/key override (Hugging Face token or Mistral
                API key, depending on provider; ignored for Ollama).
            base_url: Ollama server URL override (ignored for other providers).
        """
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        if self.provider == "mistral":
            self.model = model or settings.MISTRAL_MODEL
            self._api_key = api_key or settings.MISTRAL_API_KEY
            self._mistral_client = None  # lazily constructed
        elif self.provider == "ollama":
            self.model = model or settings.OLLAMA_MODEL
            self._ollama_base_url = base_url or settings.OLLAMA_BASE_URL
            self._ollama_client = None  # lazily constructed
        else:
            self.provider = "huggingface"
            self.model = model or settings.HF_MODEL
            self._hf_token = api_key or settings.HF_API_TOKEN
            self._hf_client = None  # lazily constructed

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Run a chat completion against the configured provider.

        Args:
            messages: A list of {"role": ..., "content": ...} messages.
            temperature: Sampling temperature.
            max_tokens: Optional cap on generated tokens.
            json_mode: If True, ask the model to return a raw JSON object.

        Returns:
            The assistant's reply text.
        """
        if self.provider == "mistral":
            return self._complete_mistral(messages, temperature, max_tokens, json_mode)
        if self.provider == "ollama":
            return self._complete_ollama(messages, temperature, max_tokens, json_mode)
        return self._complete_huggingface(messages, temperature, max_tokens, json_mode)

    def _complete_huggingface(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> str:
        """Run a chat completion against the Hugging Face Inference API.

        Free with a Hugging Face account (https://huggingface.co/settings/tokens),
        no local RAM/GPU required, unlike the Ollama provider.

        Args:
            messages: Chat messages.
            temperature: Sampling temperature.
            max_tokens: Optional cap on generated tokens; HF requires a
                positive value, so a default cap is applied if omitted.
            json_mode: If True, requests JSON-object output via response_format.

        Returns:
            The assistant's reply text.

        Raises:
            ImportError: If the optional ``huggingface_hub`` package is not
                installed.
        """
        if self._hf_client is None:
            from huggingface_hub import InferenceClient  # noqa: PLC0415

            self._hf_client = InferenceClient(model=self.model, token=self._hf_token or None)

        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1024,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._hf_client.chat_completion(**kwargs)
        return response.choices[0].message.content

    def _complete_ollama(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> str:
        """Run a chat completion against a local Ollama server.

        Args:
            messages: Chat messages.
            temperature: Sampling temperature.
            max_tokens: Optional cap on generated tokens (maps to num_predict).
            json_mode: If True, requests raw JSON output via format="json".

        Returns:
            The assistant's reply text.

        Raises:
            ImportError: If the optional ``ollama`` package is not installed.
        """
        if self._ollama_client is None:
            import ollama  # noqa: PLC0415 - intentionally lazy/optional

            self._ollama_client = ollama.Client(host=self._ollama_base_url)

        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        response = self._ollama_client.chat(
            model=self.model,
            messages=messages,
            options=options,
            format="json" if json_mode else None,
        )
        return _extract_ollama_content(response)

    def _complete_mistral(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> str:
        """Run a chat completion against the Mistral API (production path).

        Args:
            messages: Chat messages.
            temperature: Sampling temperature.
            max_tokens: Optional cap on generated tokens.
            json_mode: If True, requests JSON-mode output.

        Returns:
            The assistant's reply text.

        Raises:
            ImportError: If the optional ``mistralai`` package is not
                installed.
        """
        if self._mistral_client is None:
            from mistralai import Mistral  # noqa: PLC0415 - intentionally lazy/optional

            self._mistral_client = Mistral(api_key=self._api_key)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._mistral_client.chat.complete(**kwargs)
        return response.choices[0].message.content


def get_llm_client() -> LLMClient:
    """Build an LLMClient for the currently configured provider.

    Returns:
        A new LLMClient instance.
    """
    return LLMClient()
