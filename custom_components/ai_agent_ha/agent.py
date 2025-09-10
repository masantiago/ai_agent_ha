"""The AI Agent implementation with multiple provider support.

Example config:
ai_agent_ha:
  ai_provider: openai  # or 'llama', 'gemini', 'openrouter', 'anthropic', 'local'
  llama_token: "..."
  openai_token: "..."
  gemini_token: "..."
  openrouter_token: "..."
  anthropic_token: "..."
  local_url: "http://localhost:11434/api/generate"  # Required for local models
  # Model configuration (optional, defaults will be used if not specified)
  models:
    openai: "gpt-3.5-turbo"  # or "gpt-4", "gpt-4-turbo", etc.
    llama: "Llama-4-Maverick-17B-128E-Instruct-FP8"
    gemini: "gemini-1.5-flash"  # or "gemini-1.5-pro", "gemini-1.0-pro", etc.
    openrouter: "openai/gpt-4o"  # or any model available on OpenRouter
    anthropic: "claude-3-5-sonnet-20241022"  # or "claude-3-opus-20240229", etc.
    local: "llama3.2"  # model name for local API (optional if your API doesn't require it)
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aiohttp
import yaml  # type: ignore[import-untyped]
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import CONF_WEATHER_ENTITY, DOMAIN
from .system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_LOCAL

_LOGGER = logging.getLogger(__name__)


# === AI Client Abstractions ===
class BaseAIClient:
    async def get_response(self, messages, **kwargs):
        raise NotImplementedError


class LocalClient(BaseAIClient):
    def __init__(self, url, model=""):
        self.url = url
        self.model = model

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug(
            "Making request to local API with model: '%s' at URL: %s",
            self.model or "[NO MODEL SPECIFIED]",
            self.url,
        )

        if not self.model:
            _LOGGER.warning(
                "No model specified for local API request. Some APIs (like Ollama) require a model name."
            )
        headers = {"Content-Type": "application/json"}

        # Format user prompt from messages
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Simple formatting: prefixing each message with its role
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        # Add final prompt prefix for the assistant's response
        prompt += "Assistant: "

        # Build a generic payload that works with most local API servers
        payload = {
            "prompt": prompt,
            "stream": False,  # Disable streaming to get a single complete response
        }

        # Add model if specified
        if self.model:
            payload["model"] = self.model

        _LOGGER.debug("Local API request payload: %s", json.dumps(payload, indent=2))

        # Ollama-specific validation
        if "model" not in payload or not payload["model"]:
            _LOGGER.warning(
                "Missing 'model' field in request to local API. This may cause issues with Ollama."
            )
        elif self.url and "ollama" in self.url.lower():
            _LOGGER.debug(
                "Detected Ollama URL, ensuring model is specified: %s",
                payload.get("model"),
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Local API error %d: %s", resp.status, error_text)

                    # Provide more specific error messages for common Ollama issues
                    if resp.status == 404:
                        if "model" in payload and payload["model"]:
                            raise Exception(
                                f"Model '{payload['model']}' not found. Please ensure the model is installed in Ollama using: ollama pull {payload['model']}"
                            )
                        else:
                            raise Exception(
                                "Local API endpoint not found. Please check the URL and ensure Ollama is running."
                            )
                    elif resp.status == 400:
                        raise Exception(
                            f"Bad request to local API. Error: {error_text}"
                        )
                    else:
                        raise Exception(f"Local API error {resp.status}: {error_text}")

                try:
                    response_text = await resp.text()
                    _LOGGER.debug(
                        "Local API response (first 200 chars): %s", response_text[:200]
                    )
                    _LOGGER.debug("Local API response status: %d", resp.status)
                    _LOGGER.debug("Local API response headers: %s", dict(resp.headers))

                    # Try to parse as JSON
                    try:
                        data = json.loads(response_text)

                        # Try common response formats
                        # Ollama format - return only the response text
                        if "response" in data:
                            response_content = data["response"]
                            _LOGGER.debug(
                                "Extracted response content: %s",
                                (
                                    response_content[:100]
                                    if response_content
                                    else "[EMPTY]"
                                ),
                            )

                            # Check if response is empty or None
                            if not response_content or response_content.strip() == "":
                                _LOGGER.warning(
                                    "Ollama returned empty response. Full data: %s",
                                    data,
                                )
                                # Check if this is a loading response
                                if data.get("done_reason") == "load":
                                    _LOGGER.warning(
                                        "Ollama is still loading the model. Please wait and try again."
                                    )
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI model is still loading. Please wait a moment and try again.",
                                        }
                                    )
                                elif data.get("done") is False:
                                    _LOGGER.warning(
                                        "Ollama response indicates it's not done yet."
                                    )
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI is still processing your request. Please try again.",
                                        }
                                    )
                                else:
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI returned an empty response. Please try rephrasing your question.",
                                        }
                                    )

                            # Check if the response looks like JSON
                            response_content = response_content.strip()
                            if response_content.startswith(
                                "{"
                            ) and response_content.endswith("}"):
                                try:
                                    # Validate that it's actually JSON and contains valid request_type
                                    parsed_json = json.loads(response_content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response"
                                        )
                                        return response_content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text"
                                    )
                                    pass

                            # If it's plain text, wrap it in the expected JSON format
                            wrapped_response = {
                                "request_type": "final_response",
                                "response": response_content,
                            }
                            _LOGGER.debug("Wrapped plain text response in JSON format")
                            return json.dumps(wrapped_response)

                        # OpenAI-like format
                        elif "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                content = choice["message"]["content"]
                            elif "text" in choice:
                                content = choice["text"]
                            else:
                                content = str(data)

                            # Check if it's valid JSON with request_type
                            content = content.strip()
                            if content.startswith("{") and content.endswith("}"):
                                try:
                                    parsed_json = json.loads(content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response (OpenAI format)"
                                        )
                                        return content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text (OpenAI format)"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text (OpenAI format)"
                                    )
                                    pass

                            # Wrap in expected format if plain text
                            wrapped_response = {
                                "request_type": "final_response",
                                "response": content,
                            }
                            return json.dumps(wrapped_response)

                        # Generic content field
                        elif "content" in data:
                            content = data["content"]
                            content = content.strip()
                            if content.startswith("{") and content.endswith("}"):
                                try:
                                    parsed_json = json.loads(content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response (generic format)"
                                        )
                                        return content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text (generic format)"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text (generic format)"
                                    )
                                    pass

                            wrapped_response = {
                                "request_type": "final_response",
                                "response": content,
                            }
                            return json.dumps(wrapped_response)

                        # Handle case where no standard fields are found
                        _LOGGER.warning(
                            "No standard response fields found in local API response. Full response: %s",
                            data,
                        )

                        # Check for Ollama-specific edge cases
                        if data.get("done_reason") == "load":
                            return json.dumps(
                                {
                                    "request_type": "final_response",
                                    "response": "The AI model is still loading. Please wait a moment and try again.",
                                }
                            )
                        elif data.get("done") is False:
                            return json.dumps(
                                {
                                    "request_type": "final_response",
                                    "response": "The AI is still processing your request. Please try again.",
                                }
                            )
                        elif "message" in data:
                            # Some APIs use "message" field
                            message_content = data["message"]
                            if (
                                isinstance(message_content, dict)
                                and "content" in message_content
                            ):
                                content = message_content["content"]
                            else:
                                content = str(message_content)
                            return json.dumps(
                                {"request_type": "final_response", "response": content}
                            )

                        # Return the whole data as string if we can't find a specific field
                        return json.dumps(
                            {
                                "request_type": "final_response",
                                "response": f"Received unexpected response format from local API: {str(data)}",
                            }
                        )

                    except json.JSONDecodeError:
                        # If not JSON, check if it's a JSON response that got corrupted by wrapping
                        response_text = response_text.strip()
                        if response_text.startswith("{") and response_text.endswith(
                            "}"
                        ):
                            try:
                                parsed_json = json.loads(response_text)
                                if (
                                    isinstance(parsed_json, dict)
                                    and "request_type" in parsed_json
                                ):
                                    _LOGGER.debug(
                                        "Local model provided valid JSON response (direct)"
                                    )
                                    return response_text
                            except json.JSONDecodeError:
                                pass

                        # If not valid JSON, wrap the raw text in expected format
                        _LOGGER.debug("Response is not JSON, wrapping plain text")
                        wrapped_response = {
                            "request_type": "final_response",
                            "response": response_text,
                        }
                        return json.dumps(wrapped_response)

                except Exception as e:
                    _LOGGER.error("Failed to parse local API response: %s", str(e))
                    raise Exception(f"Failed to parse local API response: {str(e)}")


class LlamaClient(BaseAIClient):
    def __init__(self, token, model="Llama-4-Maverick-17B-128E-Instruct-FP8"):
        self.token = token
        self.model = model
        self.api_url = "https://api.llama.com/v1/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Llama API with model: %s", self.model)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        _LOGGER.debug("Llama request payload: %s", json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Llama API error %d: %s", resp.status, error_text)
                    raise Exception(f"Llama API error {resp.status}")
                data = await resp.json()
                # Extract text from Llama response
                completion = data.get("completion_message", {})
                content = completion.get("content", {})
                return content.get("text", str(data))


class OpenAIClient(BaseAIClient):
    def __init__(self, token, model="gpt-3.5-turbo"):
        self.token = token
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def _get_token_parameter(self):
        """Determine which token parameter to use based on the model."""
        # Models that require max_completion_tokens instead of max_tokens
        completion_token_models = ["o3-mini", "o3", "o1-mini", "o1-preview", "o1"]

        # Check if the model name contains any of the newer model identifiers
        model_lower = self.model.lower()
        if any(model_id in model_lower for model_id in completion_token_models):
            return "max_completion_tokens"
        return "max_tokens"

    def _is_restricted_model(self):
        """Check if the model has restricted parameters (no temperature, top_p, etc.)."""
        # Models that don't support temperature, top_p and other parameters
        restricted_models = ["o3-mini", "o3", "o1-mini", "o1-preview", "o1"]

        model_lower = self.model.lower()
        return any(model_id in model_lower for model_id in restricted_models)

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to OpenAI API with model: %s", self.model)

        # Validate token
        if not self.token or not self.token.startswith("sk-"):
            raise Exception("Invalid OpenAI API key format")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        # Determine which token parameter to use
        token_param = self._get_token_parameter()
        is_restricted = self._is_restricted_model()
        _LOGGER.debug(
            "Using token parameter '%s' for model: %s (restricted: %s)",
            token_param,
            self.model,
            is_restricted,
        )

        # Build payload with model-appropriate parameters
        payload = {"model": self.model, "messages": messages, token_param: 2048}

        # Only add temperature and top_p for models that support them
        if not is_restricted:
            payload.update({"temperature": 0.7, "top_p": 0.9})

        _LOGGER.debug("OpenAI request payload: %s", json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                response_text = await resp.text()
                _LOGGER.debug("OpenAI API response status: %d", resp.status)
                _LOGGER.debug("OpenAI API response: %s", response_text[:500])

                if resp.status != 200:
                    _LOGGER.error("OpenAI API error %d: %s", resp.status, response_text)
                    raise Exception(f"OpenAI API error {resp.status}: {response_text}")

                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    _LOGGER.error("Failed to parse OpenAI response as JSON: %s", str(e))
                    raise Exception(
                        f"Invalid JSON response from OpenAI: {response_text[:200]}"
                    )

                # Extract text from OpenAI response
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "")
                    if not content:
                        _LOGGER.warning("OpenAI returned empty content in message")
                        _LOGGER.debug(
                            "Full OpenAI response: %s", json.dumps(data, indent=2)
                        )
                    return content
                else:
                    _LOGGER.warning("OpenAI response missing expected structure")
                    _LOGGER.debug(
                        "Full OpenAI response: %s", json.dumps(data, indent=2)
                    )
                    return str(data)


class GeminiClient(BaseAIClient):
    def __init__(self, token, model="gemini-1.5-flash"):
        self.token = token
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Gemini API with model: %s", self.model)

        # Validate token
        if not self.token:
            raise Exception("Missing Gemini API key")

        headers = {"Content-Type": "application/json"}

        # Convert OpenAI-style messages to Gemini format
        gemini_contents = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # Gemini doesn't have a system role, so we prepend it to the first user message
                if not gemini_contents:
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": f"System: {content}"}]}
                    )
                else:
                    # Add system message as user message
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": f"System: {content}"}]}
                    )
            elif role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": content}]})

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "maxOutputTokens": 2048,
            },
        }

        # Add API key as query parameter
        url_with_key = f"{self.api_url}?key={self.token}"

        _LOGGER.debug("Gemini request payload: %s", json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url_with_key,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                response_text = await resp.text()
                _LOGGER.debug("Gemini API response status: %d", resp.status)
                _LOGGER.debug("Gemini API response: %s", response_text[:500])

                if resp.status != 200:
                    _LOGGER.error("Gemini API error %d: %s", resp.status, response_text)
                    raise Exception(f"Gemini API error {resp.status}: {response_text}")

                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    _LOGGER.error("Failed to parse Gemini response as JSON: %s", str(e))
                    raise Exception(
                        f"Invalid JSON response from Gemini: {response_text[:200]}"
                    )

                # Extract text from Gemini response
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts:
                        content = parts[0].get("text", "")
                        if not content:
                            _LOGGER.warning("Gemini returned empty text content")
                            _LOGGER.debug(
                                "Full Gemini response: %s", json.dumps(data, indent=2)
                            )
                        return content
                    else:
                        _LOGGER.warning("Gemini response missing parts")
                        _LOGGER.debug(
                            "Full Gemini response: %s", json.dumps(data, indent=2)
                        )
                else:
                    _LOGGER.warning("Gemini response missing expected structure")
                    _LOGGER.debug(
                        "Full Gemini response: %s", json.dumps(data, indent=2)
                    )
                return str(data)


class AnthropicClient(BaseAIClient):
    def __init__(self, token, model="claude-3-5-sonnet-20241022"):
        self.token = token
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Anthropic API with model: %s", self.model)
        headers = {
            "x-api-key": self.token,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Convert OpenAI-style messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # Anthropic uses a separate system parameter
                system_message = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})

        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.7,
            "messages": anthropic_messages,
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message

        _LOGGER.debug("Anthropic request payload: %s", json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Anthropic API error %d: %s", resp.status, error_text)
                    raise Exception(f"Anthropic API error {resp.status}")
                data = await resp.json()
                # Extract text from Anthropic response
                content_blocks = data.get("content", [])
                if content_blocks and isinstance(content_blocks, list):
                    # Get the text from the first content block
                    for block in content_blocks:
                        if block.get("type") == "text":
                            return block.get("text", str(data))
                return str(data)


class OpenRouterClient(BaseAIClient):
    def __init__(self, token, model="openai/gpt-4o"):
        self.token = token
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to OpenRouter API with model: %s", self.model)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://home-assistant.io",  # Optional for OpenRouter rankings
            "X-Title": "Home Assistant AI Agent",  # Optional for OpenRouter rankings
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        _LOGGER.debug("OpenRouter request payload: %s", json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error(
                        "OpenRouter API error %d: %s", resp.status, error_text
                    )
                    raise Exception(f"OpenRouter API error {resp.status}")
                data = await resp.json()
                # Extract text from OpenRouter response (OpenAI-compatible format)
                choices = data.get("choices", [])
                if not choices:
                    _LOGGER.warning("OpenRouter response missing choices")
                    _LOGGER.debug(
                        "Full OpenRouter response: %s", json.dumps(data, indent=2)
                    )
                    return str(data)
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", str(data))
                return str(data)


# === Main Agent ===
class AiAgentHaAgent:
    """Agent for handling queries with dynamic data requests and multiple AI providers."""


    def __init__(self, hass: HomeAssistant, config: Dict[str, Any]):
        """Initialize the agent with provider selection."""
        self.hass = hass
        self.config = config
        self.conversation_history: List[Dict[str, Any]] = []
        self._cache: Dict[str, Any] = {}
        self.ai_client: BaseAIClient
        self._cache_timeout = 300  # 5 minutes
        self._max_retries = 10
        self._retry_delay = 1  # seconds
        self._rate_limit = 60  # requests per minute
        self._last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()

        provider = config.get("ai_provider", "openai")
        models_config = config.get("models", {})

        _LOGGER.debug("Initializing AiAgentHaAgent with provider: %s", provider)
        _LOGGER.debug("Models config loaded: %s", models_config)

        # Set the appropriate system prompt based on provider
        if provider == "local":
            self.system_prompt = SYSTEM_PROMPT_LOCAL
            _LOGGER.debug("Using local-optimized system prompt")
        else:
            self.system_prompt = SYSTEM_PROMPT
            _LOGGER.debug("Using standard system prompt")

        # Initialize the appropriate AI client with model selection
        if provider == "openai":
            model = models_config.get("openai", "gpt-3.5-turbo")
            self.ai_client = OpenAIClient(config.get("openai_token"), model)
        elif provider == "gemini":
            model = models_config.get("gemini", "gemini-1.5-flash")
            self.ai_client = GeminiClient(config.get("gemini_token"), model)
        elif provider == "openrouter":
            model = models_config.get("openrouter", "openai/gpt-4o")
            self.ai_client = OpenRouterClient(config.get("openrouter_token"), model)
        elif provider == "anthropic":
            model = models_config.get("anthropic", "claude-3-5-sonnet-20241022")
            self.ai_client = AnthropicClient(config.get("anthropic_token"), model)
        elif provider == "local":
            model = models_config.get("local", "")
            url = config.get("local_url")
            if not url:
                _LOGGER.error("Missing local_url for local provider")
                raise Exception("Missing local_url configuration for local provider")
            self.ai_client = LocalClient(url, model)
        else:  # default to llama if somehow specified
            model = models_config.get("llama", "Llama-4-Maverick-17B-128E-Instruct-FP8")
            self.ai_client = LlamaClient(config.get("llama_token"), model)

        _LOGGER.debug(
            "AiAgentHaAgent initialized successfully with provider: %s, model: %s",
            provider,
            model,
        )

    def _validate_api_key(self) -> bool:
        """Validate the API key format."""
        provider = self.config.get("ai_provider", "openai")

        if provider == "openai":
            token = self.config.get("openai_token")
        elif provider == "gemini":
            token = self.config.get("gemini_token")
        elif provider == "openrouter":
            token = self.config.get("openrouter_token")
        elif provider == "anthropic":
            token = self.config.get("anthropic_token")
        elif provider == "local":
            token = self.config.get("local_url")
        else:
            token = self.config.get("llama_token")

        if not token or not isinstance(token, str):
            return False

        # For local provider, validate URL format
        if provider == "local":
            return bool(token.startswith(("http://", "https://")))

        # Add more specific validation based on your API key format
        return len(token) >= 32

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        if current_time - self._request_window_start >= 60:
            self._request_count = 0
            self._request_window_start = current_time

        if self._request_count >= self._rate_limit:
            return False

        self._request_count += 1
        return True

    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if it's still valid."""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if time.time() - timestamp < self._cache_timeout:
                return data
            del self._cache[key]
        return None

    def _set_cached_data(self, key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._cache[key] = (time.time(), data)

    def _sanitize_automation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize automation configuration to prevent injection attacks."""
        sanitized: Dict[str, Any] = {}
        for key, value in config.items():
            if key in ["alias", "description"]:
                # Sanitize strings
                sanitized[key] = str(value).strip()[:100]  # Limit length
            elif key in ["trigger", "condition", "action"]:
                # Validate arrays
                if isinstance(value, list):
                    sanitized[key] = value
            elif key == "mode":
                # Validate mode
                if value in ["single", "restart", "queued", "parallel"]:
                    sanitized[key] = value
        return sanitized

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem."""
        # Remove or replace unsafe characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Replace spaces and other whitespace with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it's not empty
        if not sanitized:
            sanitized = "automation"
        # Limit length
        return sanitized[:50]

    async def find_entity_by_name(self, search_term: str, domain: str = None) -> Optional[str]:
        """Find entity by searching friendly names and entity IDs."""
        search_lower = search_term.lower()
        candidates = []
        
        for state in self.hass.states.async_all():
            entity_id = state.entity_id
            
            # Skip if domain specified and doesn't match
            if domain and not entity_id.startswith(f"{domain}."):
                continue
                
            # Check friendly name
            friendly_name = state.attributes.get("friendly_name") or ""
            if search_lower in friendly_name.lower():
                candidates.append((entity_id, friendly_name))
                
            # Check entity ID
            if search_lower in entity_id.lower():
                candidates.append((entity_id, entity_id))
        
        # Return the best match (prioritize exact matches)
        if candidates:
            # Sort by relevance (exact matches first, then contains)
            candidates.sort(key=lambda x: (
                0 if search_lower == x[1].lower() else 1,  # Exact friendly name match
                0 if search_lower == x[0].split('.')[1].lower() else 1,  # Exact entity name match
                len(x[1])  # Shorter names first
            ))
            return candidates[0][0]
        
        return None

    async def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Get the state of a specific entity."""
        try:
            _LOGGER.debug("Requesting entity state for: %s", entity_id)
            state = self.hass.states.get(entity_id)
            if not state:
                # Try to find entity by name if direct lookup fails
                domain = entity_id.split('.')[0] if '.' in entity_id else None
                search_term = entity_id.split('.')[1] if '.' in entity_id else entity_id
                found_entity = await self.find_entity_by_name(search_term, domain)
                
                if found_entity:
                    _LOGGER.info("Found entity '%s' for search term '%s'", found_entity, entity_id)
                    state = self.hass.states.get(found_entity)
                    entity_id = found_entity  # Update entity_id to the found one
                
                if not state:
                    _LOGGER.warning("Entity not found: %s", entity_id)
                    return {"error": f"Entity {entity_id} not found"}

            result = {
                "entity_id": state.entity_id,
                "state": state.state,
                "last_changed": (
                    state.last_changed.isoformat() if state.last_changed else None
                ),
                "friendly_name": state.attributes.get("friendly_name"),
                "attributes": {
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in state.attributes.items()
                },
            }
            _LOGGER.debug("Retrieved entity state: %s", json.dumps(result))
            return result
        except Exception as e:
            _LOGGER.exception("Error getting entity state: %s", str(e))
            return {"error": f"Error getting entity state: {str(e)}"}

    async def get_entities_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all entities for a specific domain."""
        try:
            _LOGGER.debug("Requesting all entities for domain: %s", domain)
            states = [
                state
                for state in self.hass.states.async_all()
                if state.entity_id.startswith(f"{domain}.")
            ]
            _LOGGER.debug("Found %d entities in domain %s", len(states), domain)
            return [await self.get_entity_state(state.entity_id) for state in states]
        except Exception as e:
            _LOGGER.exception("Error getting entities by domain: %s", str(e))
            return [{"error": f"Error getting entities for domain {domain}: {str(e)}"}]

    async def get_entities_by_area(self, area_id: str) -> List[Dict[str, Any]]:
        """Get all entities for a specific area."""
        try:
            _LOGGER.debug("Requesting all entities for area: %s", area_id)

            # Get entity registry to find entities assigned to the area
            from homeassistant.helpers import device_registry as dr
            from homeassistant.helpers import entity_registry as er

            entity_registry = er.async_get(self.hass)
            device_registry = dr.async_get(self.hass)

            entities_in_area = []

            # Find entities assigned to the area (directly or through their device)
            for entity in entity_registry.entities.values():
                # Check if entity is directly assigned to the area
                if entity.area_id == area_id:
                    entities_in_area.append(entity.entity_id)
                # Check if entity's device is assigned to the area
                elif entity.device_id:
                    device = device_registry.devices.get(entity.device_id)
                    if device and device.area_id == area_id:
                        entities_in_area.append(entity.entity_id)

            _LOGGER.debug(
                "Found %d entities in area %s", len(entities_in_area), area_id
            )

            # Get state information for each entity
            result = []
            for entity_id in entities_in_area:
                state_info = await self.get_entity_state(entity_id)
                if not state_info.get("error"):  # Only include entities that exist
                    result.append(state_info)

            return result

        except Exception as e:
            _LOGGER.exception("Error getting entities by area: %s", str(e))
            return [{"error": f"Error getting entities for area {area_id}: {str(e)}"}]

    async def get_entities(self, area_id=None, area_ids=None) -> List[Dict[str, Any]]:
        """Get entities by area(s) - flexible method that supports single area or multiple areas."""
        try:
            # Handle different parameter formats
            areas_to_process = []

            if area_ids:
                # Multiple areas provided
                if isinstance(area_ids, list):
                    areas_to_process = area_ids
                else:
                    areas_to_process = [area_ids]
            elif area_id:
                # Single area provided
                if isinstance(area_id, list):
                    areas_to_process = area_id
                else:
                    areas_to_process = [area_id]
            else:
                return [{"error": "No area_id or area_ids provided"}]

            _LOGGER.debug("Requesting entities for areas: %s", areas_to_process)

            all_entities = []
            for area in areas_to_process:
                entities_in_area = await self.get_entities_by_area(area)
                all_entities.extend(entities_in_area)

            # Remove duplicates based on entity_id
            seen_entities = set()
            unique_entities = []
            for entity in all_entities:
                if isinstance(entity, dict) and "entity_id" in entity:
                    if entity["entity_id"] not in seen_entities:
                        seen_entities.add(entity["entity_id"])
                        unique_entities.append(entity)
                else:
                    unique_entities.append(entity)  # Keep error messages

            _LOGGER.debug(
                "Found %d unique entities across %d areas",
                len(unique_entities),
                len(areas_to_process),
            )
            return unique_entities

        except Exception as e:
            _LOGGER.exception("Error getting entities: %s", str(e))
            return [{"error": f"Error getting entities: {str(e)}"}]

    async def get_calendar_events(
        self, entity_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get calendar events, optionally filtered by entity_id."""
        try:
            if entity_id:
                _LOGGER.debug(
                    "Requesting calendar events for specific entity: %s", entity_id
                )
                return [await self.get_entity_state(entity_id)]

            _LOGGER.debug("Requesting all calendar events")
            return await self.get_entities_by_domain("calendar")
        except Exception as e:
            _LOGGER.exception("Error getting calendar events: %s", str(e))
            return [{"error": f"Error getting calendar events: {str(e)}"}]

    async def get_automations(self) -> List[Dict[str, Any]]:
        """Get all automations."""
        try:
            _LOGGER.debug("Requesting all automations")
            return await self.get_entities_by_domain("automation")
        except Exception as e:
            _LOGGER.exception("Error getting automations: %s", str(e))
            return [{"error": f"Error getting automations: {str(e)}"}]

    async def get_entity_registry(self) -> List[Dict]:
        """Get entity registry entries"""
        _LOGGER.debug("Requesting all entity registry entries")
        try:
            from homeassistant.helpers import entity_registry as er

            registry = er.async_get(self.hass)
            if not registry:
                return []
            return [
                {
                    "entity_id": entry.entity_id,
                    "device_id": entry.device_id,
                    "platform": entry.platform,
                    "disabled": entry.disabled,
                    "area_id": entry.area_id,
                    "original_name": entry.original_name,
                    "unique_id": entry.unique_id,
                }
                for entry in registry.entities.values()
            ]
        except Exception as e:
            _LOGGER.exception("Error getting entity registry entries: %s", str(e))
            return [{"error": f"Error getting entity registry entries: {str(e)}"}]

    async def get_device_registry(self) -> List[Dict]:
        """Get device registry entries"""
        _LOGGER.debug("Requesting all device registry entries")
        try:
            from homeassistant.helpers import device_registry as dr

            registry = dr.async_get(self.hass)
            if not registry:
                return []
            return [
                {
                    "id": device.id,
                    "name": device.name,
                    "model": device.model,
                    "manufacturer": device.manufacturer,
                    "sw_version": device.sw_version,
                    "hw_version": device.hw_version,
                    "connections": (
                        list(device.connections) if device.connections else []
                    ),
                    "identifiers": (
                        list(device.identifiers) if device.identifiers else []
                    ),
                    "area_id": device.area_id,
                    "disabled": device.disabled_by is not None,
                    "entry_type": (
                        device.entry_type.value if device.entry_type else None
                    ),
                    "name_by_user": device.name_by_user,
                }
                for device in registry.devices.values()
            ]
        except Exception as e:
            _LOGGER.exception("Error getting device registry entries: %s", str(e))
            return [{"error": f"Error getting device registry entries: {str(e)}"}]

    async def get_history(self, entity_id: str, hours: int = 24) -> List[Dict]:
        """Get historical state changes for an entity"""
        _LOGGER.debug("Requesting historical state changes for entity: %s", entity_id)
        try:
            from homeassistant.components import history

            now = dt_util.utcnow()
            start = now - timedelta(hours=hours)

            # Get history using the history component
            history_data = await self.hass.async_add_executor_job(
                getattr(history, "get_significant_states"),
                self.hass,
                start,
                now,
                [entity_id],
            )

            # Convert to serializable format
            result = []
            for entity_id_key, states in history_data.items():
                for state in states:
                    result.append(
                        {
                            "entity_id": state.entity_id,
                            "state": state.state,
                            "last_changed": state.last_changed.isoformat(),
                            "last_updated": state.last_updated.isoformat(),
                            "attributes": dict(state.attributes),
                        }
                    )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting history: %s", str(e))
            return [{"error": f"Error getting history: {str(e)}"}]

    async def get_logbook_entries(self, hours: int = 24) -> List[Dict]:
        """Get recent logbook entries"""
        _LOGGER.debug("Requesting recent logbook entries")
        try:
            from homeassistant.components import logbook

            now = dt_util.utcnow()
            start = now - timedelta(hours=hours)

            # Get logbook entries
            entries = await self.hass.async_add_executor_job(
                getattr(logbook, "get_events"), self.hass, start, now
            )

            # Convert to serializable format
            result = []
            for entry in entries:
                result.append(
                    {
                        "when": entry.get("when"),
                        "name": entry.get("name"),
                        "message": entry.get("message"),
                        "entity_id": entry.get("entity_id"),
                        "state": entry.get("state"),
                        "domain": entry.get("domain"),
                    }
                )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting logbook entries: %s", str(e))
            return [{"error": f"Error getting logbook entries: {str(e)}"}]

    async def get_area_registry(self) -> Dict[str, Any]:
        """Get area registry information"""
        _LOGGER.debug("Get area registry information")
        try:
            from homeassistant.helpers import area_registry as ar

            registry = ar.async_get(self.hass)
            if not registry:
                return {}

            result = {}
            for area in registry.areas.values():
                result[area.id] = {
                    "name": area.name,
                    "normalized_name": area.normalized_name,
                    "picture": area.picture,
                    "icon": area.icon,
                    "floor_id": area.floor_id,
                    "labels": list(area.labels) if area.labels else [],
                }
            return result
        except Exception as e:
            _LOGGER.exception("Error getting area registry: %s", str(e))
            return {"error": f"Error getting area registry: {str(e)}"}

    async def get_person_data(self) -> List[Dict]:
        """Get person tracking information"""
        _LOGGER.debug("Requesting person tracking information")
        try:
            result = []
            for state in self.hass.states.async_all("person"):
                result.append(
                    {
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "state": state.state,
                        "latitude": state.attributes.get("latitude"),
                        "longitude": state.attributes.get("longitude"),
                        "source": state.attributes.get("source"),
                        "gps_accuracy": state.attributes.get("gps_accuracy"),
                        "last_changed": (
                            state.last_changed.isoformat()
                            if state.last_changed
                            else None
                        ),
                    }
                )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting person tracking information: %s", str(e))
            return [{"error": f"Error getting person tracking information: {str(e)}"}]

    async def get_statistics(self, entity_id: str) -> Dict:
        """Get statistics for an entity"""
        _LOGGER.debug("Requesting statistics for entity: %s", entity_id)
        try:
            from homeassistant.components import recorder

            # Check if recorder is available
            if not self.hass.data.get(recorder.DATA_INSTANCE):
                return {"error": "Recorder component is not available"}

            # from homeassistant.components.recorder.statistics import get_latest_short_term_statistics
            import homeassistant.components.recorder.statistics as stats_module

            # Get latest statistics
            stats = await self.hass.async_add_executor_job(
                # get_latest_short_term_statistics,
                stats_module.get_last_short_term_statistics,
                self.hass,
                1,
                entity_id,
                True,
                set(),
            )

            if entity_id in stats:
                stat_data = stats[entity_id][0] if stats[entity_id] else {}
                return {
                    "entity_id": entity_id,
                    "start": stat_data.get("start"),
                    "mean": stat_data.get("mean"),
                    "min": stat_data.get("min"),
                    "max": stat_data.get("max"),
                    "last_reset": stat_data.get("last_reset"),
                    "state": stat_data.get("state"),
                    "sum": stat_data.get("sum"),
                }
            else:
                return {"error": f"No statistics available for entity {entity_id}"}
        except Exception as e:
            _LOGGER.exception("Error getting statistics: %s", str(e))
            return {"error": f"Error getting statistics: {str(e)}"}

    async def get_scenes(self) -> List[Dict]:
        """Get scene configurations"""
        _LOGGER.debug("Requesting scene configurations")
        try:
            result = []
            for state in self.hass.states.async_all("scene"):
                result.append(
                    {
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "last_activated": state.attributes.get("last_activated"),
                        "icon": state.attributes.get("icon"),
                        "last_changed": (
                            state.last_changed.isoformat()
                            if state.last_changed
                            else None
                        ),
                    }
                )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting scene configurations: %s", str(e))
            return [{"error": f"Error getting scene configurations: {str(e)}"}]

    async def get_weather_data(self) -> Dict[str, Any]:
        """Get weather data from any available weather entity in the system."""
        try:
            # Find all weather entities
            weather_entities = [
                state
                for state in self.hass.states.async_all()
                if state.domain == "weather"
            ]

            if not weather_entities:
                return {
                    "error": "No weather entities found in the system. Please add a weather integration."
                }

            # Use the first available weather entity
            state = weather_entities[0]
            _LOGGER.debug("Using weather entity: %s", state.entity_id)

            # Get all available attributes
            all_attributes = state.attributes
            _LOGGER.debug(
                "Available weather attributes: %s", json.dumps(all_attributes)
            )

            # Get forecast data
            forecast = all_attributes.get("forecast", [])

            # Process forecast data
            processed_forecast = []
            for day in forecast:
                forecast_entry = {
                    "datetime": day.get("datetime"),
                    "temperature": day.get("temperature"),
                    "condition": day.get("condition"),
                    "precipitation": day.get("precipitation"),
                    "precipitation_probability": day.get("precipitation_probability"),
                    "humidity": day.get("humidity"),
                    "wind_speed": day.get("wind_speed"),
                    "wind_bearing": day.get("wind_bearing"),
                }
                # Only add entries that have at least some data
                if any(v is not None for v in forecast_entry.values()):
                    processed_forecast.append(forecast_entry)

            # Get current weather data
            current = {
                "entity_id": state.entity_id,
                "temperature": all_attributes.get("temperature"),
                "humidity": all_attributes.get("humidity"),
                "pressure": all_attributes.get("pressure"),
                "wind_speed": all_attributes.get("wind_speed"),
                "wind_bearing": all_attributes.get("wind_bearing"),
                "condition": state.state,
                "forecast_available": len(processed_forecast) > 0,
            }

            # Log the processed data for debugging
            _LOGGER.debug(
                "Processed weather data: %s",
                json.dumps(
                    {"current": current, "forecast_count": len(processed_forecast)}
                ),
            )

            return {"current": current, "forecast": processed_forecast}
        except Exception as e:
            _LOGGER.exception("Error getting weather data: %s", str(e))
            return {"error": f"Error getting weather data: {str(e)}"}

    async def create_automation(
        self, automation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new automation with validation and sanitization."""
        try:
            _LOGGER.debug(
                "Creating automation with config: %s", json.dumps(automation_config)
            )

            # Validate required fields
            if not all(
                key in automation_config for key in ["alias", "trigger", "action"]
            ):
                return {"error": "Missing required fields in automation configuration"}

            # Sanitize configuration
            sanitized_config = self._sanitize_automation_config(automation_config)

            # Generate a unique ID for the automation
            automation_id = f"ai_agent_auto_{int(time.time() * 1000)}"

            # Create the automation entry
            automation_entry = {
                "id": automation_id,
                "alias": sanitized_config["alias"],
                "description": sanitized_config.get("description", ""),
                "trigger": sanitized_config["trigger"],
                "condition": sanitized_config.get("condition", []),
                "action": sanitized_config["action"],
                "mode": sanitized_config.get("mode", "single"),
            }

            # Generate filename based on timestamp and alias
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_alias = self._sanitize_filename(sanitized_config["alias"])
            filename = f"{timestamp}_{sanitized_alias}.yaml"
            
            # Ensure automations directory exists
            automations_dir = self.hass.config.path("automations")
            await self.hass.async_add_executor_job(
                lambda: os.makedirs(automations_dir, exist_ok=True)
            )
            
            automation_file_path = os.path.join(automations_dir, filename)
            
            # Check if file already exists (very unlikely with timestamp, but safety check)
            if await self.hass.async_add_executor_job(
                lambda: os.path.exists(automation_file_path)
            ):
                return {
                    "error": f"Automation file '{filename}' already exists"
                }

            # Write automation to individual file as array
            await self.hass.async_add_executor_job(
                lambda: yaml.dump(
                    [automation_entry],
                    open(automation_file_path, "w"),
                    default_flow_style=False,
                )
            )

            # Reload automations
            await self.hass.services.async_call("automation", "reload")

            # Clear automation-related caches
            self._cache.clear()

            return {
                "success": True,
                "message": f"Automation '{automation_entry['alias']}' created successfully as {filename}",
                "automation_path": automation_file_path,
            }

        except Exception as e:
            _LOGGER.exception("Error creating automation: %s", str(e))
            return {"error": f"Error creating automation: {str(e)}"}

    async def remove_automation(self, automation_id: str) -> Dict[str, Any]:
        """Remove an automation by entity_id or alias and delete its file."""
        try:
            _LOGGER.debug("Removing automation: %s", automation_id)

            # Get all automations to find the target
            automations = await self.get_automations()
            if isinstance(automations, list) and len(automations) > 0 and "error" in automations[0]:
                return automations[0]  # Return error from get_automations

            target_automation = None
            
            # Search by entity_id or alias
            for automation in automations:
                entity_id = automation.get("entity_id", "")
                alias = automation.get("attributes", {}).get("friendly_name", "")
                
                if (entity_id == automation_id or 
                    alias == automation_id or
                    entity_id == f"automation.{automation_id}"):
                    target_automation = automation
                    break

            if not target_automation:
                return {"error": f"Automation '{automation_id}' not found"}

            # Get the automation's entity_id for service call
            entity_id = target_automation.get("entity_id")
            automation_alias = target_automation.get("attributes", {}).get("friendly_name", "")

            # First, turn off the automation to unregister it from Home Assistant
            try:
                await self.hass.services.async_call(
                    "automation",
                    "turn_off",
                    {"entity_id": entity_id}
                )
            except Exception as e:
                _LOGGER.warning("Could not turn off automation %s: %s", entity_id, str(e))

            # Find and delete the automation file
            automations_dir = self.hass.config.path("automations")
            automation_file_deleted = False
            
            if await self.hass.async_add_executor_job(lambda: os.path.exists(automations_dir)):
                # Search for files containing this automation
                for filename in await self.hass.async_add_executor_job(lambda: os.listdir(automations_dir)):
                    if not filename.endswith('.yaml'):
                        continue
                        
                    file_path = os.path.join(automations_dir, filename)
                    try:
                        # Read and parse the automation file
                        with open(file_path, 'r') as f:
                            file_content = yaml.safe_load(f)
                            
                        if isinstance(file_content, list):
                            # Check if any automation in this file matches
                            for automation in file_content:
                                if (automation.get("alias") == automation_alias or
                                    automation.get("id", "").endswith(automation_id.replace("automation.", ""))):
                                    # Delete the entire file
                                    await self.hass.async_add_executor_job(lambda: os.remove(file_path))
                                    automation_file_deleted = True
                                    _LOGGER.info("Deleted automation file: %s", filename)
                                    break
                                    
                        if automation_file_deleted:
                            break
                            
                    except Exception as e:
                        _LOGGER.warning("Error reading automation file %s: %s", filename, str(e))
                        continue

            # Reload automations to remove it from Home Assistant registry
            await self.hass.services.async_call("automation", "reload")
            
            # Additional step: Remove the entity from the Entity Registry if it still exists
            try:
                from homeassistant.helpers import entity_registry as er
                entity_registry = er.async_get(self.hass)
                
                # Try multiple times to remove the entity from registry (it may take time after reload)
                for attempt in range(3):
                    await asyncio.sleep(0.5 * (attempt + 1))  # Wait increasingly longer
                    
                    entity_entry = entity_registry.async_get(entity_id)
                    if entity_entry:
                        _LOGGER.info("Removing entity %s from registry (attempt %d)", entity_id, attempt + 1)
                        entity_registry.async_remove(entity_id)
                        break
                    else:
                        # Entity already removed from registry
                        break
                    
            except Exception as e:
                _LOGGER.warning("Could not remove entity %s from registry: %s", entity_id, str(e))

            # Clear automation-related caches
            self._cache.clear()

            if automation_file_deleted:
                return {
                    "success": True,
                    "message": f"Automation '{automation_alias}' removed successfully and file deleted"
                }
            else:
                return {
                    "success": True,
                    "message": f"Automation '{automation_alias}' unregistered from Home Assistant (file not found or could not be deleted)"
                }

        except Exception as e:
            _LOGGER.exception("Error removing automation: %s", str(e))
            return {"error": f"Error removing automation: {str(e)}"}



    async def process_query(
        self, user_query: str, provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a user query with input validation and rate limiting."""
        try:
            if not user_query or not isinstance(user_query, str):
                return {"success": False, "error": "Invalid query format"}

            # Get the correct configuration for the requested provider
            if provider and provider in self.hass.data[DOMAIN]["configs"]:
                config = self.hass.data[DOMAIN]["configs"][provider]
            else:
                config = self.config

            _LOGGER.debug(f"Processing query with provider: {provider}")
            _LOGGER.debug(f"Using config: {json.dumps(config, default=str)}")

            selected_provider = provider or config.get("ai_provider", "llama")
            models_config = config.get("models", {})

            provider_config = {
                "openai": {
                    "token_key": "openai_token",
                    "model": models_config.get("openai", "gpt-3.5-turbo"),
                    "client_class": OpenAIClient,
                },
                "gemini": {
                    "token_key": "gemini_token",
                    "model": models_config.get("gemini", "gemini-1.5-flash"),
                    "client_class": GeminiClient,
                },
                "openrouter": {
                    "token_key": "openrouter_token",
                    "model": models_config.get("openrouter", "openai/gpt-4o"),
                    "client_class": OpenRouterClient,
                },
                "llama": {
                    "token_key": "llama_token",
                    "model": models_config.get(
                        "llama", "Llama-4-Maverick-17B-128E-Instruct-FP8"
                    ),
                    "client_class": LlamaClient,
                },
                "anthropic": {
                    "token_key": "anthropic_token",
                    "model": models_config.get(
                        "anthropic", "claude-3-5-sonnet-20241022"
                    ),
                    "client_class": AnthropicClient,
                },
                "local": {
                    "token_key": "local_url",
                    "model": models_config.get("local", ""),
                    "client_class": LocalClient,
                },
            }

            # Validate provider and get configuration
            if selected_provider not in provider_config:
                _LOGGER.warning(
                    f"Invalid provider {selected_provider}, falling back to llama"
                )
                selected_provider = "llama"

            provider_settings = provider_config[selected_provider]
            token = self.config.get(provider_settings["token_key"])

            # Validate token/URL
            if not token:
                error_msg = f"No {'URL' if selected_provider == 'local' else 'token'} configured for provider {selected_provider}"
                _LOGGER.error(error_msg)
                return {"success": False, "error": error_msg}

            # Initialize client
            try:
                if selected_provider == "local":
                    # LocalClient takes (url, model)
                    self.ai_client = provider_settings["client_class"](
                        url=token, model=provider_settings["model"]
                    )
                else:
                    # Other clients take (token, model)
                    self.ai_client = provider_settings["client_class"](
                        token=token, model=provider_settings["model"]
                    )
                _LOGGER.debug(
                    f"Initialized {selected_provider} client with model {provider_settings['model']}"
                )
            except Exception as e:
                error_msg = f"Error initializing {selected_provider} client: {str(e)}"
                _LOGGER.error(error_msg)
                return {"success": False, "error": error_msg}

            # Process the query with rate limiting and retries
            if not self._check_rate_limit():
                return {
                    "success": False,
                    "error": "Rate limit exceeded. Please wait before trying again.",
                }

            # Sanitize user input
            user_query = user_query.strip()[:1000]  # Limit length and trim whitespace

            _LOGGER.debug("Processing new query: %s", user_query)

            # Check cache for identical query
            cache_key = f"query_{hash(user_query)}"
            cached_result = self._get_cached_data(cache_key)
            if cached_result:
                return (
                    dict(cached_result)
                    if isinstance(cached_result, dict)
                    else {"error": "Invalid cached result"}
                )

            # Add system message to conversation if it's the first message
            if not self.conversation_history:
                _LOGGER.debug("Adding system message to new conversation")
                self.conversation_history.append(self.system_prompt)

            # Add user query to conversation
            self.conversation_history.append({"role": "user", "content": user_query})
            _LOGGER.debug("Added user query to conversation history")

            max_iterations = 5  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                _LOGGER.debug(f"Processing iteration {iteration} of {max_iterations}")

                try:
                    # Get AI response
                    _LOGGER.debug("Requesting response from AI provider")
                    response = await self._get_ai_response()
                    _LOGGER.debug("Received response from AI provider: %s", response)

                    try:
                        # Try to parse the response as JSON with simplified approach
                        response_clean = response.strip()

                        # Remove potential BOM and other invisible characters
                        import codecs

                        if response_clean.startswith(codecs.BOM_UTF8.decode("utf-8")):
                            response_clean = response_clean[1:]

                        # Remove other common invisible characters
                        invisible_chars = [
                            "\ufeff",
                            "\u200b",
                            "\u200c",
                            "\u200d",
                            "\u2060",
                        ]
                        for char in invisible_chars:
                            response_clean = response_clean.replace(char, "")

                        _LOGGER.debug(
                            "Cleaned response length: %d", len(response_clean)
                        )
                        _LOGGER.debug(
                            "Cleaned response first 100 chars: %s", response_clean[:100]
                        )
                        _LOGGER.debug(
                            "Cleaned response last 100 chars: %s", response_clean[-100:]
                        )

                        # Simple strategy: try to parse the cleaned response directly
                        response_data = None
                        try:
                            _LOGGER.debug("Attempting basic JSON parse...")
                            response_data = json.loads(response_clean)
                            _LOGGER.debug("Basic JSON parse succeeded!")
                        except json.JSONDecodeError as e:
                            _LOGGER.warning("Basic JSON parse failed: %s", str(e))
                            _LOGGER.debug("JSON error position: %d", e.pos)
                            if e.pos < len(response_clean):
                                _LOGGER.debug(
                                    "Character at error position: %s (ord: %d)",
                                    repr(response_clean[e.pos]),
                                    ord(response_clean[e.pos]),
                                )
                                _LOGGER.debug(
                                    "Context around error: %s",
                                    repr(
                                        response_clean[max(0, e.pos - 10) : e.pos + 10]
                                    ),
                                )

                            # Fallback: try to extract JSON by finding the first { and last }
                            json_start = response_clean.find("{")
                            json_end = response_clean.rfind("}")

                            if (
                                json_start != -1
                                and json_end != -1
                                and json_end > json_start
                            ):
                                json_part = response_clean[json_start : json_end + 1]
                                _LOGGER.debug(
                                    "Trying fallback extraction from pos %d to %d",
                                    json_start,
                                    json_end,
                                )
                                _LOGGER.debug("Extracted JSON: %s", json_part[:200])

                                try:
                                    response_data = json.loads(json_part)
                                    _LOGGER.debug("Fallback JSON extraction succeeded!")
                                except json.JSONDecodeError as e2:
                                    _LOGGER.warning(
                                        "Fallback JSON extraction also failed: %s",
                                        str(e2),
                                    )
                                    raise e  # Re-raise the original error
                            else:
                                _LOGGER.warning(
                                    "Could not find JSON boundaries in response"
                                )
                                raise e  # Re-raise the original error

                        if response_data is None:
                            raise json.JSONDecodeError(
                                "All parsing strategies failed", response_clean, 0
                            )

                        _LOGGER.debug("Successfully parsed JSON response")
                        _LOGGER.debug(
                            "Parsed response type: %s",
                            response_data.get("request_type", "unknown"),
                        )

                        # Check if this is a data request (either format)
                        data_request_types = [
                            "get_entity_state",
                            "get_entities_by_domain",
                            "get_entities_by_area",
                            "get_entities",
                            "get_calendar_events",
                            "get_automations",
                            "get_entity_registry",
                            "get_device_registry",
                            "get_weather_data",
                            "get_area_registry",
                            "get_history",
                            "get_logbook_entries",
                            "get_person_data",
                            "get_statistics",
                            "get_scenes",
                            "set_entity_state",
                        ]

                        if (
                            response_data.get("request_type") == "data_request"
                            or response_data.get("request_type") in data_request_types
                        ):
                            # Handle data request (both standard format and direct request type)
                            if response_data.get("request_type") == "data_request":
                                request_type = response_data.get("request")
                            else:
                                request_type = response_data.get("request_type")
                            parameters = response_data.get("parameters", {})
                            _LOGGER.debug(
                                "Processing data request: %s with parameters: %s",
                                request_type,
                                json.dumps(parameters),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Get requested data
                            data: Union[Dict[str, Any], List[Dict[str, Any]]]
                            if request_type == "get_entity_state":
                                data = await self.get_entity_state(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_entities_by_domain":
                                data = await self.get_entities_by_domain(
                                    parameters.get("domain")
                                )
                            elif request_type == "get_entities_by_area":
                                data = await self.get_entities_by_area(
                                    parameters.get("area_id")
                                )
                            elif request_type == "get_entities":
                                data = await self.get_entities(
                                    area_id=parameters.get("area_id"),
                                    area_ids=parameters.get("area_ids"),
                                )
                            elif request_type == "get_calendar_events":
                                data = await self.get_calendar_events(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_automations":
                                data = await self.get_automations()
                            elif request_type == "get_entity_registry":
                                data = await self.get_entity_registry()
                            elif request_type == "get_device_registry":
                                data = await self.get_device_registry()
                            elif request_type == "get_weather_data":
                                data = await self.get_weather_data()
                            elif request_type == "get_area_registry":
                                data = await self.get_area_registry()
                            elif request_type == "get_history":
                                data = await self.get_history(
                                    parameters.get("entity_id"),
                                    parameters.get("hours", 24),
                                )
                            elif request_type == "get_logbook_entries":
                                data = await self.get_logbook_entries(
                                    parameters.get("hours", 24)
                                )
                            elif request_type == "get_person_data":
                                data = await self.get_person_data()
                            elif request_type == "get_statistics":
                                data = await self.get_statistics(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_scenes":
                                data = await self.get_scenes()
                            elif request_type == "set_entity_state":
                                data = await self.set_entity_state(
                                    parameters.get("entity_id"),
                                    parameters.get("state"),
                                    parameters.get("attributes"),
                                )
                            elif request_type == "create_automation":
                                data = await self.create_automation(
                                    parameters.get("automation")
                                )
                            elif request_type == "remove_automation":
                                data = await self.remove_automation(
                                    parameters.get("automation_id")
                                )
                            else:
                                data = {
                                    "error": f"Unknown request type: {request_type}"
                                }
                                _LOGGER.warning(
                                    "Unknown request type: %s", request_type
                                )

                            # Check if any data request resulted in an error
                            if isinstance(data, dict) and "error" in data:
                                return {"success": False, "error": data["error"]}
                            elif isinstance(data, list) and any(
                                "error" in item
                                for item in data
                                if isinstance(item, dict)
                            ):
                                errors = [
                                    item["error"]
                                    for item in data
                                    if isinstance(item, dict) and "error" in item
                                ]
                                return {"success": False, "error": "; ".join(errors)}

                            _LOGGER.debug(
                                "Retrieved data for request: %s",
                                json.dumps(data, default=str),
                            )

                            # Add data to conversation as a system message
                            self.conversation_history.append(
                                {
                                    "role": "system",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            continue

                        elif response_data.get("request_type") == "final_response":
                            # Add final response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Return final response with structured format
                            _LOGGER.debug(
                                "Received final response: %s",
                                response_data.get("response"),
                            )
                            # Return the complete structured response for conversation.py
                            result = response_data
                            self._set_cached_data(cache_key, result)
                            return result
                        elif response_data.get("request_type") in [
                            "get_entities",
                            "get_entities_by_area",
                        ]:
                            # Handle direct get_entities request (for backward compatibility)
                            parameters = response_data.get("parameters", {})
                            _LOGGER.debug(
                                "Processing direct get_entities request with parameters: %s",
                                json.dumps(parameters),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Get entities data
                            if response_data.get("request_type") == "get_entities":
                                data = await self.get_entities(
                                    area_id=parameters.get("area_id"),
                                    area_ids=parameters.get("area_ids"),
                                )
                            else:  # get_entities_by_area
                                data = await self.get_entities_by_area(
                                    parameters.get("area_id")
                                )

                            _LOGGER.debug(
                                "Retrieved %d entities",
                                len(data) if isinstance(data, list) else 1,
                            )

                            # Add data to conversation as a system message
                            self.conversation_history.append(
                                {
                                    "role": "system",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            continue
                        elif response_data.get("request_type") == "call_service":
                            # Handle service call request
                            domain = response_data.get("domain")
                            service = response_data.get("service")
                            target = response_data.get("target", {})
                            service_data = response_data.get("service_data", {})
                            
                            # Check for invalid automation.create calls and convert them to create_automation
                            if domain == "automation" and service == "create":
                                _LOGGER.warning("AI attempted to use invalid automation.create service call - converting to create_automation format")
                                
                                # Convert service_data to create_automation format
                                automation_data = service_data
                                
                                # Create the automation directly
                                automation_result = await self.create_automation(automation_data)
                                
                                # Return appropriate response
                                if automation_result.get("success"):
                                    result = {
                                        "success": True,
                                        "response": automation_result.get("message", "Automation created successfully")
                                    }
                                else:
                                    result = {
                                        "success": False,
                                        "response": automation_result.get("error", "Failed to create automation")
                                    }
                                
                                self._set_cached_data(cache_key, result)
                                return result

                            # Resolve nested requests in target
                            if target and "entity_id" in target:
                                entity_id_value = target["entity_id"]
                                if (
                                    isinstance(entity_id_value, dict)
                                    and "request_type" in entity_id_value
                                ):
                                    # This is a nested request, resolve it
                                    nested_request_type = entity_id_value.get(
                                        "request_type"
                                    )
                                    nested_parameters = entity_id_value.get(
                                        "parameters", {}
                                    )

                                    _LOGGER.debug(
                                        "Resolving nested request: %s with parameters: %s",
                                        nested_request_type,
                                        json.dumps(nested_parameters),
                                    )

                                    # Resolve the nested request
                                    if nested_request_type == "get_entities":
                                        entities_data = await self.get_entities(
                                            area_id=nested_parameters.get("area_id"),
                                            area_ids=nested_parameters.get("area_ids"),
                                        )
                                    elif nested_request_type == "get_entities_by_area":
                                        entities_data = await self.get_entities_by_area(
                                            nested_parameters.get("area_id")
                                        )
                                    elif (
                                        nested_request_type == "get_entities_by_domain"
                                    ):
                                        entities_data = (
                                            await self.get_entities_by_domain(
                                                nested_parameters.get("domain")
                                            )
                                        )
                                    else:
                                        _LOGGER.error(
                                            "Unsupported nested request type: %s",
                                            nested_request_type,
                                        )
                                        return {
                                            "success": False,
                                            "error": f"Unsupported nested request type: {nested_request_type}",
                                        }

                                    # Extract entity IDs from the resolved data
                                    if isinstance(entities_data, list):
                                        entity_ids = [
                                            entity.get("entity_id")
                                            for entity in entities_data
                                            if entity.get("entity_id")
                                        ]
                                        target["entity_id"] = entity_ids
                                        _LOGGER.debug(
                                            "Resolved nested request to entity IDs: %s",
                                            entity_ids,
                                        )
                                    else:
                                        _LOGGER.error(
                                            "Nested request returned unexpected data format"
                                        )
                                        return {
                                            "success": False,
                                            "error": "Nested request returned unexpected data format",
                                        }

                            # Handle backward compatibility with old format
                            if not domain or not service:
                                request = response_data.get("request")
                                parameters = response_data.get("parameters", {})

                                if request and "entity_id" in parameters:
                                    entity_id = parameters["entity_id"]
                                    # Infer domain from entity_id
                                    if "." in entity_id:
                                        domain = entity_id.split(".")[0]
                                        service = request
                                        target = {"entity_id": entity_id}
                                        # Remove entity_id from parameters to avoid duplication
                                        service_data = {
                                            k: v
                                            for k, v in parameters.items()
                                            if k != "entity_id"
                                        }
                                        _LOGGER.debug(
                                            "Converted old format: domain=%s, service=%s",
                                            domain,
                                            service,
                                        )

                            _LOGGER.debug(
                                "Processing service call: %s.%s with target: %s and data: %s",
                                domain,
                                service,
                                json.dumps(target),
                                json.dumps(service_data),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Call the service
                            data = await self.call_service(
                                domain, service, target, service_data
                            )

                            # Check if service call resulted in an error
                            if isinstance(data, dict) and "error" in data:
                                return {"success": False, "error": data["error"]}

                            _LOGGER.debug(
                                "Service call completed: %s",
                                json.dumps(data, default=str),
                            )

                            # Add data to conversation as a system message
                            self.conversation_history.append(
                                {
                                    "role": "system",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            continue
                        else:
                            _LOGGER.warning(
                                "Unknown response type: %s",
                                response_data.get("request_type"),
                            )
                            return {
                                "success": False,
                                "error": f"Unknown response type: {response_data.get('request_type')}",
                            }

                    except json.JSONDecodeError as e:
                        # Check if this is a local provider that might have already wrapped the response
                        provider = self.config.get("ai_provider", "unknown")
                        if provider == "local":
                            _LOGGER.debug(
                                "Local provider returned non-JSON response (this is normal and handled): %s",
                                response[:200],
                            )
                        else:
                            # Log more of the response to help with debugging for non-local providers
                            response_preview = (
                                response[:1000] if len(response) > 1000 else response
                            )
                            _LOGGER.warning(
                                "Failed to parse response as JSON: %s. Response length: %d. Response preview: %s",
                                str(e),
                                len(response),
                                response_preview,
                            )

                            # Log additional debugging information
                            _LOGGER.debug(
                                "First 50 characters as bytes: %s",
                                response[:50].encode("utf-8") if response else b"",
                            )
                            _LOGGER.debug(
                                "Response starts with: %s",
                                repr(response[:10]) if response else "None",
                            )

                        # Also log the response to a separate debug file for detailed analysis (non-local providers only)
                        if provider != "local":
                            try:
                                import os

                                debug_dir = "/config/ai_agent_ha_debug"

                                def write_debug_file():
                                    if not os.path.exists(debug_dir):
                                        os.makedirs(debug_dir)

                                    import datetime

                                    timestamp = datetime.datetime.now().strftime(
                                        "%Y%m%d_%H%M%S"
                                    )
                                    debug_file = os.path.join(
                                        debug_dir, f"failed_response_{timestamp}.txt"
                                    )

                                    with open(debug_file, "w", encoding="utf-8") as f:
                                        f.write(f"Timestamp: {timestamp}\n")
                                        f.write(f"Provider: {provider}\n")
                                        f.write(f"Error: {str(e)}\n")
                                        f.write(f"Response length: {len(response)}\n")
                                        f.write(
                                            f"Response bytes: {response.encode('utf-8') if response else b''}\n"
                                        )
                                        f.write(f"Response repr: {repr(response)}\n")
                                        f.write(f"Full response:\n{response}\n")

                                    return debug_file

                                # Run file operations in executor to avoid blocking
                                debug_file = await self.hass.async_add_executor_job(
                                    write_debug_file
                                )
                                _LOGGER.info(
                                    "Failed response saved to debug file: %s",
                                    debug_file,
                                )
                            except Exception as debug_error:
                                _LOGGER.debug(
                                    "Could not save debug file: %s", str(debug_error)
                                )


                        # If response is not valid JSON, try to wrap it as a final response
                        try:
                            # Truncate extremely long responses to prevent memory issues
                            response_to_wrap = response
                            if len(response) > 50000:
                                response_to_wrap = (
                                    response[:5000]
                                    + "... [Response truncated due to excessive length]"
                                )
                                _LOGGER.warning(
                                    "Truncated extremely long response from %d to 5000 characters",
                                    len(response),
                                )

                            wrapped_response = {
                                "request_type": "final_response",
                                "response": response_to_wrap,
                            }
                            # Return structured response directly
                            result = wrapped_response
                            _LOGGER.debug("Wrapped non-JSON response as final_response")
                        except Exception as wrap_error:
                            _LOGGER.error(
                                "Failed to wrap response: %s", str(wrap_error)
                            )
                            result = {
                                "success": False,
                                "error": f"Invalid response format: {str(e)}",
                            }

                        self._set_cached_data(cache_key, result)
                        return result

                except Exception as e:
                    _LOGGER.exception("Error processing AI response: %s", str(e))
                    return {
                        "success": False,
                        "error": f"Error processing AI response: {str(e)}",
                    }

            # If we've reached max iterations without a final response
            _LOGGER.warning("Reached maximum iterations without final response")
            result = {
                "success": False,
                "error": "Maximum iterations reached without final response",
            }
            self._set_cached_data(cache_key, result)
            return result

        except Exception as e:
            _LOGGER.exception("Error in process_query: %s", str(e))
            return {"success": False, "error": f"Error in process_query: {str(e)}"}

    async def _get_ai_response(self) -> str:
        """Get response from the selected AI provider with retries and rate limiting."""
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded. Please try again later.")
        retry_count = 0
        last_error = None
        # Limit conversation history to last 10 messages to prevent token overflow
        recent_messages = (
            self.conversation_history[-10:]
            if len(self.conversation_history) > 10
            else self.conversation_history
        )
        # Ensure system prompt is always the first message
        if not recent_messages or recent_messages[0].get("role") != "system":
            recent_messages = [self.system_prompt] + recent_messages

        _LOGGER.debug("Sending %d messages to AI provider", len(recent_messages))
        _LOGGER.debug("AI provider: %s", self.config.get("ai_provider", "unknown"))

        while retry_count < self._max_retries:
            try:
                _LOGGER.debug(
                    "Attempt %d/%d: Calling AI client",
                    retry_count + 1,
                    self._max_retries,
                )
                response = await self.ai_client.get_response(recent_messages)
                _LOGGER.debug(
                    "AI client returned response of length: %d", len(response or "")
                )
                _LOGGER.debug("AI response preview: %s", (response or "")[:200])

                # Check for extremely long responses that might indicate model issues
                if response and len(response) > 50000:
                    _LOGGER.warning(
                        "AI returned extremely long response (%d characters), this may indicate a model issue",
                        len(response),
                    )
                    # Check for repetitive patterns that indicate a corrupted response
                    if response.count("for its use in various fields") > 50:
                        _LOGGER.error(
                            "Detected corrupted repetitive response, aborting this iteration"
                        )
                        raise Exception(
                            "AI generated corrupted response with repetitive text. Please try again with a clearer request."
                        )

                # Check if response is empty
                if not response or response.strip() == "":
                    _LOGGER.warning(
                        "AI client returned empty response on attempt %d",
                        retry_count + 1,
                    )
                    if retry_count + 1 >= self._max_retries:
                        raise Exception(
                            "AI provider returned empty response after all retries"
                        )
                    else:
                        retry_count += 1
                        await asyncio.sleep(self._retry_delay * retry_count)
                        continue

                return str(response)
            except Exception as e:
                _LOGGER.error(
                    "AI client error on attempt %d: %s", retry_count + 1, str(e)
                )
                last_error = e
                retry_count += 1
                if retry_count < self._max_retries:
                    await asyncio.sleep(self._retry_delay * retry_count)
                continue
        raise Exception(
            f"Failed after {retry_count} retries. Last error: {str(last_error)}"
        )

    def clear_conversation_history(self) -> None:
        """Clear the conversation history and cache."""
        self.conversation_history = []
        self._cache.clear()
        _LOGGER.debug("Conversation history and cache cleared")

    async def set_entity_state(
        self, entity_id: str, state: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Set the state of an entity."""
        try:
            _LOGGER.debug(
                "Setting state for entity %s to %s with attributes: %s",
                entity_id,
                state,
                json.dumps(attributes or {}),
            )

            # Validate entity exists
            if not self.hass.states.get(entity_id):
                return {"error": f"Entity {entity_id} not found"}

            # Call the appropriate service based on the domain
            domain = entity_id.split(".")[0]

            if domain == "light":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                service_data = {"entity_id": entity_id}
                if attributes and service == "turn_on":
                    service_data.update(attributes)
                await self.hass.services.async_call("light", service, service_data)

            elif domain == "switch":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                await self.hass.services.async_call(
                    "switch", service, {"entity_id": entity_id}
                )

            elif domain == "cover":
                if state.lower() in ["open", "up"]:
                    service = "open_cover"
                elif state.lower() in ["close", "down"]:
                    service = "close_cover"
                elif state.lower() == "stop":
                    service = "stop_cover"
                else:
                    return {"error": f"Invalid state {state} for cover entity"}
                await self.hass.services.async_call(
                    "cover", service, {"entity_id": entity_id}
                )

            elif domain == "climate":
                service_data = {"entity_id": entity_id}
                if state.lower() in ["on", "true", "1"]:
                    service = "turn_on"
                elif state.lower() in ["off", "false", "0"]:
                    service = "turn_off"
                elif state.lower() in ["heat", "cool", "dry", "fan_only", "auto"]:
                    service = "set_hvac_mode"
                    service_data["hvac_mode"] = state.lower()
                else:
                    return {"error": f"Invalid state {state} for climate entity"}
                await self.hass.services.async_call("climate", service, service_data)

            elif domain == "fan":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                service_data = {"entity_id": entity_id}
                if attributes and service == "turn_on":
                    service_data.update(attributes)
                await self.hass.services.async_call("fan", service, service_data)

            else:
                # For other domains, try to set the state directly
                self.hass.states.async_set(entity_id, state, attributes or {})

            # Get the new state to confirm the change
            new_state = self.hass.states.get(entity_id)
            return {
                "success": True,
                "entity_id": entity_id,
                "new_state": new_state.state,
                "new_attributes": new_state.attributes,
            }

        except Exception as e:
            _LOGGER.exception("Error setting entity state: %s", str(e))
            return {"error": f"Error setting entity state: {str(e)}"}

    async def call_service(
        self,
        domain: str,
        service: str,
        target: Optional[Dict[str, Any]] = None,
        service_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a Home Assistant service."""
        try:
            _LOGGER.debug(
                "Calling service %s.%s with target: %s and data: %s",
                domain,
                service,
                json.dumps(target or {}),
                json.dumps(service_data or {}),
            )

            # Prepare the service call data
            call_data = {}

            # Add target entities if provided
            if target:
                if "entity_id" in target:
                    entity_ids = target["entity_id"]
                    if isinstance(entity_ids, list):
                        call_data["entity_id"] = entity_ids
                    else:
                        call_data["entity_id"] = [entity_ids]

                # Add other target properties
                for key, value in target.items():
                    if key != "entity_id":
                        call_data[key] = value

            # Add service data if provided
            if service_data:
                call_data.update(service_data)

            _LOGGER.debug("Final service call data: %s", json.dumps(call_data))

            # Check for invalid automation.create calls
            if domain == "automation" and service == "create":
                _LOGGER.error("Invalid automation.create service call detected - automation.create service does not exist")
                return {"error": "automation.create service does not exist. Use create_automation command instead."}

            # Call the service
            await self.hass.services.async_call(domain, service, call_data)

            # Get the updated states of affected entities
            result_entities = []
            if "entity_id" in call_data:
                for entity_id in call_data["entity_id"]:
                    state = self.hass.states.get(entity_id)
                    if state:
                        result_entities.append(
                            {
                                "entity_id": entity_id,
                                "state": state.state,
                                "attributes": dict(state.attributes),
                            }
                        )

            return {
                "success": True,
                "service": f"{domain}.{service}",
                "entities_affected": result_entities,
                "message": f"Successfully called {domain}.{service}",
            }

        except Exception as e:
            _LOGGER.exception(
                "Error calling service %s.%s: %s", domain, service, str(e)
            )
            return {"error": f"Error calling service {domain}.{service}: {str(e)}"}

