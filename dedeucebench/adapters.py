from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelReply:
    content: str
    tool_calls: List[
        Dict[str, Any]
    ]  # [{id, type='function', function: {name, arguments}}]
    usage: Dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}


class BaseAdapter:
    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = dict(kwargs)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelReply:
        raise NotImplementedError


def _split_system_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    remaining: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                system_parts.append(content)
        else:
            remaining.append(msg)
    system_prompt = "\n\n".join(system_parts)
    return system_prompt, remaining


def _tool_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


class OpenAICompatAdapter(BaseAdapter):
    """Adapter for OpenAI Chat Completions API and compatible endpoints (e.g., OpenRouter).

    Uses environment variables for configuration by default:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (set this for OpenRouter)
    - OPENAI_ORG (optional)
    """

    def __init__(
        self, model: str, *, base_url: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init__(model, **kwargs)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Install openai>=1.40 and set OPENAI_API_KEY (and OPENAI_BASE_URL for proxies)"
            ) from e

        # Configure client optionally with a custom base URL (e.g., OpenRouter)
        if self.base_url:
            self.client = OpenAI(base_url=self.base_url)
        else:
            self.client = OpenAI()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelReply:
        # Build common kwargs for Chat Completions (non-streaming only)
        base_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools is not None:
            base_kwargs["tools"] = tools
        if tool_choice is not None:
            base_kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            base_kwargs["response_format"] = response_format
        if isinstance(max_tokens, int) and max_tokens > 0:
            base_kwargs["max_tokens"] = int(max_tokens)
        # Some models reject sampling params; include only when explicitly set
        if temperature is not None:
            try:
                base_kwargs["temperature"] = float(temperature)
            except Exception:
                pass
        if top_p is not None:
            try:
                base_kwargs["top_p"] = float(top_p)
            except Exception:
                pass

        # Primary attempt: non-streaming Chat Completions
        try:
            kwargs = dict(base_kwargs)
            kwargs["stream"] = False
            resp = self.client.chat.completions.create(**kwargs)
        except Exception:
            # Compatibility fallback: drop response_format and sampling params
            kwargs = dict(base_kwargs)
            kwargs.pop("response_format", None)
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs["stream"] = False
            resp = self.client.chat.completions.create(**kwargs)

        # Normalize response
        choice = resp.choices[0]
        content = getattr(choice.message, "content", None) or ""
        raw_tool_calls = getattr(choice.message, "tool_calls", None) or []
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in raw_tool_calls
        ]
        usage = {
            "prompt_tokens": int(
                getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0
            ),
            "completion_tokens": int(
                getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0
            ),
            "total_tokens": int(
                getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
            ),
        }
        return ModelReply(content=content, tool_calls=tool_calls, usage=usage)


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude models with tool-use support."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Install anthropic>=0.34 and set ANTHROPIC_API_KEY to use the Anthropic adapter"
            ) from e
        self.client = Anthropic()

    def _convert_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if not tools:
            return []
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            fn = dict(tool.get("function", {}))
            name = str(fn.get("name", "")).strip()
            if not name:
                continue
            converted.append(
                {
                    "name": name,
                    "description": str(fn.get("description", "")),
                    "input_schema": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return converted

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system_prompt, remaining = _split_system_messages(messages)
        converted: List[Dict[str, Any]] = []
        for msg in remaining:
            role = str(msg.get("role", ""))
            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = "\n".join(str(x) for x in content)
                else:
                    text = str(content)
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            }
                        ],
                    }
                )
            elif role == "assistant":
                parts: List[Dict[str, Any]] = []
                content = msg.get("content", "")
                if isinstance(content, list):
                    for piece in content:
                        if isinstance(piece, str) and piece:
                            parts.append({"type": "text", "text": piece})
                elif isinstance(content, str) and content:
                    parts.append({"type": "text", "text": content})
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc.get("function", {}) or {}
                    args = _tool_arguments(fn.get("arguments", ""))
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex}",
                            "name": str(fn.get("name", "")),
                            "input": args,
                        }
                    )
                if not parts:
                    parts.append({"type": "text", "text": ""})
                converted.append({"role": "assistant", "content": parts})
            elif role == "tool":
                tool_call_id = str(msg.get("tool_call_id", ""))
                result_text = msg.get("content", "")
                if isinstance(result_text, list):
                    text_payload = "\n".join(str(x) for x in result_text)
                else:
                    text_payload = str(result_text)
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": text_payload,
                                    }
                                ],
                            }
                        ],
                    }
                )
        return (system_prompt or None), converted

    def _convert_tool_choice(
        self, tool_choice: Optional[str | Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not tool_choice:
            return None
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                return {"type": "none"}
            if tool_choice == "auto":
                return None
            if tool_choice == "required":
                return {"type": "auto"}
            return None
        # OpenAI dict form: {"type":"function","function":{"name":...}}
        choice_type = str(tool_choice.get("type", ""))
        if choice_type == "function":
            fn = tool_choice.get("function", {}) or {}
            name = fn.get("name")
            if name:
                return {"type": "tool", "name": name}
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelReply:
        system_prompt, converted_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools)
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
        }
        if system_prompt:
            request["system"] = system_prompt
        if api_tools:
            request["tools"] = api_tools
        tool_choice_payload = self._convert_tool_choice(tool_choice)
        if tool_choice_payload:
            request["tool_choice"] = tool_choice_payload
        if isinstance(max_tokens, int) and max_tokens > 0:
            request["max_output_tokens"] = int(max_tokens)
        else:
            request.setdefault(
                "max_output_tokens", int(self.kwargs.get("max_output_tokens", 1024))
            )
        if temperature is not None:
            request["temperature"] = float(temperature)
        if top_p is not None:
            request["top_p"] = float(top_p)
        # response_format is not supported directly; rely on system instructions
        resp = self.client.messages.create(**request)
        content_parts = []
        tool_calls: List[Dict[str, Any]] = []
        for block in getattr(resp, "content", []) or []:
            block_type = getattr(block, "type", "")
            if block_type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(
                    {
                        "id": getattr(block, "id", f"call_{uuid.uuid4().hex}"),
                        "type": "function",
                        "function": {
                            "name": getattr(block, "name", ""),
                            "arguments": json.dumps(args),
                        },
                    }
                )
            elif block_type == "text":
                text_value = getattr(block, "text", "")
                if text_value:
                    content_parts.append(text_value)
        usage_info = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_info, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_info, "output_tokens", 0) or 0),
            "total_tokens": int(
                (getattr(usage_info, "input_tokens", 0) or 0)
                + (getattr(usage_info, "output_tokens", 0) or 0)
            ),
        }
        content = "\n".join(content_parts)
        return ModelReply(content=content, tool_calls=tool_calls, usage=usage)


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini models via google-generativeai."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Install google-generativeai>=0.5 and set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini adapter"
            ) from e
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini adapter"
            )
        genai.configure(api_key=api_key)
        self.genai = genai

    def _convert_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Tuple[List[Any], Optional[List[str]]]:
        if not tools:
            return [], None
        try:
            from google.generativeai.types import FunctionDeclaration, Tool  # type: ignore
        except Exception:
            return [], None
        declarations: List[FunctionDeclaration] = []
        names: List[str] = []
        for tool in tools:
            fn = dict(tool.get("function", {}))
            name = str(fn.get("name", "")).strip()
            if not name:
                continue
            params = fn.get("parameters", {"type": "object", "properties": {}})
            declarations.append(
                FunctionDeclaration(
                    name=name,
                    description=str(fn.get("description", "")),
                    parameters=params,
                )
            )
            names.append(name)
        if not declarations:
            return [], None
        tool_obj = [Tool(function_declarations=declarations)]
        return tool_obj, names

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system_prompt, remaining = _split_system_messages(messages)
        history: List[Dict[str, Any]] = []
        for msg in remaining:
            role = str(msg.get("role", ""))
            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = "\n".join(str(x) for x in content)
                else:
                    text = str(content)
                history.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                parts: List[Dict[str, Any]] = []
                content = msg.get("content", "")
                if isinstance(content, list):
                    for piece in content:
                        if isinstance(piece, str) and piece:
                            parts.append({"text": piece})
                elif isinstance(content, str) and content:
                    parts.append({"text": content})
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc.get("function", {}) or {}
                    args = _tool_arguments(fn.get("arguments", ""))
                    parts.append(
                        {
                            "functionCall": {
                                "name": str(fn.get("name", "")),
                                "args": args,
                            }
                        }
                    )
                if not parts:
                    parts.append({"text": ""})
                history.append({"role": "model", "parts": parts})
            elif role == "tool":
                name = str(msg.get("name", ""))
                raw_content = msg.get("content", "")
                payload = None
                if isinstance(raw_content, str) and raw_content:
                    try:
                        payload = json.loads(raw_content)
                    except Exception:
                        payload = {"text": raw_content}
                elif isinstance(raw_content, dict):
                    payload = raw_content
                else:
                    payload = {"text": str(raw_content)}
                history.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": name,
                                    "response": payload,
                                }
                            }
                        ],
                    }
                )
        return (system_prompt or None), history

    def _convert_tool_choice(
        self, tool_choice: Optional[str | Dict[str, Any]], allowed: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        if not tool_choice:
            return None
        config: Dict[str, Any] = {"function_calling_config": {}}
        fcc = config["function_calling_config"]
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                fcc["mode"] = "NONE"
            elif tool_choice in ("auto", "required"):
                fcc["mode"] = "ANY"
            return config
        if isinstance(tool_choice, dict):
            name = tool_choice.get("name")
            if not name and isinstance(tool_choice.get("function"), dict):
                name = tool_choice["function"].get("name")
            if name:
                fcc["mode"] = "ANY"
                fcc["allowed_function_names"] = [name]
                return config
        if allowed:
            fcc["mode"] = "ANY"
            fcc["allowed_function_names"] = allowed
            return config
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelReply:
        system_prompt, history = self._convert_messages(messages)
        tool_objects, allowed_names = self._convert_tools(tools)
        generation_config: Dict[str, Any] = {}
        if isinstance(max_tokens, int) and max_tokens > 0:
            generation_config["max_output_tokens"] = int(max_tokens)
        if temperature is not None:
            generation_config["temperature"] = float(temperature)
        if top_p is not None:
            generation_config["top_p"] = float(top_p)
        model = self.genai.GenerativeModel(
            model_name=self.model,
            tools=tool_objects or None,
            system_instruction=system_prompt,
        )
        tool_config = self._convert_tool_choice(tool_choice, allowed_names)
        response = model.generate_content(
            history,
            generation_config=generation_config or None,
            tool_config=tool_config,
            stream=False,
        )
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for candidate in getattr(response, "candidates", []) or []:
            parts = getattr(candidate, "content", None)
            if not parts:
                continue
            for part in getattr(parts, "parts", []) or []:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = fc.args if isinstance(fc.args, dict) else {}
                    tool_calls.append(
                        {
                            "id": f"call_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": getattr(fc, "name", ""),
                                "arguments": json.dumps(args),
                            },
                        }
                    )
                elif hasattr(part, "text") and part.text:
                    content_parts.append(part.text)
                elif isinstance(part, dict):
                    # When using REST responses (e.g., dicts), handle accordingly
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        args = (
                            fc.get("args") if isinstance(fc.get("args"), dict) else {}
                        )
                        tool_calls.append(
                            {
                                "id": f"call_{uuid.uuid4().hex}",
                                "type": "function",
                                "function": {
                                    "name": fc.get("name", ""),
                                    "arguments": json.dumps(args),
                                },
                            }
                        )
                    elif "text" in part and part["text"]:
                        content_parts.append(str(part["text"]))
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
        total_tokens = int(
            getattr(usage_meta, "total_token_count", prompt_tokens + completion_tokens)
            or 0
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        content = "\n".join(content_parts)
        return ModelReply(content=content, tool_calls=tool_calls, usage=usage)


def get_adapter(model_spec: str, **kwargs: Any) -> Tuple[BaseAdapter, str, str]:
    """Return an adapter and decompose model spec into (provider, model_id).

    Supported specs:
    - openai:<model_id>
    - openrouter:<route_id>  (requires OPENAI_BASE_URL, OPENAI_API_KEY)
    - anthropic:<model_id>
    - gemini:<model_id>
    """
    if ":" not in model_spec:
        raise ValueError("model spec must be of form provider:model")
    provider, model_id = model_spec.split(":", 1)
    provider = provider.strip().lower()
    model_id = model_id.strip()

    if provider in ("openai", "openrouter"):
        base_url = kwargs.pop("base_url", None)
        adapter = OpenAICompatAdapter(model_id, base_url=base_url, **kwargs)
        return adapter, provider, model_id

    if provider == "anthropic":
        adapter = AnthropicAdapter(model_id, **kwargs)
        return adapter, provider, model_id

    if provider == "gemini":
        adapter = GeminiAdapter(model_id, **kwargs)
        return adapter, provider, model_id

    raise ValueError(f"unknown provider '{provider}' in model spec '{model_spec}'")
