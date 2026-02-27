"""LLM client for LM Studio's OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "sigjhl/medgemma-1.5-4b-it-MedCalcCaller"


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            formatted_calls = []
            for tc in self.tool_calls:
                if isinstance(tc, dict):
                    formatted_calls.append(tc)
                else:
                    formatted_calls.append({
                        "id": getattr(tc, "id", ""),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(tc, "name", ""),
                            "arguments": getattr(tc, "raw_arguments", "") or json.dumps(getattr(tc, "arguments", {}))
                        }
                    })
            d["tool_calls"] = formatted_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class ToolCall:
    """A parsed tool call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""


@dataclass
class CompletionResult:
    """Result from a chat completion."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None
    response_id: Optional[str] = None


class LMStudioClient:
    """
    Client for LM Studio's OpenAI-compatible API.

    Supports:
    - Chat completions
    - Tool use (function calling)
    - Streaming
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def chat_completion(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> CompletionResult:
        """Call LM Studio /chat/completions."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = False
        if tool_choice:
            payload["tool_choice"] = tool_choice

        logger.debug(f"Chat completion request: {json.dumps(payload, indent=2)}")

        attempts: List[Dict[str, Any]] = [dict(payload)]
        if "parallel_tool_calls" in payload:
            p = dict(payload)
            p.pop("parallel_tool_calls", None)
            attempts.append(p)
        if "tool_choice" in payload:
            p = dict(payload)
            p.pop("tool_choice", None)
            attempts.append(p)
        if "parallel_tool_calls" in payload and "tool_choice" in payload:
            p = dict(payload)
            p.pop("parallel_tool_calls", None)
            p.pop("tool_choice", None)
            attempts.append(p)

        unique_attempts: List[Dict[str, Any]] = []
        seen_payloads: set[str] = set()
        for attempt in attempts:
            key = json.dumps(attempt, sort_keys=True, default=str)
            if key in seen_payloads:
                continue
            seen_payloads.add(key)
            unique_attempts.append(attempt)

        response = None
        last_error: Optional[httpx.HTTPStatusError] = None
        for index, attempt_payload in enumerate(unique_attempts):
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=attempt_payload,
            )
            try:
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                is_last = index == len(unique_attempts) - 1
                if response.status_code in (400, 422) and not is_last:
                    snippet = (response.text or "").strip().replace("\n", " ")
                    if len(snippet) > 240:
                        snippet = snippet[:240] + "..."
                    logger.warning("LM Studio rejected chat payload (%s); retrying simpler payload", snippet or response.status_code)
                    continue
                raise

        if response is None:
            if last_error:
                raise last_error
            raise RuntimeError("No LM Studio response received")
        data = response.json()

        logger.debug(f"Chat completion response: {json.dumps(data, indent=2)}")

        result = self._parse_response_payload(data)

        return result

    async def responses_completion(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> CompletionResult:
        """Call /responses using chat-style history mapped to responses input items."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": self._to_responses_input(messages),
        }

        instructions = self._extract_system_prompt(messages)
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = self._to_responses_tools(tools)
            payload["parallel_tool_calls"] = False

        logger.debug("Responses request: %s", json.dumps(payload, indent=2))

        attempts: List[Dict[str, Any]] = [dict(payload)]
        if "parallel_tool_calls" in payload:
            p = dict(payload)
            p.pop("parallel_tool_calls", None)
            attempts.append(p)

        unique_attempts: List[Dict[str, Any]] = []
        seen_payloads: set[str] = set()
        for attempt in attempts:
            key = json.dumps(attempt, sort_keys=True, default=str)
            if key in seen_payloads:
                continue
            seen_payloads.add(key)
            unique_attempts.append(attempt)

        response = None
        last_error: Optional[httpx.HTTPStatusError] = None
        for index, attempt_payload in enumerate(unique_attempts):
            response = await self._client.post(
                f"{self.base_url}/responses",
                json=attempt_payload,
            )
            try:
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                is_last = index == len(unique_attempts) - 1
                if response.status_code in (400, 404, 422) and not is_last:
                    snippet = (response.text or "").strip().replace("\n", " ")
                    if len(snippet) > 240:
                        snippet = snippet[:240] + "..."
                    logger.warning("Responses payload rejected (%s); retrying simpler payload", snippet or response.status_code)
                    continue
                raise

        if response is None:
            if last_error:
                raise last_error
            raise RuntimeError("No /responses response received")

        data = response.json()
        logger.debug("Responses response: %s", json.dumps(data, indent=2))
        result = self._parse_response_payload(data)

        return result

    def _extract_system_prompt(self, messages: List[Message]) -> Optional[str]:
        for message in messages:
            if message.role == "system" and isinstance(message.content, str):
                return message.content
        return None

    def _to_responses_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function", {})
            mapped.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return mapped

    def _to_responses_input(self, messages: List[Message]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue

            if message.role in ("user", "assistant"):
                content = self._to_responses_content(message.content)
                if content:
                    items.append({
                        "role": message.role,
                        "content": content,
                    })

                for tool_call in message.tool_calls or []:
                    fn = tool_call.get("function", {})
                    call_id = tool_call.get("id")
                    if not call_id:
                        continue
                    raw_args = fn.get("arguments")
                    if isinstance(raw_args, dict):
                        raw_args = json.dumps(raw_args)
                    items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": raw_args or "{}",
                    })
                continue

            if message.role == "tool" and message.tool_call_id:
                output = message.content
                if isinstance(output, (dict, list)):
                    output = json.dumps(output)
                elif output is None:
                    output = ""
                items.append({
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": str(output),
                })

        return items

    def _to_responses_content(self, content: Any) -> List[Dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for part in content:
                if isinstance(part, str):
                    parts.append({"type": "input_text", "text": part})
                    continue
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("content")
                    if text:
                        parts.append({"type": "input_text", "text": str(text)})
                elif part_type == "image":
                    data_url = part.get("data_url")
                    if data_url:
                        parts.append({"type": "input_image", "image_url": str(data_url)})
            return parts

        return [{"type": "input_text", "text": json.dumps(content, default=str)}]

    def _parse_response_payload(self, data: Dict[str, Any]) -> CompletionResult:
        if "choices" in data:
            return self._parse_chat_completions_payload(data)
        if "output" in data:
            return self._parse_responses_payload(data)

        # Unknown payload shape; best-effort extraction
        content = data.get("output_text") or data.get("content")
        return CompletionResult(
            content=content,
            finish_reason=data.get("finish_reason", "stop"),
            usage=data.get("usage"),
            response_id=data.get("id"),
        )

    def _parse_chat_completions_payload(self, data: Dict[str, Any]) -> CompletionResult:
        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        result = CompletionResult(
            content=message.get("content"),
            finish_reason=finish_reason,
            usage=data.get("usage"),
            response_id=data.get("id"),
        )
        parsed_calls = self._parse_tool_calls_from_message(message)
        result.tool_calls = self._normalize_tool_calls(parsed_calls)
        return result

    def _parse_responses_payload(self, data: Dict[str, Any]) -> CompletionResult:
        content_chunks: List[str] = []
        tool_calls: List[ToolCall] = []

        for item in data.get("output", []) or []:
            item_type = item.get("type")
            if item_type == "message":
                content = self._extract_text_from_content(item.get("content"))
                if content:
                    content_chunks.append(content)
                tool_calls.extend(self._parse_tool_calls_from_message(item))
            elif item_type in ("tool_call", "function_call"):
                tool_call = self._parse_tool_call_item(item)
                if tool_call:
                    tool_calls.append(tool_call)
            elif item_type in ("output_text", "text"):
                text = item.get("text") or item.get("content")
                if text:
                    content_chunks.append(text)

        content = "\n".join([c for c in content_chunks if c]).strip()
        if not content:
            content = data.get("output_text")

        return CompletionResult(
            content=content,
            tool_calls=self._normalize_tool_calls(tool_calls),
            finish_reason=data.get("status", "stop"),
            usage=data.get("usage"),
            response_id=data.get("id"),
        )

    def _normalize_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """
        De-duplicate pathological tool-call bursts from small models.

        Current runtime is single-calculator per request, so keep at most one
        execute_calc call per completion payload.
        """
        if not tool_calls:
            return []

        normalized: List[ToolCall] = []
        seen_calc_info: set[str] = set()
        seen_execute_signature: set[str] = set()
        kept_execute = False

        for call in tool_calls:
            if call.name == "calc_info":
                calc_id = str(call.arguments.get("calc_id", ""))
                if calc_id and calc_id in seen_calc_info:
                    continue
                if calc_id:
                    seen_calc_info.add(calc_id)
                normalized.append(call)
                continue

            if call.name == "execute_calc":
                if kept_execute:
                    continue
                calc_id = call.arguments.get("calc_id")
                variables = call.arguments.get("variables", {})
                try:
                    signature = f"{calc_id}|{json.dumps(variables, sort_keys=True, default=str)}"
                except Exception:
                    signature = f"{calc_id}|{str(variables)}"
                if signature in seen_execute_signature:
                    continue
                seen_execute_signature.add(signature)
                normalized.append(call)
                kept_execute = True
                continue

            normalized.append(call)

        if len(normalized) != len(tool_calls):
            logger.warning(
                "Normalized tool calls from %s to %s",
                len(tool_calls),
                len(normalized),
            )
        return normalized

    def _extract_text_from_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    if part.get("type") in ("output_text", "text"):
                        text = part.get("text") or part.get("content")
                        if text:
                            parts.append(text)
            return "".join(parts)
        return ""

    def _parse_tool_calls_from_message(self, message: Dict[str, Any]) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        for tc in message.get("tool_calls", []) or []:
            func = tc.get("function") or {}
            name = func.get("name") or tc.get("name")
            args_value = func.get("arguments") if func else tc.get("arguments")
            args, raw = self._parse_tool_arguments(args_value)
            tool_calls.append(ToolCall(
                id=tc.get("id") or f"call_{len(tool_calls)}",
                name=name or "unknown",
                arguments=args,
                raw_arguments=raw,
            ))
        return tool_calls

    def _parse_tool_call_item(self, item: Dict[str, Any]) -> Optional[ToolCall]:
        func = item.get("function") or {}
        name = item.get("name") or item.get("tool_name") or func.get("name")
        args_value = item.get("arguments") or func.get("arguments")
        args, raw = self._parse_tool_arguments(args_value)
        if not name:
            return None
        return ToolCall(
            id=item.get("id") or item.get("call_id") or f"call_{name}",
            name=name,
            arguments=args,
            raw_arguments=raw,
        )

    def _parse_tool_arguments(self, args_value: Any) -> tuple[Dict[str, Any], str]:
        if isinstance(args_value, dict):
            return args_value, json.dumps(args_value)
        if isinstance(args_value, str):
            if not args_value:
                return {}, ""
            try:
                return json.loads(args_value), args_value
            except json.JSONDecodeError:
                return {}, args_value
        return {}, ""

    async def chat_completion_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Make a streaming chat completion request.

        Yields chunks with delta content or tool_calls.
        """
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if not data_str:
                        continue
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    yield payload

    async def health_check(self) -> bool:
        """Check if LM Studio server is available."""
        try:
            response = await self._client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception:
            return False

    async def get_loaded_model(self) -> Optional[str]:
        """Get the currently loaded model name."""
        try:
            response = await self._client.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    return data["data"][0]["id"]
        except Exception:
            pass
        return None

    async def list_models(self) -> List[str]:
        """Return list of available model ids from LM Studio."""
        try:
            response = await self._client.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                return [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
        return []

    async def chat_v1_stream(
        self,
        user_input: Any,
        system_prompt: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        integrations: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Make a streaming chat request using LM Studio's proprietary /api/v1/chat endpoint.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": user_input,
            "stream": True,
        }
        
        if system_prompt:
            payload["system_prompt"] = system_prompt
            
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
            
        if integrations:
            payload["integrations"] = integrations
            
        api_url = self.base_url.replace("/v1", "/api/v1/chat")
        
        async with self._client.stream(
            "POST",
            api_url,
            json=payload,
            timeout=120.0
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                await response.aread()
                import logging
                logging.getLogger(__name__).error(f"LM Studio API Error: {response.text}")
                raise e
                
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if not data_str:
                        continue
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
