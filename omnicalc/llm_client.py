"""LLM client for LM Studio's OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "medgemma"  # Will use whatever is loaded in LM Studio


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
    ) -> CompletionResult:
        """Call LM Studio /chat/completions."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if tools:
            payload["tools"] = tools

        logger.debug(f"Chat completion request: {json.dumps(payload, indent=2)}")

        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        logger.debug(f"Chat completion response: {json.dumps(data, indent=2)}")

        result = self._parse_response_payload(data)

        # Fallback: try to parse tool calls from content if tool_calls is empty
        # Some models embed tool calls in thinking tokens or malformed blocks
        if not result.tool_calls and result.content:
            parsed = self._parse_tool_calls_from_content(result.content)
            if parsed:
                result.tool_calls = parsed
                logger.info(f"Parsed {len(parsed)} tool call(s) from content")

        return result

    def _parse_tool_calls_from_content(self, content: str) -> List[ToolCall]:
        """
        Fallback parser for tool calls embedded in content.

        Handles cases where model generates tool calls inside thinking tokens
        or with malformed markers. Looks for JSON objects with "name" and
        "arguments" fields.
        """
        import re

        tool_calls = []

        # Pattern 1: Standard [TOOL_REQUEST]...[END_TOOL_REQUEST] (may be inside thinking tokens)
        tool_request_pattern = r'\[TOOL_REQUEST\]\s*(\{.*?\})\s*\[END_TOOL_REQUEST\]'
        matches = re.findall(tool_request_pattern, content, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                if "name" in data:
                    tool_calls.append(ToolCall(
                        id=f"parsed_{len(tool_calls)}",
                        name=data["name"],
                        arguments=data.get("arguments", {}),
                        raw_arguments=json.dumps(data.get("arguments", {})),
                    ))
            except json.JSONDecodeError:
                continue

        if tool_calls:
            return tool_calls

        # Pattern 2: Look for JSON with "calc_id" and tool-like structure (execute_calc)
        # This handles malformed output where model skips the markers
        calc_pattern = r'\{\s*"calc_id"\s*:\s*"([^"]+)"\s*,\s*"variables"\s*:\s*(\{[^}]*\})'
        calc_matches = re.findall(calc_pattern, content, re.DOTALL)

        for calc_id, variables_str in calc_matches:
            try:
                # Try to find the full variables object
                # Find from "variables": to the matching closing brace
                var_start = content.find(f'"calc_id": "{calc_id}"')
                if var_start == -1:
                    var_start = content.find(f'"calc_id":"{calc_id}"')
                if var_start != -1:
                    # Find the enclosing JSON object
                    brace_count = 0
                    start_idx = None
                    for i in range(var_start, -1, -1):
                        if content[i] == '{':
                            if start_idx is None:
                                start_idx = i
                            brace_count += 1
                            break

                    if start_idx is not None:
                        # Find matching close brace
                        brace_count = 0
                        for i in range(start_idx, len(content)):
                            if content[i] == '{':
                                brace_count += 1
                            elif content[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = content[start_idx:i+1]
                                    try:
                                        data = json.loads(json_str)
                                        tool_calls.append(ToolCall(
                                            id=f"parsed_exec_{len(tool_calls)}",
                                            name="execute_calc",
                                            arguments=data,
                                            raw_arguments=json_str,
                                        ))
                                    except json.JSONDecodeError:
                                        pass
                                    break
            except Exception:
                continue

        return tool_calls

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
        )

    def _parse_chat_completions_payload(self, data: Dict[str, Any]) -> CompletionResult:
        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        result = CompletionResult(
            content=message.get("content"),
            finish_reason=finish_reason,
            usage=data.get("usage"),
        )
        result.tool_calls = self._parse_tool_calls_from_message(message)
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
            tool_calls=tool_calls,
            finish_reason=data.get("status", "stop"),
            usage=data.get("usage"),
        )

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
