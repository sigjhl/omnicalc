"""
Orchestrator agent for AgentiCalc.

Simple multi-turn tool calling following the LM Studio pattern:
1. User provides clinical data
2. LLM calls calc_info to get the schema
3. LLM calls execute_calc with extracted variables
4. Results returned to user
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from .llm_client import LMStudioClient, Message, ToolCall
from .models import (
    EventType,
    ExecuteCalcResult,
    ExtractedVariable,
    InputAttachment,
    OrchestratorRequest,
    OrchestratorResponse,
    StreamEvent,
    UserLocale,
)
from .prompts import build_system_prompt
from .tools import TOOL_DEFINITIONS, ToolHandler

logger = logging.getLogger(__name__)


def _clean_assistant_chunk(text: str) -> str:
    """Remove control tags emitted by some LM Studio model variants."""
    if not text:
        return ""
    text = text.replace("<|channel|>final<|message|>", "")
    text = text.replace("<|channel|>thought<|message|>", "")
    return text


@dataclass
class Session:
    """Conversation session state."""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    locale: UserLocale = field(default_factory=UserLocale)
    last_calculator_id: Optional[str] = None
    last_variables: Dict[str, Any] = field(default_factory=dict)
    previous_response_id: Optional[str] = None


class OrchestratorAgent:
    """
    Main orchestrator agent that handles the conversation loop.

    Uses standard multi-turn tool calling with LM Studio.
    """

    def __init__(self, llm_client: LMStudioClient, tool_handler: ToolHandler, max_turns: int = 10):
        self.llm = llm_client
        self.tools = tool_handler
        self.max_turns = max_turns
        self.sessions: Dict[str, Session] = {}
        # Used to pass data from MCP endpoint to stream events
        self.last_mcp_calc_id: Optional[str] = None
        self.last_mcp_result: Optional[Any] = None
        self.last_mcp_exec_signature: Optional[str] = None

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        locale: Optional[UserLocale] = None,
    ) -> Session:
        """Get existing session or create a new one."""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if locale and session.locale != locale:
                session.locale = locale
            return session

        new_session = Session(session_id=session_id or str(uuid.uuid4()), locale=locale or UserLocale())
        if session_id:
            self.sessions[session_id] = new_session
        self.tools.reset_session()  # Reset tool state for new session
        return new_session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str):
        """Clear a session and its state."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    async def process(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """
        Process user input through the orchestrator.

        Follows the standard multi-turn tool calling pattern:
        1. Send user message with tools
        2. If tool_calls, execute them and add results to messages
        3. Repeat until no more tool calls or max turns reached
        """
        if request.model:
            logger.info("Using model override: %s", request.model)
            self.llm.model = request.model

        session = self.get_or_create_session(request.session_id, request.locale)
        api_mode = request.api_mode or "chat_completions"

        # Build system prompt with available calculators
        calculators = await self.tools.list_calculators()
        if request.allowed_calculators:
            calculators = [c for c in calculators if c.get("id") in request.allowed_calculators]
        system_prompt = build_system_prompt(
            calculators=calculators,
            locale_description=session.locale.description,
        )

        # Initialize messages if new session
        if not session.messages:
            session.messages.append(Message(role="system", content=system_prompt))

        # Add user message
        user_content = self._build_user_content(request.input, request.calculator_hint, request.attachments)
        session.messages.append(Message(role="user", content=user_content))

        last_exec_result: Optional[ExecuteCalcResult] = None

        # Agent loop
        for turn in range(self.max_turns):
            logger.debug(f"Agent turn {turn + 1}/{self.max_turns}")

            # Call LLM with tools
            if api_mode == "responses":
                result = await self.llm.responses_completion(
                    messages=session.messages,
                    tools=TOOL_DEFINITIONS,
                )
                if result.response_id:
                    session.previous_response_id = result.response_id
            else:
                result = await self.llm.chat_completion(
                    messages=session.messages,
                    tools=TOOL_DEFINITIONS,
                )

            # No tool calls = final response
            if not result.tool_calls:
                if result.content:
                    session.messages.append(Message(role="assistant", content=result.content))

                # Return with any calculation results we have
                variables = self._variables_from_dict(session.last_variables)
                return OrchestratorResponse(
                    success=last_exec_result.success if last_exec_result else True,
                    calculator_id=session.last_calculator_id,
                    variables=variables,
                    result=last_exec_result,
                    assistant_message=result.content,
                    errors=last_exec_result.error_messages() if last_exec_result and not last_exec_result.success else [],
                )

            # Process tool calls
            for tool_call in result.tool_calls:
                tool_result = await self.tools.execute_tool(tool_call.name, tool_call.arguments)

                # Add assistant message with tool call
                session.messages.append(
                    Message(
                        role="assistant",
                        tool_calls=[{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.raw_arguments,
                            },
                        }],
                    )
                )

                # Add tool result message
                session.messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(tool_result),
                        tool_call_id=tool_call.id,
                    )
                )

                # Track execute_calc results
                if tool_call.name == "execute_calc":
                    last_exec_result = ExecuteCalcResult(**tool_result)
                    session.last_calculator_id = tool_call.arguments.get("calc_id")
                    session.last_variables = tool_call.arguments.get("variables", {})

        # Max turns reached
        variables = self._variables_from_dict(session.last_variables)
        return OrchestratorResponse(
            success=False,
            calculator_id=session.last_calculator_id,
            variables=variables,
            result=last_exec_result,
            errors=["Maximum conversation turns reached"],
        )

    async def process_stream(self, request: OrchestratorRequest) -> AsyncGenerator[StreamEvent, None]:
        """Process user input using LM Studio's native agent loop with FastMCP via /api/v1/chat."""
        if request.api_mode == "responses":
            if request.model:
                logger.info("Using model override (stream): %s", request.model)
                self.llm.model = request.model

            session = self.get_or_create_session(request.session_id, request.locale)

            calculators = await self.tools.list_calculators()
            if request.allowed_calculators:
                calculators = [c for c in calculators if c.get("id") in request.allowed_calculators]
            system_prompt = build_system_prompt(
                calculators=calculators,
                locale_description=session.locale.description,
            )
            if not session.messages:
                session.messages.append(Message(role="system", content=system_prompt))

            user_content = self._build_user_content(request.input, request.calculator_hint, request.attachments)
            session.messages.append(Message(role="user", content=user_content))

            saw_success = False
            for _ in range(self.max_turns):
                result = await self.llm.responses_completion(
                    messages=session.messages,
                    tools=TOOL_DEFINITIONS,
                )
                if result.response_id:
                    session.previous_response_id = result.response_id

                if not result.tool_calls:
                    if result.content:
                        session.messages.append(Message(role="assistant", content=result.content))
                        yield StreamEvent(type=EventType.ASSISTANT_MESSAGE, data={"content": result.content})
                    return

                for tool_call in result.tool_calls:
                    if tool_call.name == "calc_info":
                        yield StreamEvent(
                            type=EventType.CALCULATOR_SELECTED,
                            data={"calc_id": tool_call.arguments.get("calc_id") or "calculator schema"},
                        )
                        await asyncio.sleep(0.2)

                    tool_result = await self.tools.execute_tool(tool_call.name, tool_call.arguments)

                    session.messages.append(
                        Message(
                            role="assistant",
                            tool_calls=[{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": tool_call.raw_arguments,
                                },
                            }],
                        )
                    )
                    session.messages.append(
                        Message(
                            role="tool",
                            content=json.dumps(tool_result),
                            tool_call_id=tool_call.id,
                        )
                    )

                    if tool_call.name != "execute_calc":
                        continue

                    calc_id = tool_call.arguments.get("calc_id")
                    variables = self._augment_variables_for_ui(calc_id, tool_call.arguments.get("variables", {}))

                    try:
                        exec_result = ExecuteCalcResult(**tool_result)
                    except Exception:
                        exec_result = None

                    if exec_result and exec_result.success:
                        yield StreamEvent(
                            type=EventType.EXTRACTING_VARIABLES,
                            data={
                                "calc_id": calc_id,
                                "variables": variables,
                            },
                        )
                        await asyncio.sleep(0.2)
                        yield StreamEvent(
                            type=EventType.CALCULATION_COMPLETE,
                            data={
                                "calc_id": calc_id,
                                "outputs": exec_result.outputs,
                                "audit_trace": exec_result.audit_trace,
                            },
                        )
                        saw_success = True
                    else:
                        errors = tool_result.get("errors", []) if isinstance(tool_result, dict) else []
                        if isinstance(errors, str):
                            errors = [errors]
                        if not isinstance(errors, list):
                            errors = [str(errors)]
                        if not errors:
                            errors = ["Calculation failed"]
                        yield StreamEvent(type=EventType.VALIDATION_ERROR, data={"errors": errors})

            if not saw_success:
                yield StreamEvent(type=EventType.ERROR, data={"error": "Maximum conversation turns reached"})
            return

        if request.model:
            logger.info("Using model override (stream): %s", request.model)
            self.llm.model = request.model

        # These are currently shared on the orchestrator instance (not per-session).
        # Clear them at the start of each stream to prevent stale result leakage.
        self.last_mcp_calc_id = None
        self.last_mcp_result = None
        self.last_mcp_exec_signature = None

        session = self.get_or_create_session(request.session_id, request.locale)

        # Build system prompt
        calculators = await self.tools.list_calculators()
        if request.allowed_calculators:
            calculators = [c for c in calculators if c.get("id") in request.allowed_calculators]
        system_prompt = build_system_prompt(
            calculators=calculators,
            locale_description=session.locale.description,
        )

        user_content = self._build_user_content(request.input, request.calculator_hint, request.attachments)
        
        # We don't append to session.messages because LM Studio maintains history via previous_response_id
        
        integrations = [
            {
                "type": "ephemeral_mcp",
                "server_label": "omnicalc_tools",
                "server_url": getattr(request, "mcp_url", None) or "http://localhost:8002/mcp/sse",
                "allowed_tools": ["calc_info", "execute_calc"]
            }
        ]

        full_content = ""
        tool_call_count = 0
        consecutive_errors = 0
        current_message_chunks: List[str] = []
        last_complete_message = ""
        saw_tool_activity = False
        successful_execution_emitted = False
        
        async for event in self.llm.chat_v1_stream(
            user_input=user_content,
            system_prompt=system_prompt,
            previous_response_id=session.previous_response_id,
            integrations=integrations,
        ):
            event_type = event.get("type")
            
            if event_type == "response_id":
                session.previous_response_id = event.get("response_id")
                
            elif event_type == "message.start":
                current_message_chunks = []

            elif event_type == "message.delta":
                delta_content = _clean_assistant_chunk(event.get("content", ""))
                full_content += delta_content
                if delta_content:
                    current_message_chunks.append(delta_content)

            elif event_type == "message.end":
                candidate = "".join(current_message_chunks).strip()
                if candidate:
                    last_complete_message = candidate
                
            elif event_type == "tool_call.name":
                saw_tool_activity = True
                current_message_chunks = []
                last_complete_message = ""
                tool_name = event.get("tool_name", "")
                if tool_name == "calc_info":
                    yield StreamEvent(type=EventType.CALCULATOR_SELECTED, data={"calc_id": "calculator schema"})
                    await asyncio.sleep(0.5)
                    
            elif event_type == "tool_call.arguments":
                saw_tool_activity = True
                current_message_chunks = []
                last_complete_message = ""
                # LM Studio emits full arguments as a dict once parsed
                args = event.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                        
                tool_call_count += 1
                if tool_call_count >= 5:
                    if successful_execution_emitted:
                        break
                    yield StreamEvent(type=EventType.ERROR, data={"error": "Failed: Maximum of 5 tool calls reached without final user response."})
                    break
                    
            elif event_type == "tool_call.success" or event_type == "tool_call.error":
                saw_tool_activity = True
                if event_type == "tool_call.error":
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
                    
                if consecutive_errors >= 5:
                    yield StreamEvent(type=EventType.ERROR, data={"error": "Failed: 5 consecutive tool call failures."})
                    break

                # Once the MCP tool finishes executing, flush the internal state to the UI
                event_tool = event.get("tool", "")
                if self.last_mcp_result is not None and event_tool == "execute_calc":
                    res = self.last_mcp_result
                    args = event.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    if hasattr(res, "success") and res.success:
                        # Show "Executing" only when backend accepted execute_calc.
                        if event_tool == "execute_calc":
                            calc_id = args.get("calc_id") or self.last_mcp_calc_id
                            variables = self._augment_variables_for_ui(calc_id, args.get("variables", {}))
                            yield StreamEvent(
                                type=EventType.EXTRACTING_VARIABLES,
                                data={
                                    "calc_id": calc_id,
                                    "variables": variables,
                                }
                            )
                            await asyncio.sleep(0.5)

                        yield StreamEvent(
                            type=EventType.CALCULATION_COMPLETE,
                            data={
                                "calc_id": self.last_mcp_calc_id,
                                "outputs": res.outputs,
                                "audit_trace": res.audit_trace,
                            },
                        )
                        successful_execution_emitted = True
                        self.last_mcp_result = None
                        self.last_mcp_calc_id = None
                        break
                    elif hasattr(res, "errors"):
                        yield StreamEvent(
                            type=EventType.VALIDATION_ERROR,
                            data={"errors": res.errors},
                        )
                    self.last_mcp_result = None
                    self.last_mcp_calc_id = None

        final_message = last_complete_message
        if not final_message and not saw_tool_activity:
            final_message = _clean_assistant_chunk(full_content).strip()
        if final_message:
            yield StreamEvent(type=EventType.ASSISTANT_MESSAGE, data={"content": final_message})

        return

    @staticmethod
    def _variables_from_dict(raw: Dict[str, Any]) -> List[ExtractedVariable]:
        """Convert raw variables dict to ExtractedVariable list."""
        variables: List[ExtractedVariable] = []
        for key, value in (raw or {}).items():
            if value is None:
                continue
            if isinstance(value, dict):
                variables.append(ExtractedVariable(
                    key=key,
                    value=value.get("value"),
                    unit=value.get("unit"),
                ))
            else:
                variables.append(ExtractedVariable(key=key, value=value))
        return variables

    @staticmethod
    def _augment_variables_for_ui(calc_id: Optional[str], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Attach labels/units for frontend variable cards when calculator schema is known."""
        try:
            from omnicalc.calculators import CALCULATORS
            if calc_id in CALCULATORS:
                calc_def = CALCULATORS[calc_id]["def"]
                unit_map = {inp.id: inp.canonical_unit for inp in calc_def.inputs}
                label_map = {inp.id: inp.label for inp in calc_def.inputs}
                augmented_vars: Dict[str, Any] = {}
                for k, v in (variables or {}).items():
                    if isinstance(v, dict):
                        v_copy = dict(v)
                        v_copy["label"] = label_map.get(k, k)
                        augmented_vars[k] = v_copy
                    else:
                        augmented_vars[k] = {"value": v, "unit": unit_map.get(k, ""), "label": label_map.get(k, k)}
                return augmented_vars
        except Exception:
            pass
        return variables or {}

    @staticmethod
    def _attachment_to_content(attachment: InputAttachment) -> Optional[Dict[str, Any]]:
        if attachment.kind == "image":
            mime_type = attachment.mime_type or "image/jpeg"
            data_url = f"data:{mime_type};base64,{attachment.data}"
            return {"type": "image", "data_url": data_url}
        if attachment.kind == "text":
            try:
                decoded = base64.b64decode(attachment.data).decode("utf-8", errors="replace").strip()
            except Exception:
                return None
            if not decoded:
                return None
            if attachment.name:
                decoded = f"[Attachment: {attachment.name}]\n{decoded}"
            return {"type": "text", "content": decoded}
        return None

    def _build_user_content(
        self,
        user_input: str,
        calculator_hint: Optional[str],
        attachments: Optional[List[InputAttachment]],
    ) -> Any:
        text = user_input or ""
        if calculator_hint:
            text = f"[Calculator hint: {calculator_hint}]\n\n{text}" if text else f"[Calculator hint: {calculator_hint}]"

        if not attachments:
            return text

        text_segments: List[str] = [text] if text else []
        media_parts: List[Dict[str, Any]] = []

        for attachment in attachments:
            part = self._attachment_to_content(attachment)
            if part:
                if part.get("type") == "text":
                    content = part.get("content")
                    if content:
                        text_segments.append(str(content))
                else:
                    media_parts.append(part)

        merged_text = "\n\n".join(segment for segment in text_segments if segment).strip()
        if not media_parts:
            return merged_text

        parts: List[Dict[str, Any]] = []
        if merged_text:
            parts.append({"type": "text", "content": merged_text})
        parts.extend(media_parts)

        if not parts:
            return text
        return parts


async def create_orchestrator(
    lm_studio_url: str = "http://localhost:1234/v1",
    model: Optional[str] = "auto",
) -> OrchestratorAgent:
    """Factory function to create an OrchestratorAgent with connected services."""
    llm_client = LMStudioClient(base_url=lm_studio_url, model=model or "auto")
    tool_handler = ToolHandler()

    # Verify connections
    if not await llm_client.health_check():
        logger.warning("LM Studio server not available at %s", lm_studio_url)

    # Choose model automatically if requested
    requested_model = model or "auto"
    available_models = await llm_client.list_models()
    if requested_model == "auto":
        if available_models:
            llm_client.model = available_models[0]
            logger.info("Auto-selected model: %s", llm_client.model)
    else:
        if available_models and requested_model not in available_models:
            fallback = available_models[0]
            logger.warning(
                "Requested model '%s' not available. Falling back to '%s'.",
                requested_model,
                fallback,
            )
            llm_client.model = fallback
        else:
            llm_client.model = requested_model

    calculators = await tool_handler.list_calculators()
    if not calculators:
        logger.warning("No calculators available from CalcSpec at %s", calcspec_url)

    return OrchestratorAgent(
        llm_client=llm_client,
        tool_handler=tool_handler,
    )
