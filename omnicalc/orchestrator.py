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


@dataclass
class Session:
    """Conversation session state."""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    locale: UserLocale = field(default_factory=UserLocale)
    last_calculator_id: Optional[str] = None
    last_variables: Dict[str, Any] = field(default_factory=dict)


class OrchestratorAgent:
    """
    Main orchestrator agent that handles the conversation loop.

    Uses standard multi-turn tool calling with LM Studio.
    """

    def __init__(
        self,
        llm_client: LMStudioClient,
        tool_handler: ToolHandler,
        max_turns: int = 10,
    ):
        self.llm = llm_client
        self.tools = tool_handler
        self.max_turns = max_turns
        self._sessions: Dict[str, Session] = {}

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        locale: Optional[UserLocale] = None,
    ) -> Session:
        """Get existing session or create a new one."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if locale:
                session.locale = locale
            return session

        new_id = session_id or str(uuid.uuid4())
        session = Session(
            session_id=new_id,
            locale=locale or UserLocale(),
        )
        self._sessions[new_id] = session
        self.tools.reset_session()  # Reset tool state for new session
        return session

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
                tool_result = await self._execute_tool(tool_call, session)

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
        """Process user input with streaming events for UI updates."""
        if request.model:
            logger.info("Using model override (stream): %s", request.model)
            self.llm.model = request.model

        session = self.get_or_create_session(request.session_id, request.locale)

        # Build system prompt
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

        for turn in range(self.max_turns):
            full_content = ""
            tool_calls_dict = {}
            emitted_names = set()

            async for chunk in self.llm.chat_completion_stream(
                messages=session.messages,
                tools=TOOL_DEFINITIONS,
            ):
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                
                delta = choices[0].get("delta", {})
                
                if "content" in delta and delta["content"]:
                    full_content += delta["content"]
                
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": tc.get("id", f"call_{idx}"),
                                "name": "",
                                "arguments": ""
                            }
                        
                        func = tc.get("function", {})
                        if "name" in func and func["name"]:
                            tool_calls_dict[idx]["name"] += func["name"]
                            
                            # Emit calculator selected event IMMEDIATELY when the name is known
                            name = tool_calls_dict[idx]["name"]
                            if name == "calc_info" and idx not in emitted_names:
                                # We don't have the calc_id yet, but we know it's loading a calculator
                                yield StreamEvent(
                                    type=EventType.CALCULATOR_SELECTED,
                                    data={"calc_id": "calculator details"},
                                )
                                await asyncio.sleep(0.01)
                                emitted_names.add(idx)

                        if "arguments" in func and func["arguments"]:
                            tool_calls_dict[idx]["arguments"] += func["arguments"]

            # Reconstruct the tool calls list
            extracted_tool_calls = []
            for idx, tc_data in sorted(tool_calls_dict.items()):
                try:
                    args = json.loads(tc_data["arguments"])
                except Exception:
                    args = {}
                
                extracted_tool_calls.append(
                    ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args,
                        raw_arguments=tc_data["arguments"]
                    )
                )

            if not extracted_tool_calls:
                # If no tool calls from the stream, maybe it's in the text (fallback)
                if full_content:
                    extracted_tool_calls = self.llm._parse_tool_calls_from_content(full_content)

            if not extracted_tool_calls:
                if full_content:
                    session.messages.append(Message(role="assistant", content=full_content))
                    yield StreamEvent(
                        type=EventType.ASSISTANT_MESSAGE,
                        data={"content": full_content},
                    )
                return

            # Add the assistant message with tool calls to session history
            session.messages.append(
                Message(
                    role="assistant",
                    content=full_content if full_content else None,
                    tool_calls=extracted_tool_calls,
                )
            )

            # Keep compatibility with existing tool execution loop
            for tool_call in extracted_tool_calls:
                # Emit events that require full arguments (calc_id, etc)
                if tool_call.name == "calc_info":
                    yield StreamEvent(
                        type=EventType.CALCULATOR_SELECTED,
                        data={"calc_id": tool_call.arguments.get("calc_id")},
                    )
                    await asyncio.sleep(0.01)
                elif tool_call.name == "execute_calc":
                    yield StreamEvent(
                        type=EventType.EXTRACTING_VARIABLES,
                        data={
                            "calc_id": tool_call.arguments.get("calc_id"),
                            "variables": tool_call.arguments.get("variables", {}),
                        },
                    )
                    # Force flush the extracting variables to the client
                    await asyncio.sleep(0.01)

                # Execute tool
                tool_result = await self._execute_tool(tool_call, session)

                # Add tool result message to history
                session.messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(tool_result),
                        tool_call_id=tool_call.id,
                    )
                )

                # Emit result events for execute_calc
                if tool_call.name == "execute_calc":
                    exec_result = ExecuteCalcResult(**tool_result)
                    session.last_calculator_id = tool_call.arguments.get("calc_id")
                    session.last_variables = tool_call.arguments.get("variables", {})

                    if exec_result.success:
                        yield StreamEvent(
                            type=EventType.CALCULATION_COMPLETE,
                            data={
                                "calc_id": tool_call.arguments.get("calc_id"),
                                "outputs": exec_result.outputs,
                                "audit_trace": exec_result.audit_trace,
                            },
                        )
                    else:
                        yield StreamEvent(
                            type=EventType.VALIDATION_ERROR,
                            data={"errors": exec_result.errors},
                        )

    async def _execute_tool(self, tool_call: ToolCall, session: Session) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        return await self.tools.execute_tool(tool_call.name, tool_call.arguments)

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
    def _attachment_to_content(attachment: InputAttachment) -> Optional[Dict[str, Any]]:
        mime_type = attachment.mime_type or "application/octet-stream"
        data_url = f"data:{mime_type};base64,{attachment.data}"
        if attachment.kind == "image":
            return {"type": "image_url", "image_url": {"url": data_url}}
        if attachment.kind == "audio":
            # LM Studio doesn't support audio parts yet
            return None
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

        parts: List[Dict[str, Any]] = []
        if text:
            parts.append({"type": "input_text", "text": text})

        for attachment in attachments:
            part = self._attachment_to_content(attachment)
            if part:
                parts.append(part)

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
