"""Pydantic models for AgentiCalc."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class UserLocale(BaseModel):
    """User's default unit preferences."""

    locale: str = "US"
    description: str = "U.S. Conventional Units"


class ExtractedVariable(BaseModel):
    """A variable extracted from user input."""

    key: str
    value: Union[float, int, bool, str, None]
    unit: Optional[str] = None
    source: Optional[str] = None  # Provenance: e.g., "transcript:0:14"
    confidence: Optional[float] = None


class ToolCallIntent(BaseModel):
    """Detected intent to call a tool."""

    tool_name: str
    raw_arguments: Optional[str] = None  # May be incomplete/malformed


class CalcInfoResult(BaseModel):
    """Result from calc_info tool."""

    calc_id: str
    title: str
    description: Optional[str] = None
    version: str
    tags: List[str] = []
    inputs: List[Dict[str, Any]]  # Input specifications
    presets: List[Dict[str, Any]] = []


class ExecuteCalcResult(BaseModel):
    """Result from execute_calc tool."""

    success: bool
    outputs: Optional[Dict[str, Any]] = None
    errors: List[Any] = []  # Can be strings or dicts from CalcSpec
    warnings: List[Any] = []  # Can be strings or dicts from CalcSpec
    audit_trace: Optional[Dict[str, Any]] = None

    def error_messages(self) -> List[str]:
        """Get error messages as strings."""
        msgs = []
        for e in self.errors:
            if isinstance(e, str):
                msgs.append(e)
            elif isinstance(e, dict):
                msgs.append(e.get("message", str(e)))
            else:
                msgs.append(str(e))
        return msgs


class EventType(str, Enum):
    """Event types for streaming responses."""

    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    THINKING = "thinking"
    CALCULATOR_SELECTED = "calculator_selected"
    EXTRACTING_VARIABLES = "extracting_variables"
    VARIABLES_EXTRACTED = "variables_extracted"
    VALIDATION_ERROR = "validation_error"
    CLARIFICATION_NEEDED = "clarification_needed"
    CALCULATION_COMPLETE = "calculation_complete"
    ASSISTANT_MESSAGE = "assistant_message"
    ERROR = "error"


class StreamEvent(BaseModel):
    """An event in the streaming response."""

    type: EventType
    data: Any
    timestamp: Optional[float] = None


class InputAttachment(BaseModel):
    """Optional multimodal input attachment."""

    kind: Literal["image", "audio"]
    data: str  # base64-encoded payload (no data URL prefix)
    mime_type: str
    name: Optional[str] = None


class OrchestratorRequest(BaseModel):
    """Request to the orchestrator endpoint."""

    input: str
    session_id: Optional[str] = None
    locale: Optional[UserLocale] = None
    calculator_hint: Optional[str] = None  # Optional hint for which calculator
    model: Optional[str] = None  # Optional LLM model override
    attachments: Optional[List[InputAttachment]] = None
    allowed_calculators: Optional[List[str]] = None  # Filter to specific calculator IDs
    mcp_url: Optional[str] = None  # Dynamic MCP URL based on request host


class OrchestratorResponse(BaseModel):
    """Final response from orchestrator (non-streaming)."""

    success: bool
    calculator_id: Optional[str] = None
    variables: List[ExtractedVariable] = []
    result: Optional[ExecuteCalcResult] = None
    clarification_question: Optional[str] = None
    assistant_message: Optional[str] = None
    errors: List[str] = []


# JSON Schema for structured extraction
def build_extraction_schema(inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a JSON schema for structured variable extraction
    based on calculator input specifications.

    Note: LM Studio/Outlines doesn't support union types like ["string", "null"].
    We use simple types and make fields optional via the required array.
    """
    properties = {}

    for inp in inputs:
        inp_id = inp["id"]
        inp_type = inp.get("type", "number")

        # Map CalcSpec types to JSON schema types
        # Note: unit/source are optional (not in required), so missing = null semantically
        if inp_type in ("number", "int"):
            prop_schema = {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["value"],
            }
        elif inp_type == "bool":
            prop_schema = {
                "type": "object",
                "properties": {
                    "value": {"type": "boolean"},
                },
                "required": ["value"],
            }
        elif inp_type == "enum":
            enum_values = inp.get("enum_values", [])
            prop_schema = {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": enum_values} if enum_values else {"type": "string"},
                },
                "required": ["value"],
            }
        else:
            prop_schema = {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                },
                "required": ["value"],
            }

        properties[inp_id] = prop_schema

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "extracted_variables",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "object",
                        "properties": properties,
                    },
                    "missing": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "ambiguous": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["variables", "missing", "ambiguous"],
            },
        },
    }
