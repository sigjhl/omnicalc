"""Tool definitions and handlers for the orchestrator agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import CalcInfoResult, ExecuteCalcResult
from .calculators import CALCULATORS

logger = logging.getLogger(__name__)

# Tool definitions for LLM function calling
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calc_info",
            "description": (
                "Get the input schema for a clinical calculator. "
                "Returns field names, types, units, and constraints needed for execute_calc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calc_id": {
                        "type": "string",
                        "description": "Calculator ID (e.g., meld_na, wells_dvt, child_pugh, apri, fib4)",
                    },
                },
                "required": ["calc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_calc",
            "description": (
                "Execute a calculation with extracted variables. "
                "Returns the calculated result or validation errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calc_id": {
                        "type": "string",
                        "description": "Calculator ID",
                    },
                    "variables": {
                        "type": "object",
                        "description": (
                            "Variables as key-value pairs. "
                            "For numeric inputs with units: {\"value\": number, \"unit\": string}. "
                            "For booleans: true/false. For enums: string value."
                        ),
                    },
                },
                "required": ["calc_id", "variables"],
            },
        },
    },
]


class ToolHandler:
    """
    Handles tool execution for the orchestrator.

    Connects to the CalcSpec API to get calculator info and execute calculations.
    Enforces the rule that calc_info must be called before execute_calc.
    """

    def __init__(self, calcspec_base_url: str = ""):
        # calcspec_base_url is kept for backwards compatibility but not used
        self._calc_info_cache: Dict[str, CalcInfoResult] = {}
        self._session_calc_info_calls: set[str] = set()
        self._calculator_list_cache: List[Dict[str, str]] = []

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def reset_session(self):
        """Reset session state (called when starting a new conversation)."""
        self._session_calc_info_calls.clear()

    def has_calc_info(self, calc_id: str) -> bool:
        """Check if calc_info was called for this calculator in current session."""
        resolved_id = self._resolve_calc_id(calc_id)
        return resolved_id in self._session_calc_info_calls

    def get_cached_calc_info(self, calc_id: str) -> Optional[CalcInfoResult]:
        """Get cached calc_info result."""
        resolved_id = self._resolve_calc_id(calc_id)
        return self._calc_info_cache.get(resolved_id)

    async def list_calculators(self) -> List[Dict[str, str]]:
        """Fetch available calculators locally."""
        try:
            self._calculator_list_cache = [
                {
                    "id": cid,
                    "title": c["def"].title,
                    "description": c["def"].description,
                    "version": c["def"].version
                }
                for cid, c in CALCULATORS.items()
            ]
            return self._calculator_list_cache
        except Exception as e:
            logger.error(f"Failed to list calculators: {e}")
            return self._calculator_list_cache  # Return stale cache on error

    def _resolve_calc_id(self, input_id: str) -> str:
        """
        Resolve a potentially malformed calculator ID to a valid one.
        Handles checking case-insensitive matches against the cached list.
        """
        if not self._calculator_list_cache:
            return input_id
        
        # 1. Exact match
        for calc in self._calculator_list_cache:
            if calc.get("id") == input_id:
                return input_id
                
        # 2. Case-insensitive match
        input_lower = input_id.lower()
        for calc in self._calculator_list_cache:
            cid = calc.get("id", "")
            if cid.lower() == input_lower:
                return cid
        
        # 3. Fallback: Return original and let it fail downstream if invalid
        return input_id

    async def calc_info(self, calc_id: str) -> CalcInfoResult:
        """
        Get calculator input schema.

        This fetches the full specification for a calculator including:
        - All input definitions (id, label, type, required, unit, constraints)
        - Accepted units with conversion factors
        - Synonyms for extraction matching
        - Available presets
        """
        try:
            if calc_id not in CALCULATORS:
                raise ValueError(f"Calculator '{calc_id}' not found")
                
            calc_def = CALCULATORS[calc_id]["def"]
            input_model = CALCULATORS[calc_id].get("input_model")

            inputs = []
            if input_model:
                for field_name, field_info in input_model.model_fields.items():
                    extra = field_info.json_schema_extra or {}
                    inputs.append({
                        "id": field_name,
                        "label": field_info.description or field_name,
                        "type": "number", # Defaulting to number for now
                        "required": field_info.is_required(),
                        "canonical_unit": extra.get("unit", ""),
                        "synonyms": extra.get("synonyms", []),
                        "constraints": extra.get("constraints", {})
                    })
            elif CALCULATORS[calc_id].get("schema"):
                # Schema-backed calculators expose prebuilt input definitions.
                inputs = CALCULATORS[calc_id]["schema"].get("inputs", [])

            result = CalcInfoResult(
                calc_id=calc_id,
                title=calc_def.title,
                description=calc_def.description,
                version=calc_def.version,
                tags=calc_def.tags,
                inputs=inputs,
                presets=calc_def.presets,
            )

            # Cache and mark as called
            self._calc_info_cache[calc_id] = result
            self._session_calc_info_calls.add(calc_id)

            return result

        except Exception as e:
            if "not found" in str(e).lower():
                raise ValueError(str(e))
            raise

    async def _get_schema_from_validation(self, calc_id: str) -> Dict[str, Any]:
        """
        Fallback: get schema info by making an empty run request
        and parsing the validation errors.
        """
        # This is a workaround until we add a proper schema endpoint
        # For now, return empty schema
        logger.warning(f"Schema endpoint not available for {calc_id}, using fallback")
        return {"inputs": [], "presets": [], "description": None}

    async def execute_calc(
        self, calc_id: str, variables: Dict[str, Any]
    ) -> ExecuteCalcResult:
        """
        Execute calculation with extracted variables.

        Enforces that calc_info must be called first for this calculator.

        Args:
            calc_id: Calculator ID
            variables: Extracted variables in CalcSpec input format

        Returns:
            ExecuteCalcResult with outputs or errors
        """
        # Auto-bootstrap schema if model skipped calc_info.
        if not self.has_calc_info(calc_id):
            try:
                await self.calc_info(calc_id)
            except ValueError as e:
                return ExecuteCalcResult(
                    success=False,
                    errors=[str(e)],
                )

        try:
            if calc_id not in CALCULATORS:
                return ExecuteCalcResult(
                    success=False,
                    errors=[f"Calculator '{calc_id}' not found"],
                )
            
            run_fn = CALCULATORS[calc_id]["run"]
            data = run_fn(variables)

            return ExecuteCalcResult(
                success=data.get("success", False),
                outputs=data.get("outputs"),
                errors=data.get("errors", []),
                warnings=data.get("warnings", []),
                audit_trace=data.get("audit_trace"),
            )

        except Exception as e:
            return ExecuteCalcResult(
                success=False,
                errors=[f"Calculation failed: {e}"],
            )

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool by name and return the result as a dict.

        This is the main entry point for tool execution from the orchestrator.
        """
        if tool_name == "calc_info":
            calc_id = arguments.get("calc_id")
            if not calc_id:
                return {"error": "Missing required parameter: calc_id"}
            try:
                result = await self.calc_info(calc_id)
                return result.model_dump()
            except ValueError as e:
                return {"error": str(e)}

        elif tool_name == "execute_calc":
            calc_id = arguments.get("calc_id")
            variables = arguments.get("variables", {})
            if not calc_id:
                return {"error": "Missing required parameter: calc_id"}
            result = await self.execute_calc(calc_id, variables)
            return result.model_dump()

        else:
            return {"error": f"Unknown tool: {tool_name}"}


def format_calc_info_for_extraction(calc_info: CalcInfoResult) -> str:
    """
    Format calculator info as a string prompt for the extraction phase.

    This is used in the structured extraction prompt to guide the LLM
    on what variables to extract and in what format.
    """
    lines = [
        f"Calculator: {calc_info.title} ({calc_info.calc_id})",
        "",
        "Required inputs:",
    ]

    for inp in calc_info.inputs:
        inp_id = inp.get("id", "unknown")
        label = inp.get("label", inp_id)
        inp_type = inp.get("type", "number")
        required = inp.get("required", False)
        unit = inp.get("canonical_unit", "")
        synonyms = inp.get("synonyms", [])
        constraints = inp.get("constraints", {})

        req_marker = "*" if required else ""
        unit_str = f" (default_unit: {unit})" if unit else ""

        line = f"  - {inp_id}{req_marker}: {label}{unit_str} [{inp_type}]"

        if synonyms:
            line += f" (also known as: {', '.join(synonyms)})"

        if constraints:
            if "min" in constraints:
                line += f" min={constraints['min']}"
            if "max" in constraints:
                line += f" max={constraints['max']}"

        lines.append(line)

    if calc_info.presets:
        lines.append("")
        lines.append("Available presets:")
        for preset in calc_info.presets:
            lines.append(f"  - {preset.get('id')}: {preset.get('name', '')}")

    return "\n".join(lines)
