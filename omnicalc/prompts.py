"""System prompts for the orchestrator agent."""

from __future__ import annotations

from typing import Any, Dict, List

ORCHESTRATOR_SYSTEM_PROMPT = """You are OmniCalc, a clinical calculator assistant.

## Available Calculators
{calculator_list}

## Workflow
1. Identify which calculator to use from the clinical data
2. Call `calc_info` to get the exact input field names and units
3. Call `execute_calc` with extracted variables using the exact field IDs from the schema
4. After successful calculation, STOP calling tools. Respond with just "Done."

## Rules
- If the user explicitly provides a unit, use it.
- If no unit is provided, assume the User's Locale Default below is in effect.
- **CRITICAL**: If the User's Locale Default differs from the calculator's `canonical_unit` (or if you are unsure), you MUST pass the unit along with the value as `{{"value": number, "unit": "string"}}`.
- If required variables are missing, ask for the specific missing values.
- If calculation fails, state the error briefly.
- Never perform arithmetic yourself - always use `execute_calc`.
- NEVER call `execute_calc` again if you already have a successful result.
- Be concise. No interpretation or explanation unless asked.

## Locale Defaults
- {locale_description}
"""


EXTRACTION_PROMPT_TEMPLATE = """Extract clinical variables from the following input for the {calculator_name} calculator.

## Calculator Schema
{schema_description}

## User Input
{user_input}

## Instructions
- Extract each variable mentioned in the input
- Include the unit if explicitly stated, otherwise use null
- For numeric values, extract the number only
- For boolean values, look for explicit yes/no, true/false, or presence/absence statements
- Mark variables as missing if they cannot be found in the input
- Mark variables as ambiguous if multiple conflicting values are found

Extract the variables now:"""


CLARIFICATION_PROMPT_TEMPLATE = """The following required variables are missing for {calculator_name}:

Missing variables:
{missing_list}

Please ask the clinician for these specific values in a natural, concise way."""


def build_system_prompt(
    calculators: List[Dict[str, str]],
    locale_description: str,
) -> str:
    """Build the system prompt with available calculators and locale settings."""

    # Format calculator list
    calc_lines = []
    for calc in calculators:
        calc_id = calc.get("id", "unknown")
        desc = calc.get("description", "")
        # Format: - {id} - {description}
        calc_lines.append(f"- {calc_id} - {desc}")

    calculator_list = "\n".join(calc_lines) if calc_lines else "- No calculators loaded"

    return ORCHESTRATOR_SYSTEM_PROMPT.format(
        calculator_list=calculator_list,
        locale_description=locale_description,
    )


def build_extraction_prompt(
    calculator_name: str,
    schema_description: str,
    user_input: str,
) -> str:
    """Build the extraction prompt for structured generation."""
    return EXTRACTION_PROMPT_TEMPLATE.format(
        calculator_name=calculator_name,
        schema_description=schema_description,
        user_input=user_input,
    )


def build_clarification_prompt(
    calculator_name: str,
    missing_variables: List[Dict[str, Any]],
) -> str:
    """Build a prompt to generate a clarification question."""
    missing_list = []
    for var in missing_variables:
        var_id = var.get("id", "unknown")
        label = var.get("label", var_id)
        unit = var.get("canonical_unit", "")
        unit_str = f" (in {unit})" if unit else ""
        missing_list.append(f"- {label}{unit_str}")

    return CLARIFICATION_PROMPT_TEMPLATE.format(
        calculator_name=calculator_name,
        missing_list="\n".join(missing_list),
    )
