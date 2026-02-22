"""
Local calculator registry for AgentiCalc.
Replaces the old CalcSpec DSL external server.
"""

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

class CalcInput(BaseModel):
    id: str
    label: str
    type: str = "number"
    required: bool = True
    canonical_unit: str = ""
    synonyms: List[str] = []
    constraints: Dict[str, Any] = {}

class CalculatorDef(BaseModel):
    id: str
    title: str
    description: str
    version: str
    tags: List[str]
    inputs: Any  # Keep for backwards compatibility if needed
    presets: List[Dict[str, Any]] = []

def _clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))

def _parse_var(raw: Any, default_unit: str) -> tuple[float, str]:
    """Parse a variable that might be a dict or a raw value."""
    if isinstance(raw, dict):
        return float(raw.get("value", 0)), raw.get("unit", default_unit)
    return float(raw) if raw is not None else 0.0, default_unit

from pydantic import Field

class MeldNaInput(BaseModel):
    """MELD-Na Score Inputs"""
    serum_bilirubin: Any = Field(..., description="Serum Bilirubin", json_schema_extra={"unit": "mg/dL", "synonyms": ["bili", "total bilirubin"]})
    inr: Any = Field(..., description="INR", json_schema_extra={"unit": "", "synonyms": ["prothrombin time ratio"]})
    serum_creatinine: Any = Field(..., description="Serum Creatinine", json_schema_extra={"unit": "mg/dL", "synonyms": ["cr", "creatinine", "creat"]})
    serum_sodium: Any = Field(..., description="Serum Sodium", json_schema_extra={"unit": "mEq/L", "synonyms": ["na", "sodium"]})

def run_meld_na(vars_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the MELD-Na calculation."""
    try:
        # In the tools.py it allows variables to be extracted from Dict
        # but also parses aliases. Pydantic aliases could be used, but since we rely on the schema
        # synonyms right now, let's parse using the fields.
        input_data = MeldNaInput.model_validate(vars_dict)
        
        raw_bili = input_data.serum_bilirubin
        raw_inr = input_data.inr
        raw_cr = input_data.serum_creatinine
        raw_na = input_data.serum_sodium
        
        bili_val, bili_unit = _parse_var(raw_bili, "mg/dL")
        inr_val, inr_unit = _parse_var(raw_inr, "")
        cr_val, cr_unit = _parse_var(raw_cr, "mg/dL")
        na_val, na_unit = _parse_var(raw_na, "mEq/L")

        # Conversions
        if bili_unit.lower() == "umol/l":
            bili_val = bili_val / 17.1
            bili_unit = "mg/dL (converted from umol/L)"
            
        if cr_unit.lower() == "umol/l":
            cr_val = cr_val / 88.4
            cr_unit = "mg/dL (converted from umol/L)"
            
        if na_unit.lower() == "mmol/l":
            # 1 mmol/L = 1 mEq/L for Sodium
            na_unit = "mEq/L"

        # MELD rules:
        # Bili, INR, Cr < 1.0 are set to 1.0
        # Cr > 4.0 is set to 4.0
        bili = max(1.0, bili_val)
        inr = max(1.0, inr_val)
        cr = _clamp(cr_val, 1.0, 4.0)

        # Na bounded between 125 and 137
        na = _clamp(na_val, 125.0, 137.0)

        # Standard MELD
        meld_i = 0.957 * math.log(cr) + 0.378 * math.log(bili) + 1.120 * math.log(inr) + 0.643
        meld = round(meld_i * 10)
        
        # Max MELD is 40
        meld = min(40, meld)

        if meld > 11:
            meld_na = meld + 1.32 * (137 - na) - (0.033 * meld * (137 - na))
            meld_na = round(meld_na)
        else:
            meld_na = meld

        meld_na = min(40, meld_na)

        inputs_used = {
            "serum_bilirubin": f"{bili_val:.2f} {bili_unit}".strip(),
            "inr": f"{inr_val:.2f}".strip(),
            "serum_creatinine": f"{cr_val:.2f} {cr_unit}".strip(),
            "serum_sodium": f"{na_val:.1f} {na_unit}".strip()
        }

        return {
            "success": True,
            "outputs": {
                "MELD Score": meld,
                "MELD-Na Score": meld_na
            },
            "audit_trace": {
                "inputs_used": inputs_used,
                "log": ["Computed MELD-Na based on standard OPTN formula."]
            },
            "errors": [],
            "warnings": []
        }
    except Exception as e:
        return {"success": False, "errors": [f"Calculation error: {e}"]}

# Registry of all calculators
CALCULATORS: Dict[str, Dict[str, Any]] = {
    "meld_na": {
        "def": CalculatorDef(
            id="meld_na",
            title="MELD-Na Score",
            description="Model for End-Stage Liver Disease (MELD) and MELD-Na score for 3-month mortality.",
            version="1.0",
            tags=["hepatology", "mortality"],
            inputs=None
        ),
        "run": run_meld_na,
        "input_model": MeldNaInput
    }
}
