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
    inputs: List[CalcInput]
    presets: List[Dict[str, Any]] = []

def _clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))

def _parse_var(raw: Any, default_unit: str) -> tuple[float, str]:
    """Parse a variable that might be a dict or a raw value."""
    if isinstance(raw, dict):
        return float(raw.get("value", 0)), raw.get("unit", default_unit)
    return float(raw) if raw is not None else 0.0, default_unit

def run_meld_na(vars_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the MELD-Na calculation."""
    try:
        raw_bili = vars_dict.get("serum_bilirubin", vars_dict.get("bilirubin"))
        raw_inr = vars_dict.get("inr")
        raw_cr = vars_dict.get("serum_creatinine", vars_dict.get("creatinine", vars_dict.get("cr")))
        raw_na = vars_dict.get("serum_sodium", vars_dict.get("sodium", vars_dict.get("na")))
        
        if raw_bili is None or raw_inr is None or raw_cr is None or raw_na is None:
            return {"success": False, "errors": ["Missing required variables for MELD-Na."]}
            
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
                "meld_score": meld,
                "meld_na_score": meld_na
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
            inputs=[
                CalcInput(id="serum_bilirubin", label="Serum Bilirubin", canonical_unit="mg/dL", synonyms=["bili", "total bilirubin"]),
                CalcInput(id="inr", label="INR", canonical_unit="", synonyms=["prothrombin time ratio"]),
                CalcInput(id="serum_creatinine", label="Serum Creatinine", canonical_unit="mg/dL", synonyms=["cr", "creatinine", "creat"]),
                CalcInput(id="serum_sodium", label="Serum Sodium", canonical_unit="mEq/L", synonyms=["na", "sodium"])
            ]
        ),
        "run": run_meld_na
    }
}
