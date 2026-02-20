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

def run_meld_na(vars: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the MELD-Na calculation."""
    try:
        bili = float(vars.get("serum_bilirubin", vars.get("bilirubin", 0)))
        inr = float(vars.get("inr", 0))
        cr = float(vars.get("serum_creatinine", vars.get("creatinine", vars.get("cr", 0))))
        na = float(vars.get("serum_sodium", vars.get("sodium", vars.get("na", 0))))
        
        if not (bili and inr and cr and na):
            return {"success": False, "errors": ["Missing required variables for MELD-Na."]}

        # MELD rules:
        # Bili, INR, Cr < 1.0 are set to 1.0
        # Cr > 4.0 is set to 4.0
        bili = max(1.0, bili)
        inr = max(1.0, inr)
        cr = _clamp(cr, 1.0, 4.0)

        # Na bounded between 125 and 137
        na = _clamp(na, 125.0, 137.0)

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

        return {
            "success": True,
            "outputs": {
                "meld_score": meld,
                "meld_na_score": meld_na
            },
            "audit_trace": {
                "inputs_used": {"serum_bilirubin": bili, "inr": inr, "serum_creatinine": cr, "serum_sodium": na},
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
