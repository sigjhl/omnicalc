"""
Local calculator registry for OmniCalc.
All 55 MedCalc-Bench calculators with schemas matching the training data format.
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Schema models ────────────────────────────────────────────────────────────

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
    inputs: Any  # kept for backward compat
    presets: List[Dict[str, Any]] = []


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(val, hi))


def _parse_var(raw: Any, default_unit: str = "") -> tuple[float, str]:
    """Parse a variable that might be {value, unit} dict or raw number."""
    if isinstance(raw, dict):
        return float(raw.get("value", 0)), raw.get("unit", default_unit)
    return (float(raw) if raw is not None else 0.0), default_unit


def _parse_num(raw: Any, default: float = 0.0) -> float:
    """Parse a numeric variable (ignore unit)."""
    if isinstance(raw, dict):
        return float(raw.get("value", default))
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# ── Unit conversion ─────────────────────────────────────────────────────────

_UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    "height":        {"m": 100, "in": 2.54, "inches": 2.54, "ft": 30.48, "feet": 30.48},
    "weight":        {"lb": 1/2.20462, "lbs": 1/2.20462, "pounds": 1/2.20462, "g": 0.001},
    "creatinine":    {"µmol/l": 1/88.4, "umol/l": 1/88.4, "μmol/l": 1/88.4, "micromol/l": 1/88.4},
    "glucose":       {"mmol/l": 18.0182},
    "bilirubin":     {"µmol/l": 1/17.1, "umol/l": 1/17.1, "μmol/l": 1/17.1},
    "albumin":       {"g/l": 0.1},
    "bun":           {"mmol/l": 2.8011},
    "hemoglobin":    {"g/l": 0.1},
    "cholesterol":   {"mmol/l": 38.67},
    "hdl":           {"mmol/l": 38.67},
    "triglycerides": {"mmol/l": 88.57},
    "pressure":      {"kpa": 7.50062},
    "qt_interval":   {"s": 1000, "sec": 1000, "seconds": 1000},
    "electrolyte":   {"mmol/l": 1.0},
    "wbc":           {"mm^3": 0.001, "µl": 0.001, "ul": 0.001, "/ul": 0.001, "l": 0.001, "/l": 0.001,
                      "/µl": 0.001, "/mm^3": 0.001, "cells/µl": 0.001, "cells/ul": 0.001},
    "platelets":     {"µl": 0.001, "ul": 0.001, "/ul": 0.001, "/µl": 0.001, "mm^3": 0.001,
                      "/mm^3": 0.001, "cells/µl": 0.001, "cells/ul": 0.001},
}


def _convert(raw: Any, canonical_unit: str = "", analyte: str = "", default: float = 0.0) -> float:
    """Parse a numeric variable and convert to canonical unit if needed.

    Handles:
      - raw numbers or {"value": N} → returned as-is
      - {"value": N, "unit": U} → converted if U differs from canonical
      - Temperature F→C (special formula)
      - FiO2 fraction→% (auto-detect if value ≤ 1)
    """
    if isinstance(raw, dict):
        val = float(raw.get("value", default))
        unit = raw.get("unit", "").strip()
    elif raw is None:
        return default
    else:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return default
        return val  # no unit info → assume canonical

    if not unit:
        return val

    unit_lower = unit.lower()
    canonical_lower = canonical_unit.lower()

    # Already in canonical unit
    if unit_lower == canonical_lower:
        return val

    # Temperature: F → C
    if analyte == "temperature" and unit_lower in ("f", "°f", "fahrenheit"):
        return (val - 32) * 5 / 9

    # FiO2: auto-detect fraction (0-1) vs percent
    if analyte == "fio2" and canonical_lower == "%" and val <= 1.0:
        return val * 100

    # Lookup table conversion
    factors = _UNIT_CONVERSIONS.get(analyte, {})
    factor = factors.get(unit_lower)
    if factor is not None:
        return val * factor

    # Fallback: return as-is
    return val


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, dict):
        return _parse_bool(raw.get("value", False))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.lower() in ("true", "yes", "1")
    if isinstance(raw, (int, float)):
        return bool(raw)
    return False


def _parse_str(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("value", ""))
    return str(raw) if raw is not None else ""


def _ok(result: Any, inputs_used: Dict[str, str], log: List[str]) -> Dict[str, Any]:
    return {
        "success": True,
        "outputs": {"result": result},
        "audit_trace": {"inputs_used": inputs_used, "log": log},
        "errors": [],
        "warnings": [],
    }


def _err(msg: str) -> Dict[str, Any]:
    return {"success": False, "outputs": {}, "errors": [msg], "warnings": []}


# ── Calculator implementations ──────────────────────────────────────────────

# 1. Mean Arterial Pressure ──────────────────────────────────────────────────
def run_mean_arterial_pressure(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sbp = _convert(v.get("systolic_bp"), "mmHg", "pressure")
        dbp = _convert(v.get("diastolic_bp"), "mmHg", "pressure")
        result = (1 / 3) * sbp + (2 / 3) * dbp
        return _ok(round(result, 5), {"systolic_bp": str(sbp), "diastolic_bp": str(dbp)}, ["MAP = (1/3)*SBP + (2/3)*DBP"])
    except Exception as e:
        return _err(str(e))


# 2. BMI ──────────────────────────────────────────────────────────────────────
def run_bmi(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        wt = _convert(v.get("weight"), "kg", "weight")
        ht_cm = _convert(v.get("height"), "cm", "height")
        ht_m = ht_cm / 100
        result = wt / (ht_m * ht_m)
        return _ok(round(result, 5), {"weight": f"{wt} kg", "height": f"{ht_cm} cm"}, ["BMI = weight / height^2"])
    except Exception as e:
        return _err(str(e))


# 3. BSA ──────────────────────────────────────────────────────────────────────
def run_bsa(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        wt = _convert(v.get("weight"), "kg", "weight")
        ht_cm = _convert(v.get("height"), "cm", "height")
        result = math.sqrt((wt * ht_cm) / 3600)
        return _ok(round(result, 5), {"weight": f"{wt} kg", "height": f"{ht_cm} cm"}, ["BSA = sqrt(weight*height/3600) (Mosteller)"])
    except Exception as e:
        return _err(str(e))


# 4. Ideal Body Weight ────────────────────────────────────────────────────────
def run_ideal_body_weight(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sex = _parse_str(v.get("sex")).lower()
        ht_cm = _convert(v.get("height"), "cm", "height")
        ht_in = ht_cm / 2.54
        if "female" in sex or sex == "f":
            ibw = 45.5 + 2.3 * (ht_in - 60)
        else:
            ibw = 50.0 + 2.3 * (ht_in - 60)
        return _ok(round(ibw, 5), {"sex": sex, "height": f"{ht_cm} cm"}, ["Devine formula"])
    except Exception as e:
        return _err(str(e))


# 5. Adjusted Body Weight ─────────────────────────────────────────────────────
def run_adjusted_body_weight(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sex = _parse_str(v.get("sex")).lower()
        wt = _convert(v.get("weight"), "kg", "weight")
        ht_cm = _convert(v.get("height"), "cm", "height")
        ht_in = ht_cm / 2.54
        if "female" in sex or sex == "f":
            ibw = 45.5 + 2.3 * (ht_in - 60)
        else:
            ibw = 50.0 + 2.3 * (ht_in - 60)
        abw = ibw + 0.4 * (wt - ibw)
        return _ok(round(abw, 5), {"sex": sex, "weight": f"{wt} kg", "height": f"{ht_cm} cm"}, ["ABW = IBW + 0.4*(actual - IBW)"])
    except Exception as e:
        return _err(str(e))


# 6. Target Weight ────────────────────────────────────────────────────────────
def run_target_weight(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        bmi_val = _parse_num(v.get("bmi"))
        ht_cm = _convert(v.get("height"), "cm", "height")
        ht_m = ht_cm / 100
        result = bmi_val * (ht_m ** 2)
        return _ok(round(result, 5), {"bmi": str(bmi_val), "height": f"{ht_cm} cm"}, ["target_weight = BMI * height_m^2"])
    except Exception as e:
        return _err(str(e))


# 7. Calcium Correction ───────────────────────────────────────────────────────
def run_calcium_correction(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        ca = _parse_num(v.get("serum_calcium"))
        alb = _convert(v.get("serum_albumin"), "g/dL", "albumin")
        result = 0.8 * (4.0 - alb) + ca
        return _ok(round(result, 5), {"calcium": str(ca), "albumin": str(alb)}, ["corrected_Ca = 0.8*(4-albumin) + Ca"])
    except Exception as e:
        return _err(str(e))


# 8. Anion Gap ────────────────────────────────────────────────────────────────
def run_anion_gap(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        cl = _convert(v.get("serum_chloride"), "mEq/L", "electrolyte")
        hco3 = _convert(v.get("serum_bicarbonate"), "mEq/L", "electrolyte")
        result = na - (cl + hco3)
        return _ok(round(result, 5), {"Na": str(na), "Cl": str(cl), "HCO3": str(hco3)}, ["AG = Na - (Cl + HCO3)"])
    except Exception as e:
        return _err(str(e))


# 9. Delta Gap ────────────────────────────────────────────────────────────────
def run_delta_gap(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        cl = _convert(v.get("serum_chloride"), "mEq/L", "electrolyte")
        hco3 = _convert(v.get("serum_bicarbonate"), "mEq/L", "electrolyte")
        ag = na - (cl + hco3)
        result = ag - 12
        return _ok(round(result, 5), {"Na": str(na), "Cl": str(cl), "HCO3": str(hco3)}, ["Delta Gap = AG - 12"])
    except Exception as e:
        return _err(str(e))


# 10. Delta Ratio ─────────────────────────────────────────────────────────────
def run_delta_ratio(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        cl = _convert(v.get("serum_chloride"), "mEq/L", "electrolyte")
        hco3 = _convert(v.get("serum_bicarbonate"), "mEq/L", "electrolyte")
        ag = na - (cl + hco3)
        delta_gap = ag - 12
        denom = 24 - hco3
        if denom == 0:
            return _err("Division by zero: 24 - HCO3 = 0")
        result = delta_gap / denom
        return _ok(round(result, 5), {"Na": str(na), "Cl": str(cl), "HCO3": str(hco3)}, ["Delta Ratio = (AG-12)/(24-HCO3)"])
    except Exception as e:
        return _err(str(e))


# 11. Albumin Corrected Anion Gap ─────────────────────────────────────────────
def run_albumin_corrected_anion_gap(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        cl = _convert(v.get("serum_chloride"), "mEq/L", "electrolyte")
        hco3 = _convert(v.get("serum_bicarbonate"), "mEq/L", "electrolyte")
        alb = _convert(v.get("serum_albumin"), "g/dL", "albumin")
        ag = na - (cl + hco3)
        result = ag + 2.5 * (4 - alb)
        return _ok(round(result, 5), {"Na": str(na), "Cl": str(cl), "HCO3": str(hco3), "albumin": str(alb)}, ["Alb-corrected AG = AG + 2.5*(4-albumin)"])
    except Exception as e:
        return _err(str(e))


# 12. Albumin Corrected Delta Gap ─────────────────────────────────────────────
def run_albumin_corrected_delta_gap(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        cl = _convert(v.get("serum_chloride"), "mEq/L", "electrolyte")
        hco3 = _convert(v.get("serum_bicarbonate"), "mEq/L", "electrolyte")
        alb = _convert(v.get("serum_albumin"), "g/dL", "albumin")
        ag = na - (cl + hco3)
        corrected_ag = ag + 2.5 * (4 - alb)
        result = corrected_ag - 12
        return _ok(round(result, 5), {"Na": str(na), "Cl": str(cl), "HCO3": str(hco3), "albumin": str(alb)}, ["Alb-corrected delta gap = corrected_AG - 12"])
    except Exception as e:
        return _err(str(e))


# 13. Albumin Corrected Delta Ratio ───────────────────────────────────────────
def run_albumin_corrected_delta_ratio(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        cl = _convert(v.get("serum_chloride"), "mEq/L", "electrolyte")
        hco3 = _convert(v.get("serum_bicarbonate"), "mEq/L", "electrolyte")
        alb = _convert(v.get("serum_albumin"), "g/dL", "albumin")
        ag = na - (cl + hco3)
        corrected_ag = ag + 2.5 * (4 - alb)
        corrected_delta_gap = corrected_ag - 12
        denom = 24 - hco3
        if denom == 0:
            return _err("Division by zero: 24 - HCO3 = 0")
        result = corrected_delta_gap / denom
        return _ok(round(result, 5), {"Na": str(na), "Cl": str(cl), "HCO3": str(hco3), "albumin": str(alb)}, ["Alb-corrected delta ratio"])
    except Exception as e:
        return _err(str(e))


# 14. LDL Calculated ──────────────────────────────────────────────────────────
def run_ldl_calculated(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        tc = _convert(v.get("total_cholesterol"), "mg/dL", "cholesterol")
        hdl = _convert(v.get("hdl_cholesterol"), "mg/dL", "hdl")
        tg = _convert(v.get("triglycerides"), "mg/dL", "triglycerides")
        result = tc - hdl - (tg / 5)
        return _ok(round(result, 5), {"TC": str(tc), "HDL": str(hdl), "TG": str(tg)}, ["Friedewald: LDL = TC - HDL - TG/5"])
    except Exception as e:
        return _err(str(e))


# 15. Sodium Correction for Hyperglycemia ─────────────────────────────────────
def run_sodium_correction_hyperglycemia(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        glu = _convert(v.get("serum_glucose"), "mg/dL", "glucose")
        result = na + 0.024 * (glu - 100)
        return _ok(round(result, 5), {"Na": str(na), "glucose": str(glu)}, ["Hillier 1999: corrected_Na = Na + 0.024*(glucose-100)"])
    except Exception as e:
        return _err(str(e))


# 16. Serum Osmolality ────────────────────────────────────────────────────────
def run_serum_osmolality(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        bun = _convert(v.get("bun"), "mg/dL", "bun")
        glu = _convert(v.get("serum_glucose"), "mg/dL", "glucose")
        result = 2 * na + (bun / 2.8) + (glu / 18)
        return _ok(round(result, 5), {"Na": str(na), "BUN": str(bun), "glucose": str(glu)}, ["sOsm = 2*Na + BUN/2.8 + glucose/18"])
    except Exception as e:
        return _err(str(e))


# 17. HOMA-IR ─────────────────────────────────────────────────────────────────
def run_homa_ir(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        glu = _convert(v.get("serum_glucose"), "mg/dL", "glucose")
        ins = _parse_num(v.get("serum_insulin"))
        result = (ins * glu) / 405
        return _ok(round(result, 5), {"glucose": str(glu), "insulin": str(ins)}, ["HOMA-IR = (insulin * glucose) / 405"])
    except Exception as e:
        return _err(str(e))


# 18. FENa ────────────────────────────────────────────────────────────────────
def run_fena(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        cr = _convert(v.get("serum_creatinine"), "mg/dL", "creatinine")
        u_na = _convert(v.get("urine_sodium"), "mEq/L", "electrolyte")
        u_cr = _convert(v.get("urine_creatinine"), "mg/dL", "creatinine")
        if na * u_cr == 0:
            return _err("Division by zero in FENa")
        result = (cr * u_na) / (na * u_cr) * 100
        return _ok(round(result, 5), {"Na": str(na), "Cr": str(cr), "uNa": str(u_na), "uCr": str(u_cr)}, ["FENa = (Cr*uNa)/(Na*uCr)*100"])
    except Exception as e:
        return _err(str(e))


# 19. FIB-4 ───────────────────────────────────────────────────────────────────
def run_fib4(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        age = _parse_num(v.get("age"))
        ast = _parse_num(v.get("ast"))
        alt = _parse_num(v.get("alt"))
        plt = _convert(v.get("platelet_count"), "10^9/L", "platelets")
        plt_billions = plt
        if alt <= 0 or plt_billions <= 0:
            return _err("ALT and platelet count must be > 0")
        result = (age * ast) / (plt_billions * math.sqrt(alt))
        return _ok(round(result, 5), {"age": str(age), "AST": str(ast), "ALT": str(alt), "platelets": str(plt)}, ["FIB4 = (age*AST)/(platelets*sqrt(ALT))"])
    except Exception as e:
        return _err(str(e))


# 20. Maintenance Fluids ──────────────────────────────────────────────────────
def run_maintenance_fluids(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        wt = _convert(v.get("weight"), "kg", "weight")
        if wt <= 0:
            return _err("Weight must be > 0")
        if wt < 10:
            rate = wt * 4
        elif wt <= 20:
            rate = 40 + 2 * (wt - 10)
        else:
            rate = 60 + 1 * (wt - 20)
        return _ok(round(rate, 5), {"weight": f"{wt} kg"}, ["4-2-1 rule"])
    except Exception as e:
        return _err(str(e))


# 21. Free Water Deficit ──────────────────────────────────────────────────────
def run_free_water_deficit(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        age = _parse_num(v.get("age"))
        sex = _parse_str(v.get("sex")).lower()
        wt = _convert(v.get("weight"), "kg", "weight")
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        is_female = "female" in sex or sex == "f"
        if age < 18:
            tbw_pct = 0.6
        elif age >= 65:
            tbw_pct = 0.45 if is_female else 0.5
        else:
            tbw_pct = 0.5 if is_female else 0.6
        result = tbw_pct * wt * (na / 140 - 1)
        return _ok(round(result, 5), {"age": str(age), "sex": sex, "weight": f"{wt} kg", "Na": str(na)}, ["FWD = TBW * weight * (Na/140 - 1)"])
    except Exception as e:
        return _err(str(e))


# 22. QTc Bazett ──────────────────────────────────────────────────────────────
def run_qtc_bazett(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        hr = _parse_num(v.get("heart_rate"))
        qt = _convert(v.get("qt_interval"), "msec", "qt_interval")
        rr = 60 / hr
        result = qt / math.sqrt(rr)
        return _ok(round(result, 5), {"HR": str(hr), "QT": str(qt)}, ["QTc = QT / sqrt(RR)"])
    except Exception as e:
        return _err(str(e))


# 23. QTc Fridericia ──────────────────────────────────────────────────────────
def run_qtc_fridericia(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        hr = _parse_num(v.get("heart_rate"))
        qt = _convert(v.get("qt_interval"), "msec", "qt_interval")
        rr = 60 / hr
        result = qt / (rr ** (1 / 3))
        return _ok(round(result, 5), {"HR": str(hr), "QT": str(qt)}, ["QTc = QT / RR^(1/3)"])
    except Exception as e:
        return _err(str(e))


# 24. QTc Framingham ──────────────────────────────────────────────────────────
def run_qtc_framingham(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        hr = _parse_num(v.get("heart_rate"))
        qt = _convert(v.get("qt_interval"), "msec", "qt_interval")
        rr = 60 / hr
        result = qt + 154 * (1 - rr)
        return _ok(round(result, 5), {"HR": str(hr), "QT": str(qt)}, ["QTc = QT + 154*(1-RR)"])
    except Exception as e:
        return _err(str(e))


# 25. QTc Hodges ──────────────────────────────────────────────────────────────
def run_qtc_hodges(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        hr = _parse_num(v.get("heart_rate"))
        qt = _convert(v.get("qt_interval"), "msec", "qt_interval")
        rr = 60 / hr
        result = qt + 1.75 * ((60 / rr) - 60)
        return _ok(round(result, 5), {"HR": str(hr), "QT": str(qt)}, ["QTc = QT + 1.75*(HR-60)"])
    except Exception as e:
        return _err(str(e))


# 26. QTc Rautaharju ──────────────────────────────────────────────────────────
def run_qtc_rautaharju(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        hr = _parse_num(v.get("heart_rate"))
        qt = _convert(v.get("qt_interval"), "msec", "qt_interval")
        result = qt * (120 + hr) / 180
        return _ok(round(result, 5), {"HR": str(hr), "QT": str(qt)}, ["QTc = QT*(120+HR)/180"])
    except Exception as e:
        return _err(str(e))


# 27. Creatinine Clearance (Cockcroft-Gault) ──────────────────────────────────
def run_creatinine_clearance(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        age = _parse_num(v.get("age"))
        sex = _parse_str(v.get("sex")).lower()
        wt = _convert(v.get("weight"), "kg", "weight")
        ht_cm = _convert(v.get("height"), "cm", "height")
        cr = _convert(v.get("serum_creatinine"), "mg/dL", "creatinine")

        is_female = "female" in sex or sex == "f"
        gender_coeff = 0.85 if is_female else 1.0

        # Compute BMI
        ht_m = ht_cm / 100
        bmi = wt / (ht_m * ht_m)

        # Compute IBW (Devine)
        ht_in = ht_cm / 2.54
        if is_female:
            ibw = 45.5 + 2.3 * (ht_in - 60)
        else:
            ibw = 50.0 + 2.3 * (ht_in - 60)

        # Adjusted weight selection
        if bmi < 18.5:  # underweight
            adj_wt = wt
        elif bmi <= 24.9:  # normal
            adj_wt = min(ibw, wt)
        else:  # overweight/obese
            adj_wt = ibw + 0.4 * (wt - ibw)

        result = ((140 - age) * adj_wt * gender_coeff) / (cr * 72)
        return _ok(round(result, 5), {
            "age": str(age), "sex": sex, "weight": f"{wt} kg",
            "height": f"{ht_cm} cm", "creatinine": f"{cr} mg/dL",
            "adjusted_weight": f"{round(adj_wt, 2)} kg", "BMI": f"{round(bmi, 2)}"
        }, ["Cockcroft-Gault with adjusted body weight"])
    except Exception as e:
        return _err(str(e))


# 28. CKD-EPI GFR (2021) ─────────────────────────────────────────────────────
def run_ckd_epi_gfr(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        age = _parse_num(v.get("age"))
        sex = _parse_str(v.get("sex")).lower()
        cr = _convert(v.get("serum_creatinine"), "mg/dL", "creatinine")
        is_female = "female" in sex or sex == "f"
        if is_female:
            A, gender_coeff = 0.7, 1.012
            B = -0.241 if cr <= 0.7 else -1.2
        else:
            A, gender_coeff = 0.9, 1.0
            B = -0.302 if cr <= 0.9 else -1.2
        result = 142 * ((cr / A) ** B) * (0.9938 ** age) * gender_coeff
        return _ok(round(result, 5), {"age": str(age), "sex": sex, "creatinine": str(cr)}, ["CKD-EPI 2021 creatinine equation"])
    except Exception as e:
        return _err(str(e))


# 29. MDRD GFR ────────────────────────────────────────────────────────────────
def run_mdrd_gfr(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        age = _parse_num(v.get("age"))
        sex = _parse_str(v.get("sex")).lower()
        cr = _convert(v.get("serum_creatinine"), "mg/dL", "creatinine")
        race = _parse_str(v.get("race", "")).lower()
        is_female = "female" in sex or sex == "f"
        gender_coeff = 0.742 if is_female else 1.0
        race_coeff = 1.212 if "black" in race else 1.0
        result = 175 * (cr ** -1.154) * (age ** -0.203) * gender_coeff * race_coeff
        return _ok(round(result, 5), {"age": str(age), "sex": sex, "creatinine": str(cr), "race": race}, ["MDRD GFR = 175 * Cr^-1.154 * age^-0.203 * gender * race"])
    except Exception as e:
        return _err(str(e))


# 30. MELD-Na ─────────────────────────────────────────────────────────────────
def run_meld_na(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cr_val = _convert(v.get("serum_creatinine"), "mg/dL", "creatinine")
        bili_val = _convert(v.get("serum_bilirubin"), "mg/dL", "bilirubin")
        inr_val = _parse_num(v.get("inr"))
        na_val = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        dialysis = _parse_bool(v.get("dialysis_twice", False))
        cvvhd = _parse_bool(v.get("cvvhd", False))

        # Clamping
        if dialysis or cvvhd:
            cr_val = max(cr_val, 4.0)
        cr = _clamp(max(1.0, cr_val), 1.0, 4.0)
        bili = max(1.0, bili_val)
        inr = max(1.0, inr_val)
        na = _clamp(na_val, 125.0, 137.0)

        meld_i = 0.957 * math.log(cr) + 0.378 * math.log(bili) + 1.120 * math.log(inr) + 0.643
        meld = round(meld_i * 10)
        meld = min(40, meld)

        if meld > 11:
            meld_na = meld + 1.32 * (137 - na) - (0.033 * meld * (137 - na))
            meld_na = round(meld_na)
        else:
            meld_na = meld

        meld_na = min(40, meld_na)
        return _ok(meld_na, {
            "creatinine": str(cr_val), "bilirubin": str(bili_val),
            "INR": str(inr_val), "sodium": str(na_val)
        }, ["MELD-Na (UNOS/OPTN)"])
    except Exception as e:
        return _err(str(e))


# 31. CHA2DS2-VASc ────────────────────────────────────────────────────────────
def run_cha2ds2_vasc(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        age = _parse_num(v.get("age"))
        sex = _parse_str(v.get("sex")).lower()
        score = 0
        if age >= 75:
            score += 2
        elif age >= 65:
            score += 1
        if "female" in sex or sex == "f":
            score += 1
        if _parse_bool(v.get("chf", False)):
            score += 1
        if _parse_bool(v.get("hypertension", False)):
            score += 1
        # Stroke/TIA/thromboembolism: +2 if any
        has_stroke_tia = (_parse_bool(v.get("stroke", False)) or
                          _parse_bool(v.get("tia", False)) or
                          _parse_bool(v.get("thromboembolism", False)))
        if has_stroke_tia:
            score += 2
        if _parse_bool(v.get("vascular_disease", False)):
            score += 1
        if _parse_bool(v.get("diabetes", False)):
            score += 1
        return _ok(score, {"age": str(age), "sex": sex}, ["CHA2DS2-VASc scoring"])
    except Exception as e:
        return _err(str(e))


# 32. Wells DVT ───────────────────────────────────────────────────────────────
def run_wells_dvt(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        if _parse_bool(v.get("active_cancer", False)):
            score += 1
        # Bedridden >3 days OR major surgery (combined = 1 point)
        bedridden = _parse_bool(v.get("bedridden_for_atleast_3_days", False))
        major_surg = _parse_bool(v.get("major_surgery_in_last_12_weeks", False))
        if bedridden or major_surg:
            score += 1
        if _parse_bool(v.get("calf_swelling_3cm", False)):
            score += 1
        if _parse_bool(v.get("collateral_superficial_veins", False)):
            score += 1
        if _parse_bool(v.get("leg_swollen", False)):
            score += 1
        if _parse_bool(v.get("localized_tenderness_on_deep_venuous_system", False)):
            score += 1
        if _parse_bool(v.get("pitting_edema_on_symptomatic_leg", False)):
            score += 1
        if _parse_bool(v.get("paralysis_paresis_immobilization_in_lower_extreme", False)):
            score += 1
        if _parse_bool(v.get("previous_dvt_documented", False)):
            score += 1
        if _parse_bool(v.get("alternative_to_dvt_diagnosis", False)):
            score -= 2
        return _ok(score, {}, ["Wells' DVT criteria"])
    except Exception as e:
        return _err(str(e))


# 33. Wells PE ────────────────────────────────────────────────────────────────
def run_wells_pe(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0.0
        if _parse_bool(v.get("clinical_dvt", False)):
            score += 3
        if _parse_bool(v.get("pe_number_one", False)):
            score += 3
        hr = _parse_num(v.get("heart_rate", 0))
        if hr > 100:
            score += 1.5
        immob = _parse_bool(v.get("immobilization_for_3days", False))
        surg = _parse_bool(v.get("surgery_in_past4weeks", False))
        if immob or surg:
            score += 1.5
        if _parse_bool(v.get("previous_pe", False)) or _parse_bool(v.get("previous_dvt", False)):
            score += 1.5
        if _parse_bool(v.get("hemoptysis", False)):
            score += 1
        if _parse_bool(v.get("malignancy_with_treatment", False)):
            score += 1
        return _ok(score, {"HR": str(hr)}, ["Wells' PE criteria"])
    except Exception as e:
        return _err(str(e))


# 34. PERC Rule ───────────────────────────────────────────────────────────────
def run_perc_rule(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        count = 0
        age = _parse_num(v.get("age", 0))
        if age >= 50:
            count += 1
        hr = _parse_num(v.get("heart_rate", 0))
        if hr >= 100:
            count += 1
        o2 = _parse_num(v.get("o2_saturation", 100))
        if o2 < 95:
            count += 1
        if _parse_bool(v.get("unilateral_leg_swelling", False)):
            count += 1
        if _parse_bool(v.get("hemoptysis", False)):
            count += 1
        if _parse_bool(v.get("recent_surgery_or_trauma", False)):
            count += 1
        if _parse_bool(v.get("previous_pe", False)) or _parse_bool(v.get("previous_dvt", False)):
            count += 1
        if _parse_bool(v.get("hormonal_use", False)):
            count += 1
        return _ok(count, {"age": str(age), "HR": str(hr), "O2": str(o2)}, ["PERC rule criteria count"])
    except Exception as e:
        return _err(str(e))


# 35. SIRS ────────────────────────────────────────────────────────────────────
def run_sirs(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        count = 0
        temp = _convert(v.get("temperature"), "°C", "temperature")
        if temp > 38 or temp < 36:
            count += 1
        hr = _parse_num(v.get("heart_rate"))
        if hr > 90:
            count += 1
        rr = _parse_num(v.get("respiratory_rate"))
        paco2 = _convert(v.get("paco2", 40), "mmHg", "pressure")
        if rr > 20 or paco2 < 32:
            count += 1
        wbc = _convert(v.get("wbc"), "10^9/L", "wbc")
        if wbc > 12 or wbc < 4:
            count += 1
        return _ok(count, {"temp": str(temp), "HR": str(hr), "RR": str(rr), "WBC": str(wbc)}, ["SIRS criteria count"])
    except Exception as e:
        return _err(str(e))


# 36. CURB-65 ─────────────────────────────────────────────────────────────────
def run_curb65(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        age = _parse_num(v.get("age"))
        if age >= 65:
            score += 1
        if _parse_bool(v.get("confusion", False)):
            score += 1
        bun = _convert(v.get("bun"), "mg/dL", "bun")
        if bun > 19:
            score += 1
        rr = _parse_num(v.get("respiratory_rate"))
        if rr >= 30:
            score += 1
        sbp = _convert(v.get("systolic_bp"), "mmHg", "pressure")
        dbp = _convert(v.get("diastolic_bp"), "mmHg", "pressure")
        if sbp < 90 or dbp <= 60:
            score += 1
        return _ok(score, {"age": str(age), "BUN": str(bun), "RR": str(rr), "SBP": str(sbp), "DBP": str(dbp)}, ["CURB-65 scoring"])
    except Exception as e:
        return _err(str(e))


# 37. Centor Score ────────────────────────────────────────────────────────────
def run_centor_score(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        age = _parse_num(v.get("age"))
        if 3 <= age <= 14:
            score += 1
        elif age >= 45:
            score -= 1
        temp = _convert(v.get("temperature"), "°C", "temperature")
        if temp > 38:
            score += 1
        if _parse_bool(v.get("cough_absent", False)):
            score += 1
        if _parse_bool(v.get("tender_lymph_nodes", False)):
            score += 1
        if _parse_bool(v.get("exudate_swelling_tonsils", False)):
            score += 1
        return _ok(score, {"age": str(age), "temp": str(temp)}, ["Modified Centor/McIsaac"])
    except Exception as e:
        return _err(str(e))


# 38. FeverPAIN ───────────────────────────────────────────────────────────────
def run_feverpain(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        if _parse_bool(v.get("symptom_onset", False)):
            score += 1
        if _parse_bool(v.get("purulent_tonsils", False)):
            score += 1
        if _parse_bool(v.get("fever_24_hours", False)):
            score += 1
        if _parse_bool(v.get("severe_tonsil_inflammation", False)):
            score += 1
        # cough_coryza_absent defaults to True if not provided
        cca = v.get("cough_coryza_absent")
        if cca is None or _parse_bool(cca):
            score += 1
        return _ok(score, {}, ["FeverPAIN scoring"])
    except Exception as e:
        return _err(str(e))


# 39. Glasgow Coma Score ──────────────────────────────────────────────────────
_GCS_EYE = {
    "no eye opening": 1, "eye opening to pain": 2,
    "eye opening to verbal command": 3, "eyes open spontaneously": 4,
    "not testable": 1,
}
_GCS_VERBAL = {
    "no verbal response": 1, "incomprehensible sounds": 2,
    "inappropriate words": 3, "confused": 4, "oriented": 5,
    "not testable": 1,
}
_GCS_MOTOR = {
    "no motor response": 1, "extension to pain": 2,
    "flexion to pain": 3, "withdrawal from pain": 4,
    "localizes pain": 5, "obeys commands": 6,
}


def _gcs_lookup(raw: Any, table: Dict[str, int], default: int) -> int:
    """Resolve a GCS component: try categorical string lookup, fall back to numeric."""
    s = _parse_str(raw).strip().lower()
    if s in table:
        return table[s]
    # Fall back to numeric
    return int(_parse_num(raw, default))


def run_glasgow_coma_score(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        eye = _gcs_lookup(v.get("gcs_eye", 4), _GCS_EYE, 4)
        verbal = _gcs_lookup(v.get("gcs_verbal", 5), _GCS_VERBAL, 5)
        motor = _gcs_lookup(v.get("gcs_motor", 6), _GCS_MOTOR, 6)
        result = eye + verbal + motor
        return _ok(result, {"eye": str(eye), "verbal": str(verbal), "motor": str(motor)}, ["GCS = E + V + M"])
    except Exception as e:
        return _err(str(e))


# 40. Child-Pugh ──────────────────────────────────────────────────────────────
_ASCITES_SCORE = {"absent": 1, "slight": 2, "moderate": 3}
_ENCEPH_SCORE = {"no encephalopathy": 1, "grade 1-2": 2, "grade 3-4": 3}


def run_child_pugh(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        inr = _parse_num(v.get("inr"))
        if inr < 1.7:
            score += 1
        elif inr <= 2.3:
            score += 2
        else:
            score += 3

        bili = _convert(v.get("serum_bilirubin"), "mg/dL", "bilirubin")
        if bili < 2:
            score += 1
        elif bili <= 3:
            score += 2
        else:
            score += 3

        alb = _convert(v.get("serum_albumin"), "g/dL", "albumin")
        if alb > 3.5:
            score += 1
        elif alb >= 2.8:
            score += 2
        else:
            score += 3

        ascites_str = _parse_str(v.get("ascites", "")).lower()
        score += _ASCITES_SCORE.get(ascites_str, 1)

        enceph_str = _parse_str(v.get("encephalopathy", "")).lower()
        score += _ENCEPH_SCORE.get(enceph_str, 1)

        return _ok(score, {"INR": str(inr), "bilirubin": str(bili), "albumin": str(alb)}, ["Child-Pugh scoring"])
    except Exception as e:
        return _err(str(e))


# 41. HAS-BLED ────────────────────────────────────────────────────────────────
def run_has_bled(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        age = _parse_num(v.get("age", 0))
        if age > 65:
            score += 1
        if _parse_bool(v.get("hypertension", False)):
            score += 1
        if _parse_bool(v.get("liver_disease_has_bled", False)):
            score += 1
        if _parse_bool(v.get("renal_disease_has_bled", False)):
            score += 1
        if _parse_bool(v.get("stroke", False)):
            score += 1
        if _parse_bool(v.get("prior_bleeding", False)):
            score += 1
        if _parse_bool(v.get("labile_inr", False)):
            score += 1
        if _parse_bool(v.get("medications_for_bleeding", False)):
            score += 1
        # Alcohol: boolean in schema
        if _parse_bool(v.get("alcoholic_drinks", False)):
            score += 1
        return _ok(score, {"age": str(age)}, ["HAS-BLED scoring"])
    except Exception as e:
        return _err(str(e))


# 42. HEART Score ─────────────────────────────────────────────────────────────
def run_heart_score(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        # History
        history = _parse_str(v.get("history", "")).lower()
        if "highly" in history:
            score += 2
        elif "moderately" in history:
            score += 1
        # else slightly suspicious or unknown = 0

        # EKG
        ekg = _parse_str(v.get("electrocardiogram", "")).lower()
        if "significant" in ekg or "st" in ekg:
            score += 2
        elif "non-specific" in ekg or "nonspecific" in ekg or "non specific" in ekg:
            score += 1

        # Age
        age = _parse_num(v.get("age", 0))
        if age >= 65:
            score += 2
        elif age >= 45:
            score += 1

        # Risk factors
        risk_count = 0
        has_athero = _parse_bool(v.get("atherosclerotic_disease", False))
        for key in ["hypertension", "hypercholesterolemia", "diabetes_mellitus", "obesity", "smoking", "family_with_cvd"]:
            if _parse_bool(v.get(key, False)):
                risk_count += 1
        if _parse_bool(v.get("tia", False)):
            risk_count += 1

        if has_athero or risk_count >= 3:
            score += 2
        elif risk_count >= 1:
            score += 1

        # Troponin
        trop = _parse_str(v.get("initial_troponin", "")).lower()
        if ">3" in trop or "3x" in trop or "high" in trop:
            score += 2
        elif "1-3" in trop or "2x" in trop or "slightly" in trop:
            score += 1

        return _ok(score, {"age": str(age)}, ["HEART score"])
    except Exception as e:
        return _err(str(e))


# 43. Revised Cardiac Risk Index ──────────────────────────────────────────────
def run_rcri(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        if _parse_bool(v.get("elevated_risk_surgery", False)):
            score += 1
        if _parse_bool(v.get("ischemetic_heart_disease", False)):
            score += 1
        if _parse_bool(v.get("congestive_heart_failure", False)):
            score += 1
        if _parse_bool(v.get("cerebrovascular_disease", False)):
            score += 1
        if _parse_bool(v.get("pre_operative_insulin_treatment", False)):
            score += 1
        cr = _convert(v.get("pre_operative_creatinine", 0), "mg/dL", "creatinine")
        if cr > 2:
            score += 1
        return _ok(score, {"creatinine": str(cr)}, ["RCRI scoring"])
    except Exception as e:
        return _err(str(e))


# 44. Charlson Comorbidity Index ──────────────────────────────────────────────
def run_cci(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        age = _parse_num(v.get("age", 0))
        if 50 <= age < 60:
            score += 1
        elif 60 <= age < 70:
            score += 2
        elif 70 <= age < 80:
            score += 3
        elif age >= 80:
            score += 4

        # 1-point items
        for key in ["mi", "chf", "peripheral_vascular_disease", "connective_tissue_disease",
                     "dementia", "copd", "peptic_ucler_disease"]:
            if _parse_bool(v.get(key, False)):
                score += 1

        # CVA/TIA combined = 1 point
        if _parse_bool(v.get("cva", False)) or _parse_bool(v.get("tia", False)):
            score += 1

        # 2-point items
        for key in ["hemiplegia", "moderate_to_severe_ckd", "leukemia", "lymphoma"]:
            if _parse_bool(v.get(key, False)):
                score += 2

        # Diabetes: categorical lookup
        _CCI_DIABETES = {"none or diet-controlled": 0, "uncomplicated": 1, "end-organ damage": 2}
        diabetes_str = _parse_str(v.get("diabetes_mellitus", "")).lower()
        score += _CCI_DIABETES.get(diabetes_str, 0)

        # Liver disease: categorical lookup
        _CCI_LIVER = {"none": 0, "mild": 1, "moderate to severe": 3}
        liver_str = _parse_str(v.get("liver_disease", "")).lower()
        score += _CCI_LIVER.get(liver_str, 0)

        # Solid tumor: categorical lookup
        _CCI_TUMOR = {"none": 0, "localized": 2, "metastatic": 6}
        tumor_str = _parse_str(v.get("solid_tumor", "")).lower()
        score += _CCI_TUMOR.get(tumor_str, 0)

        # AIDS
        if _parse_bool(v.get("aids", False)):
            score += 6

        return _ok(score, {"age": str(age)}, ["CCI scoring"])
    except Exception as e:
        return _err(str(e))


# 45. Framingham Risk Score ───────────────────────────────────────────────────
def run_framingham_chd(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        age = _parse_num(v.get("age"))
        sex = _parse_str(v.get("sex")).lower()
        tc = _convert(v.get("total_cholesterol"), "mg/dL", "cholesterol")
        hdl = _convert(v.get("hdl_cholesterol"), "mg/dL", "hdl")
        sbp = _convert(v.get("systolic_bp"), "mmHg", "pressure")
        smoker = 1 if _parse_bool(v.get("smoker", False)) else 0
        bp_med = 1 if _parse_bool(v.get("bp_medicine", False)) else 0
        is_female = "female" in sex or sex == "f"

        ln = math.log
        if not is_female:
            # Male
            age_smoke = min(age, 70)
            risk_score = (52.00961 * ln(age) + 20.014077 * ln(tc) + -0.905964 * ln(hdl)
                          + 1.305784 * ln(sbp) + 0.241549 * bp_med + 12.096316 * smoker
                          + -4.605038 * ln(age) * ln(tc)
                          + -2.84367 * ln(age_smoke) * smoker
                          + -2.93323 * ln(age) * ln(age) - 172.300168)
            ten_yr = (1 - 0.9402 ** math.exp(risk_score)) * 100
        else:
            # Female
            age_smoke = min(age, 78)
            risk_score = (31.764001 * ln(age) + 22.465206 * ln(tc) + -1.187731 * ln(hdl)
                          + 2.552905 * ln(sbp) + 0.420251 * bp_med + 13.07543 * smoker
                          + -5.060998 * ln(age) * ln(tc)
                          + -2.996945 * ln(age_smoke) * smoker
                          - 146.5933061)
            ten_yr = (1 - 0.98767 ** math.exp(risk_score)) * 100

        result = round(ten_yr, 3)
        return _ok(result, {"age": str(age), "sex": sex, "TC": str(tc), "HDL": str(hdl), "SBP": str(sbp)}, ["Framingham CHD risk score"])
    except Exception as e:
        return _err(str(e))


# 46. PSI/PORT Score ──────────────────────────────────────────────────────────
def run_psi_cap(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        age = _parse_num(v.get("age"))
        sex = _parse_str(v.get("sex")).lower()
        is_female = "female" in sex or sex == "f"
        score = int(age)
        if is_female:
            score -= 10
        if _parse_bool(v.get("nursing_home_resident", False)):
            score += 10
        if _parse_bool(v.get("neoplastic_disease", False)):
            score += 30
        if _parse_bool(v.get("liver_disease", False)):
            score += 20
        if _parse_bool(v.get("chf", False)):
            score += 10
        if _parse_bool(v.get("cerebrovascular_disease", False)):
            score += 10
        if _parse_bool(v.get("renal_disease", False)):
            score += 10
        if _parse_bool(v.get("altered_mental_status", False)):
            score += 20
        rr = _parse_num(v.get("respiratory_rate", 0))
        if rr >= 30:
            score += 20
        sbp = _convert(v.get("systolic_bp", 120), "mmHg", "pressure")
        if sbp < 90:
            score += 20
        temp = _convert(v.get("temperature", 37), "°C", "temperature")
        if temp < 35 or temp >= 40:
            score += 15
        hr = _parse_num(v.get("heart_rate", 80))
        if hr >= 125:
            score += 10
        ph = _parse_num(v.get("ph", 7.4))
        if ph < 7.35:
            score += 30
        bun = _convert(v.get("bun", 10), "mg/dL", "bun")
        if bun >= 30:
            score += 20
        na = _convert(v.get("serum_sodium", 140), "mEq/L", "electrolyte")
        if na < 130:
            score += 20
        glu = _convert(v.get("serum_glucose", 100), "mg/dL", "glucose")
        if glu >= 250:
            score += 10
        hct = _parse_num(v.get("hematocrit", 40))
        if hct < 30:
            score += 10
        po2 = _convert(v.get("partial_pressure_o2", 80), "mmHg", "pressure")
        if po2 < 60:
            score += 10
        if _parse_bool(v.get("pleural_effusion", False)):
            score += 10
        return _ok(score, {"age": str(age), "sex": sex}, ["PSI/PORT scoring"])
    except Exception as e:
        return _err(str(e))


# 47. APACHE II ───────────────────────────────────────────────────────────────
def run_apache2(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        age = _parse_num(v.get("age"))
        if age < 45:
            pass
        elif age <= 54:
            score += 2
        elif age <= 64:
            score += 3
        elif age <= 74:
            score += 5
        else:
            score += 6

        # Organ failure / immunocompromise
        if _parse_bool(v.get("organ_failure_or_immunocompromise", False)):
            surg = _parse_str(v.get("surgery_type", "")).lower()
            if surg in ("nonoperative", "emergency"):
                score += 5
            elif surg == "elective":
                score += 2

        # Temperature
        temp = _convert(v.get("temperature"), "°C", "temperature")
        if temp >= 41:
            score += 4
        elif temp >= 39:
            score += 3
        elif temp >= 38.5:
            score += 1
        elif temp >= 36:
            pass
        elif temp >= 34:
            score += 1
        elif temp >= 32:
            score += 2
        elif temp >= 30:
            score += 3
        else:
            score += 4

        # MAP
        sbp = _convert(v.get("systolic_bp"), "mmHg", "pressure")
        dbp = _convert(v.get("diastolic_bp"), "mmHg", "pressure")
        map_val = _parse_num(v.get("mean_arterial_pressure", 0))
        if map_val == 0 and sbp > 0:
            map_val = (1 / 3) * sbp + (2 / 3) * dbp
        if map_val > 159:
            score += 4
        elif map_val > 129:
            score += 3
        elif map_val > 109:
            score += 2
        elif map_val >= 70:
            pass
        elif map_val >= 50:
            score += 2
        else:
            score += 4

        # Heart rate
        hr = _parse_num(v.get("heart_rate"))
        if hr >= 180:
            score += 4
        elif hr >= 140:
            score += 3
        elif hr >= 110:
            score += 2
        elif hr >= 70:
            pass
        elif hr >= 55:
            score += 2
        elif hr >= 40:
            score += 3
        else:
            score += 4

        # Respiratory rate
        rr = _parse_num(v.get("respiratory_rate"))
        if rr >= 50:
            score += 4
        elif rr >= 35:
            score += 3
        elif rr >= 25:
            score += 1
        elif rr >= 12:
            pass
        elif rr >= 10:
            score += 1
        elif rr >= 6:
            score += 2
        else:
            score += 4

        # Oxygenation
        fio2 = _convert(v.get("fio2", 21), "%", "fio2")
        if fio2 >= 50:
            aa = _convert(v.get("aa_gradient", 0), "mmHg", "pressure")
            if aa > 499:
                score += 4
            elif aa >= 350:
                score += 3
            elif aa >= 200:
                score += 2
        else:
            pao2 = _convert(v.get("pao2", 80), "mmHg", "pressure")
            if pao2 > 70:
                pass
            elif pao2 >= 61:
                score += 1
            elif pao2 >= 55:
                score += 3
            else:
                score += 4

        # pH
        ph = _parse_num(v.get("ph"))
        if ph >= 7.70:
            score += 4
        elif ph >= 7.60:
            score += 3
        elif ph >= 7.50:
            score += 1
        elif ph >= 7.33:
            pass
        elif ph >= 7.25:
            score += 2
        elif ph >= 7.15:
            score += 3
        else:
            score += 4

        # Sodium (mEq/L = mmol/L for monovalent)
        na = _convert(v.get("serum_sodium"), "mEq/L", "electrolyte")
        if na >= 180:
            score += 4
        elif na >= 160:
            score += 3
        elif na >= 155:
            score += 2
        elif na >= 150:
            score += 1
        elif na >= 130:
            pass
        elif na >= 120:
            score += 2
        elif na >= 111:
            score += 3
        else:
            score += 4

        # Potassium
        k = _convert(v.get("serum_potassium"), "mEq/L", "electrolyte")
        if k >= 7.0:
            score += 4
        elif k >= 6.0:
            score += 3
        elif k >= 5.5:
            score += 1
        elif k >= 3.5:
            pass
        elif k >= 3.0:
            score += 1
        elif k >= 2.5:
            score += 2
        else:
            score += 4

        # Creatinine
        cr = _convert(v.get("serum_creatinine"), "mg/dL", "creatinine")
        arf = _parse_bool(v.get("acute_renal_failure", False))
        crf = _parse_bool(v.get("chronic_renal_failure", False))

        if arf:
            if cr >= 3.5:
                score += 8
            elif cr >= 2.0:
                score += 6
            elif cr >= 1.5:
                score += 4
            elif cr >= 0.6:
                pass
            else:
                score += 2
        elif crf:
            if cr >= 3.5:
                score += 4
            elif cr >= 2.0:
                score += 3
            elif cr >= 1.5:
                score += 2
            elif cr >= 0.6:
                pass
            else:
                score += 2
        else:
            if cr >= 3.5:
                score += 4
            elif cr >= 2.0:
                score += 3
            elif cr >= 1.5:
                score += 2
            elif cr >= 0.6:
                pass
            else:
                score += 2

        # Hematocrit
        hct = _parse_num(v.get("hematocrit"))
        if hct >= 60:
            score += 4
        elif hct >= 50:
            score += 2
        elif hct >= 46:
            score += 1
        elif hct >= 30:
            pass
        elif hct >= 20:
            score += 2
        else:
            score += 4

        # WBC (in 10^9/L)
        wbc = _convert(v.get("wbc"), "10^9/L", "wbc")
        if wbc >= 40:
            score += 4
        elif wbc >= 20:
            score += 2
        elif wbc >= 15:
            score += 1
        elif wbc >= 3:
            pass
        elif wbc >= 1:
            score += 2
        else:
            score += 4

        # GCS
        gcs = int(_parse_num(v.get("glasgow_coma_score", 15)))
        score += (15 - gcs)

        return _ok(score, {"age": str(age)}, ["APACHE II scoring"])
    except Exception as e:
        return _err(str(e))


# 48. SOFA Score ──────────────────────────────────────────────────────────────
def run_sofa(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        pao2 = _convert(v.get("pao2"), "mmHg", "pressure")
        fio2 = _convert(v.get("fio2"), "%", "fio2")
        if fio2 > 0:
            ratio = pao2 / (fio2 / 100)
        else:
            ratio = 999

        mech_vent = _parse_bool(v.get("on_mechanical_ventilation", False))
        cpap = _parse_bool(v.get("cpap", False))
        vent = mech_vent or cpap

        if ratio >= 400:
            pass
        elif ratio >= 300:
            score += 1
        elif ratio >= 200:
            score += 2
        elif ratio < 200 and not vent:
            score += 2
        elif ratio >= 100 and vent:
            score += 3
        elif ratio < 100 and vent:
            score += 4

        # Cardiovascular
        dopa = _parse_num(v.get("dopamine", 0))
        dobu = _parse_num(v.get("dobutamine", 0))
        epi = _parse_num(v.get("epinephrine", 0))
        norepi = _parse_num(v.get("norepinephrine", 0))

        if dopa > 15 or epi > 0.1 or norepi > 0.1:
            score += 4
        elif dopa > 5 or (0 < epi <= 0.1) or (0 < norepi <= 0.1):
            score += 3
        elif (0 < dopa <= 5) or dobu > 0:
            score += 2
        else:
            # Check hypotension
            sbp = _convert(v.get("systolic_bp", 120), "mmHg", "pressure")
            dbp = _convert(v.get("diastolic_bp", 80), "mmHg", "pressure")
            map_val = (1 / 3) * sbp + (2 / 3) * dbp
            hypotension = _parse_bool(v.get("hypotension", False))
            if map_val < 70 or hypotension:
                score += 1

        # GCS
        gcs = int(_parse_num(v.get("glasgow_coma_score", 15)))
        if gcs < 6:
            score += 4
        elif gcs <= 9:
            score += 3
        elif gcs <= 12:
            score += 2
        elif gcs <= 14:
            score += 1

        # Bilirubin
        bili = _convert(v.get("serum_bilirubin", 0), "mg/dL", "bilirubin")
        if bili >= 12:
            score += 4
        elif bili >= 6:
            score += 3
        elif bili >= 2:
            score += 2
        elif bili >= 1.2:
            score += 1

        # Platelets (10^9/L)
        plt = _convert(v.get("platelet_count", 150), "10^9/L", "platelets")
        if plt < 20:
            score += 4
        elif plt < 50:
            score += 3
        elif plt < 100:
            score += 2
        elif plt < 150:
            score += 1

        # Creatinine / urine output
        cr = _convert(v.get("serum_creatinine", 0), "mg/dL", "creatinine")
        uo = _parse_num(v.get("urine_output", 1000))
        if cr > 5.0 or uo < 200:
            score += 4
        elif cr >= 3.5 or uo < 500:
            score += 3
        elif cr >= 2.0:
            score += 2
        elif cr >= 1.2:
            score += 1

        return _ok(score, {"PaO2/FiO2": str(round(ratio, 1))}, ["SOFA scoring"])
    except Exception as e:
        return _err(str(e))


# 49. Glasgow-Blatchford ──────────────────────────────────────────────────────
def run_glasgow_blatchford(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        sex = _parse_str(v.get("sex")).lower()
        is_female = "female" in sex or sex == "f"

        bun = _convert(v.get("bun"), "mg/dL", "bun")
        if bun >= 70:
            score += 6
        elif bun >= 28:
            score += 4
        elif bun >= 22.4:
            score += 3
        elif bun >= 18.2:
            score += 2

        hgb = _convert(v.get("hemoglobin"), "g/dL", "hemoglobin")
        if is_female:
            if hgb < 10:
                score += 6
            elif hgb < 12:
                score += 1
        else:
            if hgb < 10:
                score += 6
            elif hgb < 12:
                score += 3
            elif hgb < 13:
                score += 1

        sbp = _convert(v.get("systolic_bp"), "mmHg", "pressure")
        if sbp < 90:
            score += 3
        elif sbp < 100:
            score += 2
        elif sbp < 110:
            score += 1

        hr = _parse_num(v.get("heart_rate", 0))
        if hr >= 100:
            score += 1

        if _parse_bool(v.get("melena_present", False)):
            score += 1
        if _parse_bool(v.get("syncope", False)):
            score += 2
        if _parse_bool(v.get("hepatic_disease_history", False)):
            score += 2
        if _parse_bool(v.get("cardiac_failure", False)):
            score += 2

        return _ok(score, {"sex": sex, "BUN": str(bun), "Hgb": str(hgb), "SBP": str(sbp)}, ["Glasgow-Blatchford scoring"])
    except Exception as e:
        return _err(str(e))


# 50. Caprini VTE ─────────────────────────────────────────────────────────────
def run_caprini_vte(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = 0
        age = _parse_num(v.get("age"))
        if age <= 40:
            pass
        elif age <= 60:
            score += 1
        elif age <= 74:
            score += 2
        else:
            score += 3

        # Surgery type
        surg = _parse_str(v.get("surgery_type", "none")).lower()
        surgery_pts = {"none": 0, "minor": 1, "major": 2, "laparoscopic": 2,
                       "arthroscopic": 2, "elective major lower extremity arthroplasty": 5}
        score += surgery_pts.get(surg, 0)

        # Mobility
        mob = _parse_str(v.get("mobility", "normal")).lower()
        if "bed rest" in mob:
            score += 1
        elif "confined" in mob or ">72" in mob:
            score += 2

        # BMI
        bmi_val = _parse_num(v.get("bmi", 0))
        if bmi_val > 25:
            score += 1

        # 1-point boolean items
        one_pt = ["major_surgery", "chf", "sepsis", "pneumonia", "varicose_veins",
                   "current_swollen_legs", "inflammatory_bowel_disease",
                   "acute_myocardial_infarction", "copd"]
        for key in one_pt:
            if _parse_bool(v.get(key, False)):
                score += 1

        # 2-point items
        two_pt = ["immobilizing_plaster_case", "current_central_venuous", "malignancy"]
        for key in two_pt:
            if _parse_bool(v.get(key, False)):
                score += 2

        # 3-point items
        three_pt = ["previous_dvt", "previous_pe", "family_history_thrombosis",
                     "positive_factor_v", "positive_prothrombin", "serum_homocysteine",
                     "positive_lupus_anticoagulant", "elevated_anticardiolipin_antibody",
                     "heparin_induced_thrombocytopenia", "congenital_acquired_thrombophilia"]
        for key in three_pt:
            if _parse_bool(v.get(key, False)):
                score += 3

        # 5-point items
        five_pt = ["hip_pelvis_leg_fracture", "stroke", "multiple_trauma",
                    "acute_spinal_chord_injury"]
        for key in five_pt:
            if _parse_bool(v.get(key, False)):
                score += 5

        return _ok(score, {"age": str(age)}, ["Caprini VTE scoring"])
    except Exception as e:
        return _err(str(e))


# 51. Estimated Due Date ──────────────────────────────────────────────────────
def _parse_date(raw: Any) -> Optional[datetime]:
    s = _parse_str(raw)
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def run_estimated_due_date(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        lmp = _parse_date(v.get("last_menstrual_date"))
        if lmp is None:
            return _err("Cannot parse last_menstrual_date")
        cycle = int(_parse_num(v.get("cycle_length", 28)))
        due = lmp + timedelta(weeks=40) + timedelta(days=(cycle - 28))
        result = due.strftime("%m/%d/%Y")
        return _ok(result, {"LMP": lmp.strftime("%m/%d/%Y"), "cycle": str(cycle)}, ["Naegele's rule"])
    except Exception as e:
        return _err(str(e))


# 52. Estimated Conception Date ────────────────────────────────────────────────
def run_estimated_conception_date(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        lmp = _parse_date(v.get("last_menstrual_date"))
        if lmp is None:
            return _err("Cannot parse last_menstrual_date")
        conception = lmp + timedelta(weeks=2)
        result = conception.strftime("%m/%d/%Y")
        return _ok(result, {"LMP": lmp.strftime("%m/%d/%Y")}, ["LMP + 2 weeks"])
    except Exception as e:
        return _err(str(e))


# 53. Estimated Gestational Age ────────────────────────────────────────────────
def run_estimated_gestational_age(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        current = _parse_date(v.get("current_date"))
        lmp = _parse_date(v.get("last_menstrual_date"))
        if current is None or lmp is None:
            return _err("Cannot parse dates")
        delta = abs((current - lmp).days)
        weeks = delta // 7
        days = delta % 7
        result = f"{weeks} weeks and {days} days"
        return _ok(result, {"current": current.strftime("%m/%d/%Y"), "LMP": lmp.strftime("%m/%d/%Y")}, ["Gestational age from LMP"])
    except Exception as e:
        return _err(str(e))


# 54. Steroid Conversion ──────────────────────────────────────────────────────
STEROID_EQUIV = {
    "betamethasone iv": 1,
    "cortisone po": 33.33,
    "dexamethasone iv": 1,
    "dexamethasone po": 1,
    "hydrocortisone iv": 26.67,
    "hydrocortisone po": 26.67,
    "methylprednisolone iv": 5.33,
    "methylprednisolone po": 5.33,
    "prednisolone po": 6.67,
    "prednisone po": 6.67,
    "triamcinolone iv": 5.33,
}


def run_steroid_conversion(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_raw = _parse_str(v.get("input_steroid", ""))
        target_raw = _parse_str(v.get("target_steroid", ""))

        # Parse input: try to extract drug name, dose, route
        # Format might be "Prednisone PO 40 mg" or similar
        input_lower = input_raw.lower().strip()
        target_lower = target_raw.lower().strip()

        # Find source steroid and dose
        src_key = None
        src_dose = None
        for key in STEROID_EQUIV:
            if key in input_lower:
                src_key = key
                break
        if src_key is None:
            # Try partial match
            for key in STEROID_EQUIV:
                drug_name = key.split()[0]
                if drug_name in input_lower:
                    src_key = key
                    break

        # Extract dose
        import re
        dose_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mg|mcg)", input_lower)
        if dose_match:
            src_dose = float(dose_match.group(1))
        else:
            nums = re.findall(r"\d+(?:\.\d+)?", input_lower)
            if nums:
                src_dose = float(nums[0])

        # Find target steroid
        tgt_key = None
        for key in STEROID_EQUIV:
            if key in target_lower:
                tgt_key = key
                break
        if tgt_key is None:
            for key in STEROID_EQUIV:
                drug_name = key.split()[0]
                if drug_name in target_lower:
                    tgt_key = key
                    break

        if src_key is None or tgt_key is None or src_dose is None:
            return _err(f"Could not parse steroids: src={input_raw}, tgt={target_raw}")

        result = src_dose * (STEROID_EQUIV[tgt_key] / STEROID_EQUIV[src_key])
        return _ok(round(result, 5), {"input": input_raw, "target": target_raw}, ["Steroid equivalency conversion"])
    except Exception as e:
        return _err(str(e))


# 55. Morphine Milligram Equivalents ──────────────────────────────────────────
MME_FACTORS = {
    "codeine": 0.15,
    "fentanyl buccal": 0.13,
    "fentanyl patch": 2.4,
    "hydrocodone": 1,
    "hydromorphone": 5,
    "methadone": 4.7,
    "morphine": 1,
    "oxycodone": 1.5,
    "oxymorphone": 3,
    "tapentadol": 0.4,
    "tramadol": 0.2,
    "buprenorphine": 10,
}

# Also map with tallman lettering
MME_ALIASES = {
    "fentanyl buccal": 0.13,
    "fentanyl patch": 2.4,
}


def run_mme(v: Dict[str, Any]) -> Dict[str, Any]:
    try:
        meds = v.get("medications", [])
        if not isinstance(meds, list):
            return _err("medications must be a list")

        total_mme = 0.0
        details = []
        for med in meds:
            if isinstance(med, dict):
                drug = med.get("drug", "").lower()
                dose = float(med.get("dose", 0))
                dpd = float(med.get("doses_per_day", 1))
                # Find conversion factor
                factor = None
                for key, f in MME_FACTORS.items():
                    if key in drug:
                        factor = f
                        break
                if factor is None:
                    continue
                mme = dose * dpd * factor
                total_mme += mme
                details.append(f"{drug}: {dose}*{dpd}*{factor}={mme}")

        return _ok(round(total_mme, 5), {"medications": str(len(meds))}, details or ["No opioid medications found"])
    except Exception as e:
        return _err(str(e))


# ── Import JSON schemas ─────────────────────────────────────────────────────

import json
from pathlib import Path

_SCHEMA_PATH = Path(__file__).with_name("all_calc_schemas.json")
_SCHEMAS: Dict[str, Any] = {}
if _SCHEMA_PATH.exists():
    try:
        _SCHEMAS = json.loads(_SCHEMA_PATH.read_text())
    except Exception:
        pass


def _make_calc_entry(calc_id: str, run_fn, fallback_title: str = "", fallback_desc: str = "", fallback_tags: Optional[List[str]] = None):
    schema = _SCHEMAS.get(calc_id, {})
    return {
        "def": CalculatorDef(
            id=calc_id,
            title=schema.get("title", fallback_title),
            description=schema.get("description", fallback_desc),
            version=schema.get("version", "1.0"),
            tags=schema.get("tags", fallback_tags or []),
            inputs=None,
        ),
        "run": run_fn,
        "input_model": None,
        "schema": schema,
    }


# ── Calculator Registry ─────────────────────────────────────────────────────

CALCULATORS: Dict[str, Dict[str, Any]] = {
    "mean_arterial_pressure": _make_calc_entry("mean_arterial_pressure", run_mean_arterial_pressure, "Mean Arterial Pressure (MAP)"),
    "bmi": _make_calc_entry("bmi", run_bmi, "Body Mass Index (BMI)"),
    "bsa": _make_calc_entry("bsa", run_bsa, "Body Surface Area Calculator"),
    "ideal_body_weight": _make_calc_entry("ideal_body_weight", run_ideal_body_weight, "Ideal Body Weight"),
    "adjusted_body_weight": _make_calc_entry("adjusted_body_weight", run_adjusted_body_weight, "Adjusted Body Weight"),
    "target_weight": _make_calc_entry("target_weight", run_target_weight, "Target Weight"),
    "calcium_correction": _make_calc_entry("calcium_correction", run_calcium_correction, "Calcium Correction"),
    "anion_gap": _make_calc_entry("anion_gap", run_anion_gap, "Anion Gap"),
    "delta_gap": _make_calc_entry("delta_gap", run_delta_gap, "Delta Gap"),
    "delta_ratio": _make_calc_entry("delta_ratio", run_delta_ratio, "Delta Ratio"),
    "albumin_corrected_anion_gap": _make_calc_entry("albumin_corrected_anion_gap", run_albumin_corrected_anion_gap, "Albumin Corrected Anion Gap"),
    "albumin_corrected_delta_gap": _make_calc_entry("albumin_corrected_delta_gap", run_albumin_corrected_delta_gap, "Albumin Corrected Delta Gap"),
    "albumin_corrected_delta_ratio": _make_calc_entry("albumin_corrected_delta_ratio", run_albumin_corrected_delta_ratio, "Albumin Corrected Delta Ratio"),
    "ldl_calculated": _make_calc_entry("ldl_calculated", run_ldl_calculated, "LDL Calculated"),
    "sodium_correction_hyperglycemia": _make_calc_entry("sodium_correction_hyperglycemia", run_sodium_correction_hyperglycemia, "Sodium Correction for Hyperglycemia"),
    "serum_osmolality": _make_calc_entry("serum_osmolality", run_serum_osmolality, "Serum Osmolality"),
    "homa_ir": _make_calc_entry("homa_ir", run_homa_ir, "HOMA-IR"),
    "fena": _make_calc_entry("fena", run_fena, "Fractional Excretion of Sodium (FENa)"),
    "fib4": _make_calc_entry("fib4", run_fib4, "Fibrosis-4 (FIB-4) Index"),
    "maintenance_fluids": _make_calc_entry("maintenance_fluids", run_maintenance_fluids, "Maintenance Fluids"),
    "free_water_deficit": _make_calc_entry("free_water_deficit", run_free_water_deficit, "Free Water Deficit"),
    "qtc_bazett": _make_calc_entry("qtc_bazett", run_qtc_bazett, "QTc Bazett Calculator"),
    "qtc_fridericia": _make_calc_entry("qtc_fridericia", run_qtc_fridericia, "QTc Fridericia Calculator"),
    "qtc_framingham": _make_calc_entry("qtc_framingham", run_qtc_framingham, "QTc Framingham Calculator"),
    "qtc_hodges": _make_calc_entry("qtc_hodges", run_qtc_hodges, "QTc Hodges Calculator"),
    "qtc_rautaharju": _make_calc_entry("qtc_rautaharju", run_qtc_rautaharju, "QTc Rautaharju Calculator"),
    "creatinine_clearance": _make_calc_entry("creatinine_clearance", run_creatinine_clearance, "Creatinine Clearance (Cockcroft-Gault)"),
    "ckd_epi_gfr": _make_calc_entry("ckd_epi_gfr", run_ckd_epi_gfr, "CKD-EPI GFR (2021)"),
    "mdrd_gfr": _make_calc_entry("mdrd_gfr", run_mdrd_gfr, "MDRD GFR"),
    "meld_na": _make_calc_entry("meld_na", run_meld_na, "MELD-Na (UNOS/OPTN)"),
    "cha2ds2_vasc": _make_calc_entry("cha2ds2_vasc", run_cha2ds2_vasc, "CHA2DS2-VASc Score"),
    "wells_dvt": _make_calc_entry("wells_dvt", run_wells_dvt, "Wells' Criteria for DVT"),
    "wells_pe": _make_calc_entry("wells_pe", run_wells_pe, "Wells' Criteria for PE"),
    "perc_rule": _make_calc_entry("perc_rule", run_perc_rule, "PERC Rule"),
    "sirs": _make_calc_entry("sirs", run_sirs, "SIRS Criteria"),
    "curb65": _make_calc_entry("curb65", run_curb65, "CURB-65 Score"),
    "centor_score": _make_calc_entry("centor_score", run_centor_score, "Centor Score"),
    "feverpain": _make_calc_entry("feverpain", run_feverpain, "FeverPAIN Score"),
    "glasgow_coma_score": _make_calc_entry("glasgow_coma_score", run_glasgow_coma_score, "Glasgow Coma Score (GCS)"),
    "child_pugh": _make_calc_entry("child_pugh", run_child_pugh, "Child-Pugh Score"),
    "has_bled": _make_calc_entry("has_bled", run_has_bled, "HAS-BLED Score"),
    "heart_score": _make_calc_entry("heart_score", run_heart_score, "HEART Score"),
    "rcri": _make_calc_entry("rcri", run_rcri, "Revised Cardiac Risk Index"),
    "cci": _make_calc_entry("cci", run_cci, "Charlson Comorbidity Index"),
    "framingham_chd": _make_calc_entry("framingham_chd", run_framingham_chd, "Framingham Risk Score"),
    "psi_cap": _make_calc_entry("psi_cap", run_psi_cap, "PSI Score: Pneumonia Severity Index"),
    "apache2": _make_calc_entry("apache2", run_apache2, "APACHE II Score"),
    "sofa": _make_calc_entry("sofa", run_sofa, "SOFA Score"),
    "glasgow_blatchford": _make_calc_entry("glasgow_blatchford", run_glasgow_blatchford, "Glasgow-Blatchford Bleeding Score"),
    "caprini_vte": _make_calc_entry("caprini_vte", run_caprini_vte, "Caprini Score"),
    "estimated_due_date": _make_calc_entry("estimated_due_date", run_estimated_due_date, "Estimated Due Date"),
    "estimated_conception_date": _make_calc_entry("estimated_conception_date", run_estimated_conception_date, "Estimated Date of Conception"),
    "estimated_gestational_age": _make_calc_entry("estimated_gestational_age", run_estimated_gestational_age, "Estimated Gestational Age"),
    "steroid_conversion": _make_calc_entry("steroid_conversion", run_steroid_conversion, "Steroid Conversion Calculator"),
    "mme": _make_calc_entry("mme", run_mme, "Morphine Milligram Equivalents (MME)"),
}

# ── Schema overrides: expose exact categorical options via calc_info ─────────
_SCHEMA_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "glasgow_coma_score": {
        "gcs_eye": {"type": "categorical", "constraints": {
            "allowed_values": list(_GCS_EYE.keys()),
        }},
        "gcs_verbal": {"type": "categorical", "constraints": {
            "allowed_values": list(_GCS_VERBAL.keys()),
        }},
        "gcs_motor": {"type": "categorical", "constraints": {
            "allowed_values": list(_GCS_MOTOR.keys()),
        }},
    },
    "child_pugh": {
        "ascites": {"type": "categorical", "constraints": {
            "allowed_values": list(_ASCITES_SCORE.keys()),
        }},
        "encephalopathy": {"type": "categorical", "constraints": {
            "allowed_values": list(_ENCEPH_SCORE.keys()),
        }},
    },
    "cci": {
        "diabetes_mellitus": {"type": "categorical", "constraints": {
            "allowed_values": ["None or diet-controlled", "Uncomplicated", "End-organ damage"],
        }},
        "liver_disease": {"type": "categorical", "constraints": {
            "allowed_values": ["None", "Mild", "Moderate to severe"],
        }},
        "solid_tumor": {"type": "categorical", "constraints": {
            "allowed_values": ["None", "Localized", "Metastatic"],
        }},
    },
}

# Apply overrides to CALCULATORS schemas
for _calc_id, _field_overrides in _SCHEMA_OVERRIDES.items():
    if _calc_id in CALCULATORS:
        _schema = CALCULATORS[_calc_id].get("schema", {})
        _inputs = _schema.get("inputs", [])
        for _inp in _inputs:
            _fid = _inp.get("id", "")
            if _fid in _field_overrides:
                _inp.update(_field_overrides[_fid])
