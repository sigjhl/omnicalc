# OmniCalc

Voice-first clinical scoring agent for the MedGemma Impact Challenge 2026.

OmniCalc converts natural clinical language into structured variables, then computes results with deterministic Python calculators.

Core principle: **AI Extracts, Engine Computes**.

## Quick Start

### Prerequisites
- Python + `uv`
- LM Studio running locally at `http://localhost:1234`
- A compatible model loaded in LM Studio (for example, your MedGemma fine-tune)

### Run
```bash
uv run uvicorn omnicalc.api:app --reload --host 0.0.0.0 --port 8002
```

Open `http://localhost:8002`.

## Configuration

- `LM_STUDIO_URL` (default: `http://localhost:1234/v1`)
- `MODEL_NAME` (default: `sigjhl/medgemma-1.5-4b-it-MedCalcCaller`)
- `MEDASR_BACKEND` (default: `mlx`)
- `MEDASR_MODEL_PATH` (default: `google/medasr`)

## Smoke Test (Optional)

Requires LM Studio + model loaded.

```bash
uv run python scripts/eval_medcalc_bench.py \
  --dataset scripts/sample_cases_meld_na.jsonl \
  --out /tmp/omnicalc_eval.json
```

## Notes

- `NOTICE.txt` contains additional HAI-DEF terms notice.
- `LICENSE` defines repository licensing.

## Disclaimer

Research/demo software only. Not a medical device and not for clinical deployment as-is.
