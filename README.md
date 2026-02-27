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
- `OMNICALC_ORCHESTRATE_API_MODE` (`chat_completions` or `responses`, default: `responses`)
- `OMNICALC_STREAM_API_MODE` (`chat_v1` or `responses`, default: `responses`)
- `MEDASR_BACKEND` (default: `mlx`)
- `MEDASR_MODEL_PATH` (default: `google/medasr`)

## Engine Compatibility

OmniCalc is now **`/v1/responses`-first** by default.

You can generally use any engine/provider if it is OpenAI-compatible and supports:
- `GET /v1/models`
- `POST /v1/responses` with function tools (`type: "function"`)
- function call outputs mapped back into conversation state (equivalent of `function_call_output`)

If your engine does not support `/v1/responses`, you can still use:
- `OMNICALC_ORCHESTRATE_API_MODE=chat_completions`

Streaming remains:
- `OMNICALC_STREAM_API_MODE=responses` for portable OpenAI-compatible engines, or
- `OMNICALC_STREAM_API_MODE=chat_v1` for LM Studio-only mode.

Notes:
- `chat_v1` is LM Studio-specific (`/api/v1/chat`) and not expected on other providers.
- Tool calls must be valid structured function calls. We do **not** rely on malformed marker recovery.

## Notes

- `NOTICE.txt` contains additional HAI-DEF terms notice.
- `LICENSE` defines repository licensing.

## Disclaimer

Research/demo software only. Not a medical device and not for clinical deployment as-is.
