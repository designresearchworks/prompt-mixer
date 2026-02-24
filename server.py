import os
import json
import threading
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6")
ENV_PATH = os.path.join(os.getcwd(), ".env")
PROMPT_STORE_DIR = os.path.join(os.getcwd(), "data")
PROMPT_PREAMBLE_PATH = os.path.join(PROMPT_STORE_DIR, "base_preamble.txt")
PROMPT_CLOSING_PATH = os.path.join(PROMPT_STORE_DIR, "base_closing.txt")
APP_CONFIG_PATH = os.path.join(PROMPT_STORE_DIR, "app_config.json")
API_COUNTER_PATH = os.path.join(PROMPT_STORE_DIR, "api_call_counter.json")
API_COUNTER_LOCK = threading.Lock()
DEFAULT_APP_CONFIG = {
    "default_theme": "dark",
    "default_model": "groq:llama-3.3-70b-versatile",
    "default_thought": "deep thought",
    "default_fish_concepts": "animal, vegetable, mineral",
    "cadence_min_ms": 0,
    "cadence_max_ms": 5000,
    "cadence_default_ms": 2500,
    "infinite_inactivity_ms": 30000,
    "touch_ui_grace_ms": 320,
    "chip_max_speed": 0.9,
    "fish_scale_min_pct": -100,
    "fish_scale_max_pct": 100,
    "fish_scale_default_pct": 0,
    "chip_scale_base": 0.902,
    "chip_scale_weight_factor": 0.584,
    "rnd_base_min_factor": 0.35,
    "rnd_base_max_factor": 0.65,
    "rnd_launch_min_factor": 0.10,
    "rnd_launch_max_factor": 0.40,
    "deadzone_half": 0.10,
    "merge_hold_ms": 1000,
    "join_glow_ms": 650,
    "model_progress_default_ms": 3000,
}


class EnvKeysPayload(BaseModel):
    openrouter_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


class PromptSectionPayload(BaseModel):
    section: str
    content: str


class PromptTemplatesResetPayload(BaseModel):
    preamble: str
    closing: str


def _upsert_env_values(path: str, updates: dict[str, str]) -> None:
    lines: list[str] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    remaining = dict(updates)
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            result.append(line)
            continue

        key = line.split("=", 1)[0].strip()
        if key in remaining:
            result.append(f"{key}={remaining.pop(key)}")
        else:
            result.append(line)

    for key, value in remaining.items():
        result.append(f"{key}={value}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(result).rstrip() + "\n")


def _read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, value: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _normalize_app_config(raw: Optional[dict]) -> dict:
    if not isinstance(raw, dict):
        return {}

    out: dict = {}
    for key, default in DEFAULT_APP_CONFIG.items():
        if key not in raw:
            continue
        val = raw.get(key)

        if isinstance(default, str):
            if val is None:
                continue
            out[key] = str(val)
            continue

        if isinstance(default, int):
            try:
                out[key] = int(val)
            except (TypeError, ValueError):
                continue
            continue

        if isinstance(default, float):
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                continue
            continue

    # Basic bounds and consistency guards.
    if "cadence_min_ms" in out:
        out["cadence_min_ms"] = max(0, out["cadence_min_ms"])
    if "cadence_max_ms" in out:
        out["cadence_max_ms"] = max(0, out["cadence_max_ms"])
    if "cadence_default_ms" in out:
        out["cadence_default_ms"] = max(0, out["cadence_default_ms"])
    if "infinite_inactivity_ms" in out:
        out["infinite_inactivity_ms"] = max(5000, out["infinite_inactivity_ms"])
    if "touch_ui_grace_ms" in out:
        out["touch_ui_grace_ms"] = max(0, out["touch_ui_grace_ms"])
    if "chip_max_speed" in out:
        out["chip_max_speed"] = max(0.05, out["chip_max_speed"])
    if "fish_scale_min_pct" in out:
        out["fish_scale_min_pct"] = max(-200, min(0, out["fish_scale_min_pct"]))
    if "fish_scale_max_pct" in out:
        out["fish_scale_max_pct"] = max(0, min(400, out["fish_scale_max_pct"]))
    if "fish_scale_default_pct" in out:
        out["fish_scale_default_pct"] = max(-200, min(400, out["fish_scale_default_pct"]))
    if "chip_scale_base" in out:
        out["chip_scale_base"] = max(0.2, min(3.0, out["chip_scale_base"]))
    if "chip_scale_weight_factor" in out:
        out["chip_scale_weight_factor"] = max(0.0, min(3.0, out["chip_scale_weight_factor"]))
    if "rnd_base_min_factor" in out:
        out["rnd_base_min_factor"] = max(0.0, min(2.0, out["rnd_base_min_factor"]))
    if "rnd_base_max_factor" in out:
        out["rnd_base_max_factor"] = max(0.0, min(2.0, out["rnd_base_max_factor"]))
    if "rnd_launch_min_factor" in out:
        out["rnd_launch_min_factor"] = max(0.0, min(2.0, out["rnd_launch_min_factor"]))
    if "rnd_launch_max_factor" in out:
        out["rnd_launch_max_factor"] = max(0.0, min(2.0, out["rnd_launch_max_factor"]))
    if "deadzone_half" in out:
        out["deadzone_half"] = max(0.0, min(0.49, out["deadzone_half"]))
    if "merge_hold_ms" in out:
        out["merge_hold_ms"] = max(200, out["merge_hold_ms"])
    if "join_glow_ms" in out:
        out["join_glow_ms"] = max(100, out["join_glow_ms"])
    if "model_progress_default_ms" in out:
        out["model_progress_default_ms"] = max(300, out["model_progress_default_ms"])
    if "default_theme" in out:
        theme = str(out["default_theme"]).strip().lower()
        out["default_theme"] = "light" if theme == "light" else "dark"

    return out


def _get_current_app_config() -> dict:
    saved = _normalize_app_config(_read_json(APP_CONFIG_PATH))
    merged = dict(DEFAULT_APP_CONFIG)
    merged.update(saved)

    # Cross-field constraints after merge.
    if merged["cadence_max_ms"] < merged["cadence_min_ms"]:
        merged["cadence_max_ms"] = merged["cadence_min_ms"]
    merged["cadence_default_ms"] = max(
        merged["cadence_min_ms"],
        min(merged["cadence_max_ms"], merged["cadence_default_ms"]),
    )

    if merged["fish_scale_max_pct"] < merged["fish_scale_min_pct"]:
        merged["fish_scale_max_pct"] = merged["fish_scale_min_pct"]
    merged["fish_scale_default_pct"] = max(
        merged["fish_scale_min_pct"],
        min(merged["fish_scale_max_pct"], merged["fish_scale_default_pct"]),
    )

    if merged["rnd_base_max_factor"] < merged["rnd_base_min_factor"]:
        merged["rnd_base_max_factor"] = merged["rnd_base_min_factor"]
    if merged["rnd_launch_max_factor"] < merged["rnd_launch_min_factor"]:
        merged["rnd_launch_max_factor"] = merged["rnd_launch_min_factor"]

    return merged


def _read_api_call_count() -> int:
    data = _read_json(API_COUNTER_PATH)
    if not isinstance(data, dict):
        return 0
    try:
        n = int(data.get("total_api_calls", 0))
    except (TypeError, ValueError):
        n = 0
    return max(0, n)


def _increment_api_call_count() -> int:
    with API_COUNTER_LOCK:
        current = _read_api_call_count()
        next_value = current + 1
        _write_json(API_COUNTER_PATH, {"total_api_calls": next_value})
        return next_value


def _ensure_api_counter_file() -> None:
    if os.path.exists(API_COUNTER_PATH):
        return
    _write_json(API_COUNTER_PATH, {"total_api_calls": 0})


_ensure_api_counter_file()


@app.post("/api/generate")
async def generate(request: Request):
    _increment_api_call_count()
    body = await request.json()
    prompt = body.get("prompt", "")
    model = body.get("model") or DEFAULT_MODEL

    # 🔁 Dynamic provider routing
    if model.startswith("groq:"):
        actual_model = model.replace("groq:", "")
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        actual_model = model
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    async def stream_tokens():
        try:
            stream = client.chat.completions.create(
                model=actual_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text is None:
                    continue

                # Escape for SSE
                safe = text.replace("\\", "\\\\").replace("\n", "\\n")
                yield f"data: {safe}\n\n"

            yield "data: [DONE]\n\n"

        except Exception:
            yield "data: [ERROR]\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


@app.post("/api/save-keys")
async def save_keys(payload: EnvKeysPayload):
    global OPENROUTER_API_KEY, GROQ_API_KEY

    updates: dict[str, str] = {}

    if payload.openrouter_api_key is not None and payload.openrouter_api_key.strip():
        value = payload.openrouter_api_key.strip()
        updates["OPENROUTER_API_KEY"] = value
        OPENROUTER_API_KEY = value

    if payload.groq_api_key is not None and payload.groq_api_key.strip():
        value = payload.groq_api_key.strip()
        updates["GROQ_API_KEY"] = value
        GROQ_API_KEY = value

    if not updates:
        return {"ok": False, "message": "No keys provided"}

    _upsert_env_values(ENV_PATH, updates)
    return {"ok": True, "message": "Saved keys to .env", "saved": list(updates.keys())}


@app.get("/api/prompt-templates")
async def get_prompt_templates():
    return {
        "ok": True,
        "preamble": _read_text(PROMPT_PREAMBLE_PATH),
        "closing": _read_text(PROMPT_CLOSING_PATH),
    }


@app.post("/api/prompt-templates/save")
async def save_prompt_section(payload: PromptSectionPayload):
    section = (payload.section or "").strip().lower()
    content = payload.content or ""
    if section not in ("preamble", "closing"):
        return {"ok": False, "message": "section must be 'preamble' or 'closing'"}

    path = PROMPT_PREAMBLE_PATH if section == "preamble" else PROMPT_CLOSING_PATH
    _write_text(path, content)
    return {"ok": True, "saved": section}


@app.post("/api/prompt-templates/reset")
async def reset_prompt_templates(payload: PromptTemplatesResetPayload):
    _write_text(PROMPT_PREAMBLE_PATH, payload.preamble or "")
    _write_text(PROMPT_CLOSING_PATH, payload.closing or "")
    return {"ok": True, "saved": ["preamble", "closing"]}


@app.get("/api/app-config")
async def get_app_config():
    return {
        "ok": True,
        "config": _get_current_app_config(),
        "defaults": dict(DEFAULT_APP_CONFIG),
    }


@app.post("/api/app-config/save")
async def save_app_config(request: Request):
    body = await request.json()
    payload = body.get("config") if isinstance(body, dict) and "config" in body else body
    updates = _normalize_app_config(payload if isinstance(payload, dict) else {})

    current = _get_current_app_config()
    current.update(updates)
    # Re-run constraints through the same merger path by overlaying on defaults.
    persisted = dict(DEFAULT_APP_CONFIG)
    persisted.update(current)
    # Save as full expanded config for transparency.
    _write_json(APP_CONFIG_PATH, persisted)

    return {"ok": True, "config": _get_current_app_config()}


@app.post("/api/app-config/reset")
async def reset_app_config():
    _write_json(APP_CONFIG_PATH, dict(DEFAULT_APP_CONFIG))
    return {"ok": True, "config": dict(DEFAULT_APP_CONFIG)}


@app.get("/api/stats")
async def get_stats():
    return {
        "ok": True,
        "total_api_calls": _read_api_call_count(),
    }


# Static files last
app.mount("/", StaticFiles(directory="public", html=True), name="static")
