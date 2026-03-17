"""
Prompt Mixer backend service.

What:
- Hosts UI pages and API endpoints for text generation + fisheye image generation.
- Persists runtime config/templates/stats to local files under `data/`.

Why:
- Keep deployment simple (single service) while retaining state across restarts.

How:
- FastAPI routes orchestrate provider calls (OpenRouter/Groq/Replicate/Daydream)
  and file-backed persistence helpers.
"""

import os
import json
import shutil
import io
import threading
import time
import uuid
import traceback
import mimetypes
import zipfile
import urllib.error
import urllib.request
import urllib.parse
import base64
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image, PngImagePlugin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

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
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
DAYDREAM_API_KEY = os.getenv("DAYDREAM_API_KEY")
DEFAULT_REPLICATE_MODEL = os.getenv(
    "REPLICATE_MODEL",
    "prunaai/flux-fast",
)
REPLICATE_API_BASE = "https://api.replicate.com/v1"
DAYDREAM_API_BASE = os.getenv("DAYDREAM_API_BASE", "https://api.daydream.live")
DAYDREAM_ROUTE_HINTS: dict[str, Optional[tuple[str, str]]] = {
    "update": None,
    "status": None,
    "stop": None,
}
PROMPT_STORE_DIR = os.path.join(BASE_DIR, "data")
PROMPT_PREAMBLE_PATH = os.path.join(PROMPT_STORE_DIR, "base_preamble.txt")
PROMPT_CLOSING_PATH = os.path.join(PROMPT_STORE_DIR, "base_closing.txt")
APP_CONFIG_PATH = os.path.join(PROMPT_STORE_DIR, "app_config.json")
API_COUNTER_PATH = os.path.join(PROMPT_STORE_DIR, "api_call_counter.json")
FISHEYE_IMAGE_DIR = os.path.join(PROMPT_STORE_DIR, "fisheye_images")
FISHEYE_IMAGE_INDEX_PATH = os.path.join(PROMPT_STORE_DIR, "fisheye_images_index.json")
FISHEYE_GAME_ROUNDS_PATH = os.path.join(PROMPT_STORE_DIR, "fisheye_game_rounds.json")
SNAPSHOT_DIR = os.path.join(PROMPT_STORE_DIR, "snapshots")
SNAPSHOT_IMAGE_DIR = os.path.join(SNAPSHOT_DIR, "images")
API_COUNTER_LOCK = threading.Lock()
FISHEYE_GAME_LOCK = threading.Lock()
EMBEDDED_IMAGE_METADATA_KEY = "pthinktank_state"
EMBEDDED_IMAGE_METADATA_VERSION = 1
DEFAULT_APP_CONFIG = {
    "default_theme": "dark",
    "default_model": "groq:llama-3.3-70b-versatile",
    "default_thought": "deep thought",
    "default_fish_concepts": "animal, vegetable, mineral",
    "default_fisheye_enabled": True,
    "default_keep_brief": False,
    "fisheye_image_history_limit": 1000,
    "slw_delay_ms": 800,
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
    "llm_deterministic_mode": False,
    "llm_deterministic_seed": 2110,
}
DEFAULT_BASE_PREAMBLE = (
    "You have several transformation concepts to apply to the thought. Each has two properties:\n"
    "- Weight (0.00 = ignore, 1.00 = apply strongly)\n"
    "- Optionally, lean toward a sensibility"
)
DEFAULT_BASE_CLOSING = (
    "Reimagine the original thought applying these transformations proportionally.\n"
    "Higher-weighted concepts should have a more visible effect on the output.\n"
    "Concepts that lean toward a sensibility should pull the style in that direction.\n"
    "Preserve the core meaning of the original thought.\n"
    "Only output your new idea, no meta commentary, none of the original prompt."
)


class EnvKeysPayload(BaseModel):
    openrouter_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    replicate_api_key: Optional[str] = None


class PromptSectionPayload(BaseModel):
    section: str
    content: str


class PromptTemplatesResetPayload(BaseModel):
    preamble: str
    closing: str


class VisualisePayload(BaseModel):
    output_text: str
    prompt_model: Optional[str] = None
    image_model: Optional[str] = None
    image_seed: Optional[int] = None
    prompt_strength: Optional[float] = None
    use_input_image: Optional[bool] = False
    input_image_ref: Optional[str] = None
    input_image_url: Optional[str] = None
    creation_state: Optional[dict] = None


class FisheyeImageImportPayload(BaseModel):
    image_ref: Optional[str] = None
    source_url: Optional[str] = None


class SnapshotCreatePayload(BaseModel):
    name: Optional[str] = None
    state: dict


class FisheyeGameFishPayload(BaseModel):
    label: Optional[str] = None
    x_frac: float
    y_frac: float


class FisheyeGameRoundCreatePayload(BaseModel):
    thought: str
    decoy_thought: str
    fish: list[FisheyeGameFishPayload]
    direction_1: Optional[str] = None
    direction_2: Optional[str] = None
    keep_brief: Optional[bool] = True
    base_preamble: Optional[str] = None
    base_closing: Optional[str] = None
    prompt_model: Optional[str] = None
    image_model: Optional[str] = None
    image_seed: Optional[int] = None


class FisheyeGameGuessPayload(BaseModel):
    thought_index: int
    fish: list[FisheyeGameFishPayload]


class DaydreamCreatePayload(BaseModel):
    prompt: str
    model_id: Optional[str] = None


class DaydreamUpdatePayload(BaseModel):
    stream_id: str
    prompt: Optional[str] = None
    model_id: Optional[str] = None


class DaydreamStopPayload(BaseModel):
    stream_id: str


# Environment writer: used by the settings UI to persist API keys to `.env`.
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


# Reads a UTF-8 text file and returns None when the file does not exist.
def _read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Writes UTF-8 text to disk, creating parent directories when needed.
def _write_text(path: str, value: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


# Reads a JSON object from disk and returns None on missing or invalid data.
def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# Writes a JSON object to disk with stable formatting.
def _write_json(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=True)
        f.write("\n")


# Reads a list of JSON objects from disk, accepting either a bare list or an items wrapper.
def _read_json_list(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [x for x in data.get("items", []) if isinstance(x, dict)]
    return []


# Writes a list of JSON objects using the app's items wrapper format.
def _write_json_list(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, indent=2, ensure_ascii=True)
        f.write("\n")


# Config normalization exists so malformed UI values do not break runtime behavior.
def _normalize_app_config(raw: Optional[dict]) -> dict:
    if not isinstance(raw, dict):
        return {}

    out: dict = {}
    for key, default in DEFAULT_APP_CONFIG.items():
        if key not in raw:
            continue
        val = raw.get(key)

        if isinstance(default, bool):
            if isinstance(val, bool):
                out[key] = val
                continue
            if isinstance(val, (int, float)):
                out[key] = bool(val)
                continue
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    out[key] = True
                    continue
                if lowered in {"0", "false", "no", "off"}:
                    out[key] = False
                    continue
            continue

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
    if "fisheye_image_history_limit" in out:
        out["fisheye_image_history_limit"] = max(1, min(100000, out["fisheye_image_history_limit"]))
    if "slw_delay_ms" in out:
        out["slw_delay_ms"] = max(0, min(5000, out["slw_delay_ms"]))
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
    if "llm_deterministic_seed" in out:
        out["llm_deterministic_seed"] = max(0, min(2147483647, out["llm_deterministic_seed"]))
    if "default_theme" in out:
        theme = str(out["default_theme"]).strip().lower()
        out["default_theme"] = "light" if theme == "light" else "dark"

    return out


# Returns the current app configuration merged over defaults.
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


# Reads the persisted API call counter from disk.
def _read_api_call_count() -> int:
    data = _read_json(API_COUNTER_PATH)
    if not isinstance(data, dict):
        return 0
    try:
        n = int(data.get("total_api_calls", 0))
    except (TypeError, ValueError):
        n = 0
    return max(0, n)


# Increments the API call counter and returns the new value.
def _increment_api_call_count() -> int:
    with API_COUNTER_LOCK:
        current = _read_api_call_count()
        next_value = current + 1
        _write_json(API_COUNTER_PATH, {"total_api_calls": next_value})
        return next_value


# Creates the API counter file if it does not exist yet.
def _ensure_api_counter_file() -> None:
    if os.path.exists(API_COUNTER_PATH):
        return
    _write_json(API_COUNTER_PATH, {"total_api_calls": 0})


# Provider router: same UI can target Groq or OpenRouter models using one API route.
def _build_model_client(model: str) -> tuple[OpenAI, str]:
    if model.startswith("groq:"):
        return (
            OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            ),
            model.replace("groq:", ""),
        )
    return (
        OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        ),
        model,
    )


# Attempts to coerce a loose value into a boolean.
def _coerce_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


# Normalizes a seed value and falls back when the provided value is invalid.
def _coerce_seed(value, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(0, min(2147483647, parsed))


# Determines the final seed and temperature settings for a text generation request.
def _resolve_text_determinism(
    deterministic_override=None,
    seed_override=None,
) -> tuple[bool, int]:
    cfg = _get_current_app_config()
    cfg_deterministic = bool(cfg.get("llm_deterministic_mode", False))
    cfg_seed = _coerce_seed(cfg.get("llm_deterministic_seed"), 2110)
    deterministic = _coerce_bool(deterministic_override)
    if deterministic is None:
        deterministic = cfg_deterministic
    seed = _coerce_seed(seed_override, cfg_seed)
    return deterministic, seed


# Executes a chat completion request against the selected model provider.
def _chat_completion_create(
    client: OpenAI,
    deterministic: bool,
    seed: int,
    **kwargs,
):
    payload = dict(kwargs)
    if deterministic:
        payload["temperature"] = 0
        payload["top_p"] = 1
        payload["presence_penalty"] = 0
        payload["frequency_penalty"] = 0
        payload["seed"] = seed
    try:
        return client.chat.completions.create(**payload)
    except Exception as exc:
        # Some provider/model combinations reject `seed`; keep deterministic sampling settings.
        if deterministic and "seed" in str(exc).lower():
            payload.pop("seed", None)
            return client.chat.completions.create(**payload)
        raise


# Calls the Replicate HTTP API and returns the parsed JSON response.
def _replicate_http_json(
    url: str,
    method: str = "GET",
    payload: Optional[dict] = None,
    prefer_wait: bool = False,
) -> dict:
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        raise ValueError("REPLICATE_API_TOKEN is not configured")

    headers = {
        "Authorization": f"Bearer {replicate_token}",
        "Accept": "application/json",
    }
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if prefer_wait:
        headers["Prefer"] = "wait"

    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Replicate HTTP {exc.code}: {raw}") from exc

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Replicate returned non-JSON payload: {raw[:240]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Replicate returned unexpected response shape")
    return parsed


# Downloads provider-hosted image bytes so we can keep a local cache/history.
def _download_binary(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(
        url=url,
        headers={"User-Agent": "prompt-mixer/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        content_type = str(resp.headers.get("Content-Type") or "")
        payload = resp.read()
    if not payload:
        raise RuntimeError("Downloaded image payload is empty")
    return payload, content_type


# Calls the Daydream API and returns the parsed JSON response.
def _daydream_http_json(
    path_or_url: str,
    method: str = "GET",
    payload: Optional[dict] = None,
    timeout: int = 120,
) -> dict:
    api_key = os.getenv("DAYDREAM_API_KEY")
    if not api_key:
        raise ValueError("DAYDREAM_API_KEY is not configured")

    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        url = path_or_url
    else:
        base = (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE).rstrip("/")
        url = base + "/" + path_or_url.lstrip("/")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Daydream HTTP {exc.code}: {raw}") from exc

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        if not raw.strip():
            return {}
        raise RuntimeError(f"Daydream returned non-JSON payload: {raw[:240]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Daydream returned unexpected response shape")
    return parsed


# Attempt wrapper returns both response and rich attempt metadata for diagnostics.
def _daydream_attempt(
    method: str,
    path: str,
    payload: Optional[dict] = None,
    timeout: int = 120,
) -> tuple[Optional[dict], dict]:
    started = time.perf_counter()
    attempt = {
        "method": method,
        "path": path,
        "ok": False,
        "elapsed_ms": 0,
    }
    try:
        data = _daydream_http_json(path, method=method, payload=payload, timeout=timeout)
        attempt["ok"] = True
        return data, attempt
    except Exception as exc:
        attempt["error"] = str(exc)
        return None, attempt
    finally:
        attempt["elapsed_ms"] = int((time.perf_counter() - started) * 1000)


# Extracts a Daydream stream id from a response payload.
def _extract_daydream_stream_id(payload: dict) -> Optional[str]:
    candidates = [
        payload.get("stream_id"),
        payload.get("id"),
        (payload.get("stream") or {}).get("id") if isinstance(payload.get("stream"), dict) else None,
        (payload.get("data") or {}).get("id") if isinstance(payload.get("data"), dict) else None,
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


# Builds a stream-specific file path from a template path.
def _path_to_template(path: str, stream_id: str) -> str:
    stream = (stream_id or "").strip()
    if not stream:
        return path
    return path.replace(stream, "{id}", 1)


# Infers a file extension for an image from its URL and content type.
def _infer_image_extension(source_url: str, content_type: str) -> str:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    ext = mimetypes.guess_extension(ct) if ct else None
    if ext in (".jpe",):
        ext = ".jpg"
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"):
        return ext

    path_ext = os.path.splitext(urllib.parse.urlparse(source_url).path)[1].lower()
    if path_ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"):
        return path_ext
    return ".jpg"


# Normalizes embedded image metadata into a JSON-safe dict.
def _sanitize_embedded_image_metadata(metadata: Optional[dict]) -> Optional[dict]:
    if not isinstance(metadata, dict):
        return None
    try:
        safe = json.loads(json.dumps(metadata))
    except Exception:
        return None
    return safe if isinstance(safe, dict) else None


# Merges existing embedded image metadata with new values and app metadata fields.
def _merge_embedded_image_metadata(existing: Optional[dict], updates: Optional[dict]) -> Optional[dict]:
    merged: dict = {}
    safe_existing = _sanitize_embedded_image_metadata(existing)
    safe_updates = _sanitize_embedded_image_metadata(updates)
    if isinstance(safe_existing, dict):
        merged.update(safe_existing)
    if isinstance(safe_updates, dict):
        merged.update(safe_updates)
    if not merged:
        return None
    merged["app"] = "PThinkTank"
    merged["schema_version"] = EMBEDDED_IMAGE_METADATA_VERSION
    merged["saved_at"] = int(time.time())
    return merged


# Reads embedded PThinkTank metadata from raw image bytes when present.
def _extract_embedded_image_metadata_from_bytes(payload: bytes) -> Optional[dict]:
    if not payload:
        return None
    try:
        with Image.open(io.BytesIO(payload)) as img:
            raw = None
            if hasattr(img, "text") and isinstance(getattr(img, "text"), dict):
                raw = img.text.get(EMBEDDED_IMAGE_METADATA_KEY)
            if raw is None and isinstance(getattr(img, "info", None), dict):
                raw = img.info.get(EMBEDDED_IMAGE_METADATA_KEY)
            if not isinstance(raw, str) or not raw.strip():
                return None
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# Reads embedded PThinkTank metadata from a locally stored image file.
def _read_embedded_image_metadata(file_path: str) -> Optional[dict]:
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            return _extract_embedded_image_metadata_from_bytes(f.read())
    except Exception:
        return None


# Converts an image into a PNG file with embedded PThinkTank metadata.
def _write_png_with_embedded_metadata(payload: bytes, file_path: str, metadata: Optional[dict]) -> None:
    pnginfo = None
    safe_metadata = _sanitize_embedded_image_metadata(metadata)
    if safe_metadata:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text(
            EMBEDDED_IMAGE_METADATA_KEY,
            json.dumps(safe_metadata, separators=(",", ":"), ensure_ascii=True),
        )

    with Image.open(io.BytesIO(payload)) as img:
        img.load()
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        converted = img.convert("RGBA" if has_alpha else "RGB")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        converted.save(file_path, format="PNG", pnginfo=pnginfo, optimize=True)


# Extracts a restorable UI state payload from embedded image metadata when present.
def _extract_restorable_state_from_embedded_metadata(metadata: Optional[dict]) -> Optional[dict]:
    safe = _sanitize_embedded_image_metadata(metadata)
    if not safe:
        return None
    for key in ("snapshot_state", "creation_state", "state"):
        value = safe.get(key)
        if isinstance(value, dict):
            return value
    return None


# Downloads and stores a fisheye image as a PNG with embedded metadata.
def _store_fisheye_image(source_url: str, embedded_metadata: Optional[dict] = None) -> tuple[str, str]:
    payload, content_type = _download_binary(source_url)
    return _store_fisheye_image_payload(
        payload=payload,
        content_type=content_type,
        source_url=source_url,
        embedded_metadata=embedded_metadata,
    )


# Stores image bytes as a fisheye image PNG with embedded metadata.
def _store_fisheye_image_payload(
    payload: bytes,
    content_type: str = "",
    source_url: str = "",
    embedded_metadata: Optional[dict] = None,
) -> tuple[str, str]:
    os.makedirs(FISHEYE_IMAGE_DIR, exist_ok=True)
    source_ext = _infer_image_extension(source_url, content_type)
    existing_metadata = _extract_embedded_image_metadata_from_bytes(payload)
    merged_metadata = _merge_embedded_image_metadata(
        existing_metadata,
        {
            "source_url": source_url,
            "source_extension": source_ext,
            **(_sanitize_embedded_image_metadata(embedded_metadata) or {}),
        },
    )
    image_ref = uuid.uuid4().hex[:20]
    filename = f"{image_ref}.png"
    file_path = os.path.join(FISHEYE_IMAGE_DIR, filename)
    _write_png_with_embedded_metadata(payload, file_path, merged_metadata)

    items = _read_json_list(FISHEYE_IMAGE_INDEX_PATH)
    items.append(
        {
            "ref": image_ref,
            "filename": filename,
            "created_at": int(time.time()),
            "source_url": source_url,
            "embedded_metadata": bool(merged_metadata),
        }
    )

    retention_limit = int(_get_current_app_config().get("fisheye_image_history_limit", 1000))
    retention_limit = max(1, min(100000, retention_limit))

    # Keep only the most recent configured number of images.
    if len(items) > retention_limit:
        stale = items[:-retention_limit]
        items = items[-retention_limit:]
        for old in stale:
            old_name = str(old.get("filename") or "").strip()
            if not old_name:
                continue
            old_path = os.path.join(FISHEYE_IMAGE_DIR, old_name)
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
            except Exception:
                pass

    _write_json_list(FISHEYE_IMAGE_INDEX_PATH, items)
    return image_ref, f"/api/fisheye-images/{image_ref}"


# Looks up a stored fisheye image record by its ref.
def _find_fisheye_image_record(image_ref: str) -> Optional[dict]:
    ref = (image_ref or "").strip()
    if not ref:
        return None
    items = _read_json_list(FISHEYE_IMAGE_INDEX_PATH)
    return next((x for x in items if str(x.get("ref", "")).strip() == ref), None)


# Builds the JSON path for a snapshot id.
def _snapshot_path(snapshot_id: str) -> str:
    safe_id = (snapshot_id or "").strip()
    return os.path.join(SNAPSHOT_DIR, f"{safe_id}.json")


# Builds the lightweight snapshot summary returned to the UI.
def _snapshot_summary(record: dict) -> dict:
    return {
        "id": str(record.get("id") or "").strip(),
        "name": str(record.get("name") or "").strip(),
        "created_at": int(record.get("created_at") or 0),
    }


# Reads a full snapshot record from disk.
def _read_snapshot(snapshot_id: str) -> Optional[dict]:
    data = _read_json(_snapshot_path(snapshot_id))
    return data if isinstance(data, dict) else None


# Loads and sorts all stored snapshots for the snapshot picker.
def _list_snapshots() -> list[dict]:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    items: list[dict] = []
    for name in os.listdir(SNAPSHOT_DIR):
        if not name.endswith(".json"):
            continue
        data = _read_json(os.path.join(SNAPSHOT_DIR, name))
        if not isinstance(data, dict):
            continue
        snapshot_id = str(data.get("id") or os.path.splitext(name)[0]).strip()
        if not snapshot_id:
            continue
        items.append(
            {
                "id": snapshot_id,
                "name": str(data.get("name") or snapshot_id).strip(),
                "created_at": int(data.get("created_at") or 0),
            }
        )
    items.sort(key=lambda x: (int(x.get("created_at") or 0), str(x.get("id") or "")), reverse=True)
    return items


# Resolves the local file path for a stored fisheye image ref.
def _local_fisheye_file_path(image_ref: str) -> Optional[str]:
    record = _find_fisheye_image_record(image_ref)
    if not record:
        return None
    filename = str(record.get("filename") or "").strip()
    if not filename:
        return None
    file_path = os.path.join(FISHEYE_IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        return None
    return file_path


# Extracts a fisheye image ref from one of the app's served image URLs.
def _coerce_fisheye_ref_from_url(image_url: Optional[str]) -> Optional[str]:
    raw = (image_url or "").strip()
    if not raw:
        return None
    parsed = urllib.parse.urlparse(raw)
    path = parsed.path or raw
    marker = "/api/fisheye-images/"
    if marker not in path:
        return None
    return path.rsplit("/", 1)[-1].strip() or None


# Parses a snapshot-owned image URL into its snapshot id and filename.
def _parse_snapshot_image_url(image_url: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    raw = (image_url or "").strip()
    if not raw:
        return None, None
    path = urllib.parse.urlparse(raw).path or raw
    marker = "/api/snapshots/"
    if marker not in path or "/images/" not in path:
        return None, None
    tail = path.split(marker, 1)[1]
    snapshot_id, _, image_name = tail.partition("/images/")
    snapshot_id = snapshot_id.strip().strip("/")
    image_name = image_name.strip().strip("/")
    if not snapshot_id or not image_name or "/" in image_name:
        return None, None
    return snapshot_id, image_name


# Builds the on-disk path for a snapshot-owned image file.
def _snapshot_image_path(snapshot_id: str, image_name: str) -> str:
    safe_snapshot_id = (snapshot_id or "").strip()
    safe_image_name = os.path.basename((image_name or "").strip())
    return os.path.join(SNAPSHOT_IMAGE_DIR, safe_snapshot_id, safe_image_name)


# Resolves a locally available image file from the supported ref or URL forms.
def _resolve_local_image_file(
    image_ref: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Optional[str]:
    ref = (image_ref or "").strip() or (_coerce_fisheye_ref_from_url(image_url) or "")
    if ref:
        file_path = _local_fisheye_file_path(ref)
        if file_path:
            return file_path

    snapshot_id, image_name = _parse_snapshot_image_url(image_url)
    if snapshot_id and image_name:
        file_path = _snapshot_image_path(snapshot_id, image_name)
        if os.path.exists(file_path):
            return file_path
    return None


# Copies one referenced image into snapshot-owned storage and returns its served URL.
def _copy_snapshot_image(
    snapshot_id: str,
    slot_name: str,
    image_ref: Optional[str] = None,
    image_url: Optional[str] = None,
    embedded_metadata: Optional[dict] = None,
) -> Optional[dict]:
    slot = (slot_name or "").strip() or "image"
    source_file = _resolve_local_image_file(image_ref=image_ref, image_url=image_url)
    source_url = (image_url or "").strip()
    source_payload: Optional[bytes] = None
    source_metadata: Optional[dict] = None

    if source_file and os.path.exists(source_file):
        with open(source_file, "rb") as f:
            source_payload = f.read()
        source_metadata = _read_embedded_image_metadata(source_file)
    elif source_url.startswith(("http://", "https://")):
        source_payload, _ = _download_binary(source_url)
        source_metadata = _extract_embedded_image_metadata_from_bytes(source_payload)
    else:
        return None

    if not source_payload:
        return None

    image_name = f"{slot}.png"
    dest_path = _snapshot_image_path(snapshot_id, image_name)
    merged_metadata = _merge_embedded_image_metadata(source_metadata, embedded_metadata)
    _write_png_with_embedded_metadata(source_payload, dest_path, merged_metadata)
    return {
        "image_ref": None,
        "image_url": f"/api/snapshots/{snapshot_id}/images/{image_name}",
    }


# Copies all snapshot-related images into snapshot-owned storage and rewrites the saved state.
def _copy_snapshot_images_into_state(snapshot_id: str, state: dict) -> dict:
    snapshot_state = json.loads(json.dumps(state))

    def apply_copy(target: dict, ref_key: str, url_key: str, slot_name: str) -> None:
        if not isinstance(target, dict):
            return
        copied = _copy_snapshot_image(
            snapshot_id=snapshot_id,
            slot_name=slot_name,
            image_ref=target.get(ref_key),
            image_url=target.get(url_key),
        )
        if not copied:
            return
        target[ref_key] = copied["image_ref"]
        target[url_key] = copied["image_url"]

    fisheye = snapshot_state.get("fisheye")
    if isinstance(fisheye, dict):
        apply_copy(fisheye, "current_image_ref", "current_image_url", "current")
        apply_copy(fisheye, "edit_source_ref", "edit_source_url", "original")

    current_entry = snapshot_state.get("current_output_entry")
    if isinstance(current_entry, dict):
        apply_copy(current_entry, "fisheye_image_ref", "fisheye_image_url", "current-output")

    history = snapshot_state.get("output_history")
    if isinstance(history, list):
        for idx, item in enumerate(history):
            if isinstance(item, dict):
                apply_copy(item, "fisheye_image_ref", "fisheye_image_url", f"history-{idx:03d}")

    return snapshot_state


# Rewrites snapshot-owned images so each one embeds the final snapshot state payload.
def _embed_snapshot_state_into_snapshot_images(snapshot_id: str, snapshot_name: str, snapshot_state: dict) -> None:
    snapshot_dir = os.path.join(SNAPSHOT_IMAGE_DIR, (snapshot_id or "").strip())
    if not os.path.isdir(snapshot_dir):
        return
    for image_name in os.listdir(snapshot_dir):
        file_path = os.path.join(snapshot_dir, image_name)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, "rb") as f:
                payload = f.read()
            existing_metadata = _extract_embedded_image_metadata_from_bytes(payload)
            merged_metadata = _merge_embedded_image_metadata(
                existing_metadata,
                {
                    "kind": "snapshot_image",
                    "snapshot_id": snapshot_id,
                    "snapshot_name": snapshot_name,
                    "slot_name": os.path.splitext(image_name)[0],
                    "snapshot_state": snapshot_state,
                },
            )
            _write_png_with_embedded_metadata(payload, file_path, merged_metadata)
        except Exception:
            continue


# Encodes a local image file as a data URI for upstream image-edit requests.
def _build_data_uri_from_file(file_path: str) -> str:
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# Resolves the best available URI for an image-edit input image.
def _resolve_fisheye_input_image_uri(
    image_ref: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Optional[str]:
    ref = (image_ref or "").strip() or (_coerce_fisheye_ref_from_url(image_url) or "")
    if ref:
        record = _find_fisheye_image_record(ref)
        if record:
            source_url = str(record.get("source_url") or "").strip()
            if source_url.startswith(("http://", "https://")):
                return source_url

    local_file = _resolve_local_image_file(image_ref=image_ref, image_url=image_url)
    if local_file:
        return _build_data_uri_from_file(local_file)

    raw_url = (image_url or "").strip()
    if raw_url.startswith(("http://", "https://")):
        return raw_url
    return None


# Extracts a single image URL from a Replicate output payload.
def _extract_replicate_image_url(output: object) -> Optional[str]:
    if isinstance(output, str) and output.strip():
        return output.strip()
    if isinstance(output, list):
        for item in output:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip()
    if isinstance(output, dict):
        url = output.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


# Clamps a numeric value into the inclusive 0 to 1 range.
def _clamp01(value: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = 0.5
    return max(0.0, min(1.0, n))


# Normalizes fisheye game fish payloads into the stored game format.
def _normalize_game_fish(items: list[FisheyeGameFishPayload]) -> list[dict]:
    out: list[dict] = []
    for i, item in enumerate(items):
        label = (item.label or "").strip() or f"fish {i + 1}"
        out.append(
            {
                "label": label[:48],
                "x_frac": _clamp01(item.x_frac),
                "y_frac": _clamp01(item.y_frac),
            }
        )
    return out


# Reads persisted fisheye game rounds from disk.
def _read_game_rounds() -> list[dict]:
    return _read_json_list(FISHEYE_GAME_ROUNDS_PATH)


# Writes fisheye game rounds back to disk.
def _write_game_rounds(items: list[dict]) -> None:
    _write_json_list(FISHEYE_GAME_ROUNDS_PATH, items)


# Generates a new unique round code for the fisheye game.
def _new_round_code(existing_codes: set[str]) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for _ in range(64):
        code = "".join(alphabet[ord(os.urandom(1)) % len(alphabet)] for _ in range(6))
        if code not in existing_codes:
            return code
    return uuid.uuid4().hex[:6].upper()


# Computes the best average distance between target fish and guessed fish positions.
def _best_avg_distance(target_fish: list[dict], guess_fish: list[dict]) -> float:
    n = len(target_fish)
    if n <= 0 or len(guess_fish) != n:
        return 1.0
    if n == 1:
        dx = float(target_fish[0]["x_frac"]) - float(guess_fish[0]["x_frac"])
        dy = float(target_fish[0]["y_frac"]) - float(guess_fish[0]["y_frac"])
        return (dx * dx + dy * dy) ** 0.5
    if n == 2:
        dx0 = float(target_fish[0]["x_frac"]) - float(guess_fish[0]["x_frac"])
        dy0 = float(target_fish[0]["y_frac"]) - float(guess_fish[0]["y_frac"])
        dx1 = float(target_fish[1]["x_frac"]) - float(guess_fish[1]["x_frac"])
        dy1 = float(target_fish[1]["y_frac"]) - float(guess_fish[1]["y_frac"])
        d_a = ((dx0 * dx0 + dy0 * dy0) ** 0.5 + (dx1 * dx1 + dy1 * dy1) ** 0.5) / 2.0

        dx2 = float(target_fish[0]["x_frac"]) - float(guess_fish[1]["x_frac"])
        dy2 = float(target_fish[0]["y_frac"]) - float(guess_fish[1]["y_frac"])
        dx3 = float(target_fish[1]["x_frac"]) - float(guess_fish[0]["x_frac"])
        dy3 = float(target_fish[1]["y_frac"]) - float(guess_fish[0]["y_frac"])
        d_b = ((dx2 * dx2 + dy2 * dy2) ** 0.5 + (dx3 * dx3 + dy3 * dy3) ** 0.5) / 2.0
        return min(d_a, d_b)
    return 1.0


# Checks whether a fish x-position falls inside the central deadzone.
def _game_in_deadzone(x_frac: float, deadzone_half: float = 0.10) -> bool:
    x = _clamp01(x_frac)
    half = max(0.0, min(0.49, float(deadzone_half)))
    return abs(x - 0.5) <= half


# Computes how strongly a fish should lean based on its x-position.
def _game_lean_intensity(x_frac: float, deadzone_half: float = 0.10) -> float:
    x = _clamp01(x_frac)
    half = max(0.0, min(0.49, float(deadzone_half)))
    min_pct = 0.01
    if _game_in_deadzone(x, half):
        return 0.0
    if x < 0.5:
        raw = (0.5 - half - x) / max(0.0001, (0.5 - half))
    else:
        raw = (x - (0.5 + half)) / max(0.0001, (0.5 - half))
    normalized = max(0.0, min(1.0, raw))
    eased = normalized ** 1.8
    return min_pct + (1.0 - min_pct) * eased


# Builds the generated prompt text used for a fisheye game round.
def _build_fisheye_game_generated_prompt(
    thought: str,
    fish_items: list[dict],
    direction_1: str,
    direction_2: str,
    keep_brief: bool,
    base_preamble: str,
    base_closing: str,
) -> str:
    active_fish = [x for x in fish_items if (1.0 - float(x.get("y_frac", 1.0))) > 0.01]
    if not active_fish:
        return (
            'Here is the original thought:\n\n"""\n'
            + thought
            + '\n"""\n\nNo active transformation concepts (all at zero weight).'
        )

    lines: list[str] = []
    for fish in active_fish:
        label = str(fish.get("label") or "fish").strip() or "fish"
        x = _clamp01(float(fish.get("x_frac", 0.5)))
        y = _clamp01(float(fish.get("y_frac", 0.5)))
        weight = 1.0 - y
        lean_desc = ""
        if not _game_in_deadzone(x):
            if x < 0.5 and direction_1:
                lean_desc = f'leans {round(_game_lean_intensity(x) * 100)}% toward "{direction_1}"'
            elif x > 0.5 and direction_2:
                lean_desc = f'leans {round(_game_lean_intensity(x) * 100)}% toward "{direction_2}"'
        line = f"{label}: weight {weight:.2f}"
        if lean_desc:
            line += ", " + lean_desc
        lines.append(line)

    brief = "\nKeep your answer extremely brief (e.g., less than 20 words)." if keep_brief else ""
    return (
        'Here is the original thought:\n\n"""\n'
        + thought
        + '\n"""\n\n'
        + base_preamble.strip()
        + "\n\n"
        + "\n".join(lines)
        + "\n\n"
        + base_closing.strip()
        + brief
    )


# Runs a single non-streaming text generation request.
def _generate_text_once(prompt: str, model: str) -> str:
    client, actual_model = _build_model_client(model or DEFAULT_MODEL)
    deterministic, seed = _resolve_text_determinism()
    response = _chat_completion_create(
        client=client,
        deterministic=deterministic,
        seed=seed,
        model=actual_model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    if not response or not response.choices:
        raise RuntimeError("Model returned no choices")
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        out = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            txt = getattr(item, "text", None) if item is not None else None
            if isinstance(txt, str):
                parts.append(txt)
        out = "".join(parts).strip()
    else:
        out = ""
    if not out:
        raise RuntimeError("Model returned empty content")
    return out


# Builds per-model payloads from fisheye selector values (model + optional size token).
def _build_replicate_input(
    prompt: str,
    model_slug: str,
    image_seed: Optional[int] = None,
    input_image_uri: Optional[str] = None,
    prompt_strength: Optional[float] = None,
) -> dict[str, object]:
    raw = model_slug.strip()
    model = raw.lower()
    size_token = ""
    image_size = 512
    if "|" in raw:
        base, size = raw.split("|", 1)
        model = base.strip().lower()
        size_token = size.strip()
        try:
            parsed_size = int(size_token)
            image_size = max(256, min(2048, parsed_size))
        except (TypeError, ValueError):
            image_size = 512

    if model == "bytedance/seedream-4":
        seedream_size = (size_token or "2K").upper()
        if seedream_size not in {"1K", "2K", "4K"}:
            seedream_size = "2K"
        return {
            "prompt": prompt,
            "size": seedream_size,
            "aspect_ratio": "1:1",
            "enhance_prompt": True,
            "sequential_image_generation": "disabled",
            "max_images": 1,
        }

    if model == "google/imagen-3-fast":
        return {
            "prompt": prompt,
            "image_size": image_size,
            "aspect_ratio": "1:1",
            "output_format": "jpg",
            "safety_filter_level": "block_medium_and_above",
        }

    if model == "google/imagen-4-fast":
        return {
            "prompt": prompt,
            "aspect_ratio": "1:1",
            "output_format": "jpg",
            "safety_filter_level": "block_medium_and_above",
        }

    if model.startswith("sdxl-based/epicrealismxl-lightning-hades"):
        payload = {
            "prompt": prompt,
            "negative_prompt": "",
            "width": 1024,
            "height": 1024,
            "num_outputs": 1,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "scheduler": "K_EULER",
        }
        if isinstance(image_seed, int):
            payload["seed"] = max(1, min(999999, image_seed))
        return payload

    if model.startswith("bytedance/sdxl-lightning-4step"):
        payload = {
            "prompt": prompt,
            "negative_prompt": "",
            "width": 1024,
            "height": 1024,
            "num_outputs": 1,
            "num_inference_steps": 4,
            "guidance_scale": 0,
            "scheduler": "K_EULER",
        }
        if isinstance(image_seed, int):
            payload["seed"] = max(1, min(999999, image_seed))
        return payload

    if model.startswith("lucataco/ssd-1b"):
        strength_value = 0.8
        if isinstance(prompt_strength, (int, float)):
            strength_value = max(0.0, min(1.0, float(prompt_strength)))
        payload = {
            "prompt": prompt,
            "negative_prompt": "low quality image",
            "num_outputs": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 25,
        }
        if isinstance(image_seed, int):
            payload["seed"] = max(1, min(999999, image_seed))
        if input_image_uri:
            payload["image"] = input_image_uri
            payload["prompt_strength"] = strength_value
        return payload

    if model == "google/nano-banana":
        payload = {
            "prompt": prompt,
            "output_format": "jpg",
        }
        if input_image_uri:
            payload["image_input"] = [input_image_uri]
            payload["aspect_ratio"] = "match_input_image"
        else:
            payload["aspect_ratio"] = "1:1"
        return payload

    if model == "google/nano-banana-2":
        nano_resolution = (size_token or "1K").upper()
        if nano_resolution not in {"1K", "2K", "4K"}:
            nano_resolution = "1K"
        payload = {
            "prompt": prompt,
            "resolution": nano_resolution,
            "output_format": "jpg",
        }
        if input_image_uri:
            payload["image_input"] = [input_image_uri]
            payload["aspect_ratio"] = "match_input_image"
        else:
            payload["aspect_ratio"] = "1:1"
        return payload

    # Default profile: prunaai/flux-fast
    seed_value = 2110
    if isinstance(image_seed, int):
        seed_value = max(1, min(999999, image_seed))

    return {
        "seed": seed_value,
        "prompt": prompt,
        "guidance": 3.5,
        "image_size": image_size,
        "speed_mode": "Extra Juiced 🔥 (more speed)",
        "aspect_ratio": "1:1",
        "output_format": "jpg",
        "output_quality": 80,
        "num_inference_steps": 28,
    }


# Generates an image through Replicate and stores the returned result.
def _replicate_generate_image(
    prompt: str,
    model_slug: str,
    image_seed: Optional[int] = None,
    input_image_uri: Optional[str] = None,
    prompt_strength: Optional[float] = None,
) -> str:
    replicate_input = _build_replicate_input(
        prompt,
        model_slug,
        image_seed,
        input_image_uri,
        prompt_strength,
    )
    endpoint_model = model_slug.split("|", 1)[0].strip()

    prediction: Optional[dict] = None
    create_errors: list[str] = []

    # If a version is provided (owner/name:version_hash), use /v1/predictions with version hash.
    if ":" in endpoint_model:
        owner_model, version = endpoint_model.split(":", 1)
        owner_model = owner_model.strip()
        version = version.strip()
        if "/" not in owner_model or not version:
            raise ValueError(
                "Replicate model format must be owner/name:version_hash when using versioned mode"
            )
        try:
            prediction = _replicate_http_json(
                f"{REPLICATE_API_BASE}/predictions",
                method="POST",
                payload={"version": version, "input": replicate_input},
                prefer_wait=True,
            )
        except Exception as exc:
            create_errors.append(str(exc))
    else:
        if "/" not in endpoint_model:
            raise ValueError("Replicate model must be owner/name or owner/name:version_hash")
        # Non-versioned model path.
        try:
            prediction = _replicate_http_json(
                f"{REPLICATE_API_BASE}/models/{endpoint_model}/predictions",
                method="POST",
                payload={"input": replicate_input},
                prefer_wait=True,
            )
        except Exception as exc:
            create_errors.append(str(exc))

    if prediction is None:
        raise RuntimeError(" ; ".join(create_errors) or "Unable to create Replicate prediction")

    status = str(prediction.get("status") or "").lower()
    get_url = None
    urls = prediction.get("urls")
    if isinstance(urls, dict):
        val = urls.get("get")
        if isinstance(val, str) and val.strip():
            get_url = val.strip()

    if status != "succeeded" and get_url:
        for _ in range(20):
            time.sleep(1.0)
            prediction = _replicate_http_json(get_url, method="GET")
            status = str(prediction.get("status") or "").lower()
            if status in ("succeeded", "failed", "canceled"):
                break

    if status != "succeeded":
        err = prediction.get("error")
        raise RuntimeError(f"Replicate prediction status={status or 'unknown'} error={err}")

    image_url = _extract_replicate_image_url(prediction.get("output"))
    if not image_url:
        raise RuntimeError("Replicate did not return an image URL in output")
    return image_url


_ensure_api_counter_file()


# Streams a text generation response back to the frontend.
@app.post("/api/generate")
async def generate(request: Request):
    # Main text generation endpoint: streams tokens so UI can render progressively.
    _increment_api_call_count()
    body = await request.json()
    prompt = body.get("prompt", "")
    model = body.get("model") or DEFAULT_MODEL
    deterministic, seed = _resolve_text_determinism(
        deterministic_override=body.get("deterministic"),
        seed_override=body.get("seed"),
    )

    # 🔁 Dynamic provider routing
    client, actual_model = _build_model_client(model)

    async def stream_tokens():
        try:
            stream = _chat_completion_create(
                client=client,
                deterministic=deterministic,
                seed=seed,
                model=actual_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                first = choices[0]
                delta = getattr(first, "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                if not isinstance(text, str) or not text:
                    continue

                # Escape for SSE
                safe = text.replace("\\", "\\\\").replace("\n", "\\n")
                yield f"data: {safe}\n\n"

            yield "data: [DONE]\n\n"

        except Exception:
            err_detail = traceback.format_exc()
            print(f"[api/generate] upstream error:\n{err_detail}")
            err_msg = str(err_detail.splitlines()[-1]).strip() or "Unknown model error"
            safe_err = err_msg.replace("\\", "\\\\").replace("\n", "\\n")
            yield f"data: [ERROR] {safe_err}\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


# Generates a fisheye image for the current output text and returns its stored metadata.
@app.post("/api/visualise")
def visualise(payload: VisualisePayload):
    # Fisheye endpoint: turns current output text into an image and stores local reference.
    # Reload .env so newly-saved keys are picked up without process restart.
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    output_text = (payload.output_text or "").strip()
    if not output_text:
        return {"ok": False, "message": "output_text is required"}

    image_model = (payload.image_model or DEFAULT_REPLICATE_MODEL).strip()
    if not image_model:
        image_model = DEFAULT_REPLICATE_MODEL
    image_seed = payload.image_seed
    if isinstance(image_seed, int):
        image_seed = max(1, min(999999, image_seed))
    else:
        image_seed = None
    prompt_strength = payload.prompt_strength
    if isinstance(prompt_strength, (int, float)):
        prompt_strength = max(0.0, min(1.0, float(prompt_strength)))
    else:
        prompt_strength = None
    creation_state = _sanitize_embedded_image_metadata(payload.creation_state)
    use_input_image = bool(payload.use_input_image)
    input_image_uri = None
    endpoint_model = image_model.split("|", 1)[0].strip().lower()
    if use_input_image and (
        endpoint_model in {"google/nano-banana", "google/nano-banana-2"}
        or endpoint_model.startswith("lucataco/ssd-1b")
    ):
        input_image_uri = _resolve_fisheye_input_image_uri(
            image_ref=payload.input_image_ref,
            image_url=payload.input_image_url,
        )
        if not input_image_uri:
            return {
                "ok": False,
                "message": "Nano Banana edit mode requires a saved image input",
            }

    # Use current output text directly as the image-generation prompt.
    image_prompt = output_text

    try:
        _increment_api_call_count()
        image_url = _replicate_generate_image(
            image_prompt,
            image_model,
            image_seed,
            input_image_uri=input_image_uri,
            prompt_strength=prompt_strength,
        )
        image_ref, local_image_url = _store_fisheye_image(
            image_url,
            embedded_metadata={
                "kind": "fisheye_generated_image",
                "output_text": output_text,
                "prompt_model": (payload.prompt_model or "").strip(),
                "image_model": image_model,
                "image_seed": image_seed,
                "prompt_strength": prompt_strength,
                "use_input_image": use_input_image,
                "input_image_ref": (payload.input_image_ref or "").strip() or None,
                "input_image_url": (payload.input_image_url or "").strip() or None,
                "source_image_url": image_url,
                "creation_state": creation_state,
            },
        )
    except Exception as exc:
        return {"ok": False, "message": f"Failed to generate image via Replicate: {exc}"}

    return {
        "ok": True,
        "image_prompt": image_prompt,
        "image_url": local_image_url,
        "source_image_url": image_url,
        "image_ref": image_ref,
        "image_model": image_model,
        "prompt_model": None,
    }


# Imports an existing image into fisheye storage so the frontend can reference it later.
@app.post("/api/fisheye-images/import")
def import_fisheye_image(payload: FisheyeImageImportPayload):
    image_ref = (payload.image_ref or "").strip()
    if image_ref:
        record = _find_fisheye_image_record(image_ref)
        file_path = _local_fisheye_file_path(image_ref)
        embedded_metadata = _read_embedded_image_metadata(file_path) if file_path else None
        if record and file_path:
            return {
                "ok": True,
                "image_ref": image_ref,
                "image_url": f"/api/fisheye-images/{image_ref}",
                "source_url": str(record.get("source_url") or "").strip(),
                "embedded_metadata": embedded_metadata,
                "embedded_state": _extract_restorable_state_from_embedded_metadata(embedded_metadata),
            }

    source_url = (payload.source_url or "").strip()
    inferred_ref = _coerce_fisheye_ref_from_url(source_url)
    if inferred_ref:
        record = _find_fisheye_image_record(inferred_ref)
        file_path = _local_fisheye_file_path(inferred_ref)
        embedded_metadata = _read_embedded_image_metadata(file_path) if file_path else None
        if record and file_path:
            return {
                "ok": True,
                "image_ref": inferred_ref,
                "image_url": f"/api/fisheye-images/{inferred_ref}",
                "source_url": str(record.get("source_url") or "").strip(),
                "embedded_metadata": embedded_metadata,
                "embedded_state": _extract_restorable_state_from_embedded_metadata(embedded_metadata),
            }

    if not source_url:
        return {"ok": False, "message": "image_ref or source_url is required"}

    try:
        image_ref, local_image_url = _store_fisheye_image(source_url)
    except Exception as exc:
        return {"ok": False, "message": f"Failed to store fisheye image: {exc}"}

    record = _find_fisheye_image_record(image_ref) or {}
    file_path = _local_fisheye_file_path(image_ref)
    embedded_metadata = _read_embedded_image_metadata(file_path) if file_path else None
    return {
        "ok": True,
        "image_ref": image_ref,
        "image_url": local_image_url,
        "source_url": str(record.get("source_url") or "").strip(),
        "embedded_metadata": embedded_metadata,
        "embedded_state": _extract_restorable_state_from_embedded_metadata(embedded_metadata),
    }


# Imports an uploaded image file into fisheye storage and returns any embedded state.
@app.post("/api/fisheye-images/upload")
async def upload_fisheye_image(file: UploadFile = File(...)):
    filename = os.path.basename((file.filename or "").strip()) or "upload"
    content_type = str(file.content_type or "").strip()
    try:
        payload = await file.read()
    except Exception as exc:
        return {"ok": False, "message": f"Failed to read uploaded file: {exc}"}

    if not payload:
        return {"ok": False, "message": "Uploaded image is empty"}

    try:
        image_ref, local_image_url = _store_fisheye_image_payload(
            payload=payload,
            content_type=content_type,
            source_url=filename,
        )
    except Exception as exc:
        return {"ok": False, "message": f"Failed to store uploaded image: {exc}"}

    record = _find_fisheye_image_record(image_ref) or {}
    file_path = _local_fisheye_file_path(image_ref)
    embedded_metadata = _read_embedded_image_metadata(file_path) if file_path else None
    return {
        "ok": True,
        "image_ref": image_ref,
        "image_url": local_image_url,
        "source_url": str(record.get("source_url") or filename).strip(),
        "embedded_metadata": embedded_metadata,
        "embedded_state": _extract_restorable_state_from_embedded_metadata(embedded_metadata),
    }


# Returns the list of saved snapshots.
@app.get("/api/snapshots")
def list_snapshots():
    return {"ok": True, "snapshots": _list_snapshots()}


# Returns one saved snapshot record by id.
@app.get("/api/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str):
    record = _read_snapshot(snapshot_id)
    if not record:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {
        "ok": True,
        "snapshot": {
            "id": str(record.get("id") or snapshot_id).strip(),
            "name": str(record.get("name") or snapshot_id).strip(),
            "created_at": int(record.get("created_at") or 0),
            "state": record.get("state") if isinstance(record.get("state"), dict) else {},
        },
    }


# Serves an image file owned by a stored snapshot.
@app.get("/api/snapshots/{snapshot_id}/images/{image_name}")
def get_snapshot_image(snapshot_id: str, image_name: str):
    file_path = _snapshot_image_path(snapshot_id, image_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Snapshot image not found")
    media_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return FileResponse(path=file_path, media_type=media_type, filename=os.path.basename(file_path))


# Creates a snapshot record and copies any referenced images into snapshot storage.
@app.post("/api/snapshots")
def create_snapshot(payload: SnapshotCreatePayload):
    if not isinstance(payload.state, dict):
        return {"ok": False, "message": "state is required"}

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_IMAGE_DIR, exist_ok=True)
    snapshot_id = uuid.uuid4().hex[:16]
    created_at = int(time.time())
    provided_name = (payload.name or "").strip()
    display_name = provided_name or time.strftime("Snapshot %Y-%m-%d %H:%M:%S", time.localtime(created_at))
    snapshot_state = _copy_snapshot_images_into_state(snapshot_id, payload.state)
    _embed_snapshot_state_into_snapshot_images(snapshot_id, display_name, snapshot_state)
    record = {
        "id": snapshot_id,
        "name": display_name,
        "created_at": created_at,
        "state": snapshot_state,
    }
    _write_json(_snapshot_path(snapshot_id), record)
    return {"ok": True, "snapshot": _snapshot_summary(record)}


# Builds a zip of snapshot current-output images, optionally filtered by day.
@app.get("/api/snapshots-current-output-images.zip")
def download_snapshot_current_output_images_zip(day: Optional[str] = None):
    snapshots = _list_snapshots()
    day_filter = (day or "").strip()
    if day_filter:
        snapshots = [
            item
            for item in snapshots
            if time.strftime("%Y-%m-%d", time.localtime(int(item.get("created_at") or 0))) == day_filter
        ]
    archive_buffer = io.BytesIO()

    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        added = 0
        for item in snapshots:
            snapshot_id = str(item.get("id") or "").strip()
            if not snapshot_id:
                continue
            base_dir = os.path.join(SNAPSHOT_IMAGE_DIR, snapshot_id)
            if not os.path.isdir(base_dir):
                continue

            current_output_name = None
            for name in sorted(os.listdir(base_dir)):
                if name.startswith("current-output."):
                    current_output_name = name
                    break
            if not current_output_name:
                continue

            file_path = os.path.join(base_dir, current_output_name)
            if not os.path.exists(file_path):
                continue

            snapshot_name = str(item.get("name") or snapshot_id).strip() or snapshot_id
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", " ") else "_" for ch in snapshot_name).strip()
            safe_name = safe_name or snapshot_id
            ext = os.path.splitext(current_output_name)[1].lower() or ".jpg"
            arcname = f"{snapshot_id} - {safe_name}{ext}"
            zf.write(file_path, arcname=arcname)
            added += 1

        if added == 0:
            zf.writestr("README.txt", "No snapshot current-output images were found.\n")

    archive_buffer.seek(0)
    return StreamingResponse(
        archive_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="snapshot-current-output-images.zip"'},
    )


# Creates and stores a new fisheye game round.
@app.post("/api/fisheye-game/rounds")
def create_fisheye_game_round(payload: FisheyeGameRoundCreatePayload):
    # Creates one multiplayer round and generates its reference image once.
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    thought = (payload.thought or "").strip()
    decoy = (payload.decoy_thought or "").strip()
    if not thought or not decoy:
        return {"ok": False, "message": "thought and decoy_thought are required"}
    if thought.lower() == decoy.lower():
        return {"ok": False, "message": "decoy_thought must differ from thought"}

    fish_items = _normalize_game_fish(payload.fish or [])
    if len(fish_items) < 1 or len(fish_items) > 2:
        return {"ok": False, "message": "fish must contain 1 or 2 items"}

    direction_1 = (payload.direction_1 or "").strip()
    direction_2 = (payload.direction_2 or "").strip()
    keep_brief = bool(payload.keep_brief) if payload.keep_brief is not None else True
    prompt_model = (payload.prompt_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base_preamble = (payload.base_preamble or DEFAULT_BASE_PREAMBLE).strip() or DEFAULT_BASE_PREAMBLE
    base_closing = (payload.base_closing or DEFAULT_BASE_CLOSING).strip() or DEFAULT_BASE_CLOSING

    generated_prompt = _build_fisheye_game_generated_prompt(
        thought=thought,
        fish_items=fish_items,
        direction_1=direction_1,
        direction_2=direction_2,
        keep_brief=keep_brief,
        base_preamble=base_preamble,
        base_closing=base_closing,
    )

    try:
        _increment_api_call_count()
        image_prompt = _generate_text_once(generated_prompt, prompt_model)
    except Exception as exc:
        return {"ok": False, "message": f"Failed to generate image prompt via model: {exc}"}
    image_model = (payload.image_model or DEFAULT_REPLICATE_MODEL).strip() or DEFAULT_REPLICATE_MODEL
    image_seed = payload.image_seed
    if isinstance(image_seed, int):
        image_seed = max(1, min(999999, image_seed))
    else:
        image_seed = 2110

    try:
        _increment_api_call_count()
        source_url = _replicate_generate_image(image_prompt, image_model, image_seed)
        image_ref, local_image_url = _store_fisheye_image(source_url)
    except Exception as exc:
        return {"ok": False, "message": f"Failed to generate game image via Replicate: {exc}"}

    options = [thought, decoy]
    # Randomize display order once so all guessers see identical choice order for the round.
    if ord(os.urandom(1)) % 2 == 1:
        options = [decoy, thought]
    correct_index = 0 if options[0] == thought else 1

    with FISHEYE_GAME_LOCK:
        rounds = _read_game_rounds()
        existing = {str(r.get("code", "")).upper() for r in rounds}
        code = _new_round_code(existing)
        now_ts = int(time.time())
        record = {
            "code": code,
            "created_at": now_ts,
            "thought": thought,
            "decoy_thought": decoy,
            "thought_options": options,
            "correct_thought_index": correct_index,
            "fish": fish_items,
            "direction_1": direction_1,
            "direction_2": direction_2,
            "keep_brief": keep_brief,
            "base_preamble": base_preamble,
            "base_closing": base_closing,
            "prompt_model": prompt_model,
            "generated_prompt": generated_prompt,
            "image_prompt": image_prompt,
            "image_model": image_model,
            "image_ref": image_ref,
            "image_url": local_image_url,
            "guesses": [],
        }
        rounds.append(record)
        if len(rounds) > 200:
            rounds = rounds[-200:]
        _write_game_rounds(rounds)

    return {
        "ok": True,
        "round_code": code,
        "image_url": local_image_url,
        "join_url": f"/paragraph-adapter-game.html?round={code}",
        "thought_options": options,
        "fish_count": len(fish_items),
        "prompt_model": prompt_model,
        "generated_prompt": generated_prompt,
        "image_prompt": image_prompt,
    }


# Returns one fisheye game round for play.
@app.get("/api/fisheye-game/rounds/{round_code}")
def get_fisheye_game_round(round_code: str):
    code = (round_code or "").strip().upper()
    if not code:
        return {"ok": False, "message": "round_code is required"}

    with FISHEYE_GAME_LOCK:
        rounds = _read_game_rounds()
        match = next((r for r in rounds if str(r.get("code", "")).upper() == code), None)

    if not match:
        return {"ok": False, "message": "Round not found"}

    return {
        "ok": True,
        "round_code": code,
        "created_at": match.get("created_at"),
        "image_url": match.get("image_url"),
        "fish_count": len(match.get("fish", []) or []),
        "thought_options": list(match.get("thought_options", [])),
        "guess_count": len(match.get("guesses", []) or []),
    }


# Scores a fisheye game guess and returns round feedback.
@app.post("/api/fisheye-game/rounds/{round_code}/guess")
def submit_fisheye_game_guess(round_code: str, payload: FisheyeGameGuessPayload):
    code = (round_code or "").strip().upper()
    if not code:
        return {"ok": False, "message": "round_code is required"}

    with FISHEYE_GAME_LOCK:
        rounds = _read_game_rounds()
        match = next((r for r in rounds if str(r.get("code", "")).upper() == code), None)
        if not match:
            return {"ok": False, "message": "Round not found"}

        options = list(match.get("thought_options", []))
        target_fish = list(match.get("fish", []))
        if payload.thought_index < 0 or payload.thought_index >= len(options):
            return {"ok": False, "message": "Invalid thought_index"}

        guessed_fish = _normalize_game_fish(payload.fish or [])
        if len(guessed_fish) != len(target_fish):
            return {"ok": False, "message": f"Expected {len(target_fish)} fish guesses"}

        thought_correct = payload.thought_index == int(match.get("correct_thought_index", 0))
        avg_distance = _best_avg_distance(target_fish, guessed_fish)
        position_score = max(0, min(100, round((1.0 - min(1.0, avg_distance / 0.85)) * 100)))
        thought_score = 50 if thought_correct else 0
        total_score = thought_score + round(position_score * 0.5)

        guess_record = {
            "created_at": int(time.time()),
            "thought_index": int(payload.thought_index),
            "thought_correct": thought_correct,
            "fish_guess": guessed_fish,
            "avg_distance": avg_distance,
            "position_score": position_score,
            "total_score": total_score,
        }
        guesses = list(match.get("guesses", []))
        guesses.append(guess_record)
        match["guesses"] = guesses[-50:]
        _write_game_rounds(rounds)

    return {
        "ok": True,
        "round_code": code,
        "thought_correct": thought_correct,
        "correct_thought_index": int(match.get("correct_thought_index", 0)),
        "correct_thought": options[int(match.get("correct_thought_index", 0))] if options else "",
        "avg_distance": avg_distance,
        "position_score": position_score,
        "total_score": total_score,
        "target_fish": target_fish,
        "guess_fish": guessed_fish,
    }


# Creates a new Daydream stream and seeds its local file state.
@app.post("/api/daydream/stream/create")
def daydream_create_stream(payload: DaydreamCreatePayload):
    # Daydream has route variance; this endpoint probes and stores working route hints.
    prompt = (payload.prompt or "").strip()
    if not prompt:
        return {"ok": False, "message": "prompt is required"}

    model_id = (payload.model_id or "stabilityai/sdxl-turbo").strip() or "stabilityai/sdxl-turbo"
    request_payload = {
        "pipeline": "streamdiffusion",
        "params": {
            "model_id": model_id,
            "prompt": prompt,
            # Keep stream startup lightweight for faster bring-up during testing.
            "controlnets": [],
        },
    }

    attempts: list[dict] = []
    errors: list[str] = []
    for path in ("/v1/streams", "/api/v1/streams"):
        data, attempt = _daydream_attempt("POST", path, request_payload, timeout=120)
        attempts.append(attempt)
        if data is not None:
            stream_id = _extract_daydream_stream_id(data)
            return {
                "ok": True,
                "stream_id": stream_id,
                "raw": data,
                "debug": {
                    "api_base": (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE),
                    "attempts": attempts,
                },
            }
        if "error" in attempt:
            errors.append(str(attempt["error"]))

    return {
        "ok": False,
        "message": " ; ".join(errors) or "Failed to create Daydream stream",
        "debug": {
            "api_base": (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE),
            "attempts": attempts,
        },
    }


# Sends an update to an existing Daydream stream.
@app.post("/api/daydream/stream/update")
def daydream_update_stream(payload: DaydreamUpdatePayload):
    # Reuses discovered Daydream route hints where possible to reduce failed attempts.
    stream_id = (payload.stream_id or "").strip()
    if not stream_id:
        return {"ok": False, "message": "stream_id is required"}

    params: dict[str, object] = {}
    model_id = (payload.model_id or "").strip()
    if model_id:
        params["model_id"] = model_id
    if payload.prompt is not None:
        params["prompt"] = payload.prompt
    if not params:
        return {"ok": False, "message": "at least one of prompt or model_id is required"}
    request_payload = {"pipeline": "streamdiffusion", "params": params}

    attempts = [
        ("POST", f"/v1/streams/{stream_id}/update"),
        ("PATCH", f"/v1/streams/{stream_id}"),
        ("POST", f"/v1/streams/{stream_id}"),
        ("POST", f"/api/v1/streams/{stream_id}/update"),
        ("PATCH", f"/api/v1/streams/{stream_id}"),
    ]
    hint = DAYDREAM_ROUTE_HINTS.get("update")
    ordered_attempts: list[tuple[str, str]] = []
    if hint:
        hint_method, hint_tpl = hint
        hint_path = hint_tpl.replace("{id}", stream_id)
        ordered_attempts.append((hint_method, hint_path))
    for item in attempts:
        if item not in ordered_attempts:
            ordered_attempts.append(item)

    attempts_log: list[dict] = []
    errors: list[str] = []
    for method, path in ordered_attempts:
        data, attempt = _daydream_attempt(method, path, request_payload, timeout=25)
        attempts_log.append(attempt)
        if data is not None:
            DAYDREAM_ROUTE_HINTS["update"] = (method, _path_to_template(path, stream_id))
            return {
                "ok": True,
                "stream_id": stream_id,
                "raw": data,
                "debug": {
                    "api_base": (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE),
                    "route_hint": DAYDREAM_ROUTE_HINTS.get("update"),
                    "attempts": attempts_log,
                },
            }
        if "error" in attempt:
            errors.append(str(attempt["error"]))

    return {
        "ok": False,
        "message": " ; ".join(errors) or "Failed to update Daydream stream",
        "debug": {
            "api_base": (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE),
            "route_hint": DAYDREAM_ROUTE_HINTS.get("update"),
            "attempts": attempts_log,
        },
    }


# Returns current status and local file metadata for a Daydream stream.
@app.get("/api/daydream/stream/{stream_id}/status")
def daydream_stream_status(stream_id: str):
    # Poll endpoint used by UI to track stream state.
    stream = (stream_id or "").strip()
    if not stream:
        return {"ok": False, "message": "stream_id is required"}

    attempts_log: list[dict] = []
    paths = (
        f"/v1/streams/{stream}/status",
        f"/api/v1/streams/{stream}/status",
        f"/v1/streams/{stream}",
        f"/api/v1/streams/{stream}",
    )
    hint = DAYDREAM_ROUTE_HINTS.get("status")
    ordered_paths: list[str] = []
    if hint:
        _, hint_tpl = hint
        ordered_paths.append(hint_tpl.replace("{id}", stream))
    for path in paths:
        if path not in ordered_paths:
            ordered_paths.append(path)

    errors: list[str] = []
    for path in ordered_paths:
        data, attempt = _daydream_attempt("GET", path, None, timeout=90)
        attempts_log.append(attempt)
        if data is not None:
            DAYDREAM_ROUTE_HINTS["status"] = ("GET", _path_to_template(path, stream))
            return {
                "ok": True,
                "stream_id": stream,
                "raw": data,
                "debug": {
                    "api_base": (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE),
                    "route_hint": DAYDREAM_ROUTE_HINTS.get("status"),
                    "attempts": attempts_log,
                },
            }
        if "error" in attempt:
            errors.append(str(attempt["error"]))

    return {
        "ok": False,
        "message": " ; ".join(errors) or "Failed to fetch Daydream stream status",
        "debug": {
            "api_base": (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE),
            "route_hint": DAYDREAM_ROUTE_HINTS.get("status"),
            "attempts": attempts_log,
        },
    }


# Stops a Daydream stream and reports the final state.
@app.post("/api/daydream/stream/stop")
def daydream_stop_stream(payload: DaydreamStopPayload):
    # Stop endpoint with fallback route probing for compatibility.
    stream_id = (payload.stream_id or "").strip()
    if not stream_id:
        return {"ok": False, "message": "stream_id is required"}

    attempts = [
        ("DELETE", f"/v1/streams/{stream_id}"),
        ("POST", f"/v1/streams/{stream_id}/stop"),
    ]
    hint = DAYDREAM_ROUTE_HINTS.get("stop")
    ordered_attempts: list[tuple[str, str]] = []
    if hint:
        hint_method, hint_tpl = hint
        hint_path = hint_tpl.replace("{id}", stream_id)
        ordered_attempts.append((hint_method, hint_path))
    for item in attempts:
        if item not in ordered_attempts:
            ordered_attempts.append(item)

    attempts_log: list[dict] = []
    errors: list[str] = []
    for method, path in ordered_attempts:
        data, attempt = _daydream_attempt(method, path, None, timeout=8)
        attempts_log.append(attempt)
        if data is not None:
            DAYDREAM_ROUTE_HINTS["stop"] = (method, _path_to_template(path, stream_id))
            return {
                "ok": True,
                "stream_id": stream_id,
                "raw": data,
                "debug": {
                    "api_base": (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE),
                    "route_hint": DAYDREAM_ROUTE_HINTS.get("stop"),
                    "attempts": attempts_log,
                },
            }
        if "error" in attempt:
            errors.append(str(attempt["error"]))

    return {
        "ok": False,
        "message": " ; ".join(errors) or "Failed to stop Daydream stream",
        "debug": {
            "api_base": (os.getenv("DAYDREAM_API_BASE") or DAYDREAM_API_BASE),
            "route_hint": DAYDREAM_ROUTE_HINTS.get("stop"),
            "attempts": attempts_log,
        },
    }


# Serves a stored fisheye image by ref.
@app.get("/api/fisheye-images/{image_ref}")
async def get_fisheye_image(image_ref: str):
    # Serves locally cached fisheye image files by short reference id.
    items = _read_json_list(FISHEYE_IMAGE_INDEX_PATH)
    match = next((x for x in items if str(x.get("ref", "")) == image_ref), None)
    if not match:
        raise HTTPException(status_code=404, detail="Image not found")

    filename = str(match.get("filename") or "").strip()
    if not filename:
        raise HTTPException(status_code=404, detail="Image record invalid")

    file_path = os.path.join(FISHEYE_IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file missing")

    media_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return FileResponse(path=file_path, media_type=media_type, filename=filename)


# Persists API keys from the UI into the local env file.
@app.post("/api/save-keys")
async def save_keys(payload: EnvKeysPayload):
    # Saves only provided keys; blank fields are intentionally ignored.
    global OPENROUTER_API_KEY, GROQ_API_KEY, REPLICATE_API_TOKEN

    updates: dict[str, str] = {}

    if payload.openrouter_api_key is not None and payload.openrouter_api_key.strip():
        value = payload.openrouter_api_key.strip()
        updates["OPENROUTER_API_KEY"] = value
        OPENROUTER_API_KEY = value

    if payload.groq_api_key is not None and payload.groq_api_key.strip():
        value = payload.groq_api_key.strip()
        updates["GROQ_API_KEY"] = value
        GROQ_API_KEY = value

    if payload.replicate_api_key is not None and payload.replicate_api_key.strip():
        value = payload.replicate_api_key.strip()
        updates["REPLICATE_API_TOKEN"] = value
        REPLICATE_API_TOKEN = value

    if not updates:
        return {"ok": False, "message": "No keys provided"}

    _upsert_env_values(ENV_PATH, updates)
    return {"ok": True, "message": "Saved keys to .env", "saved": list(updates.keys())}


# Returns the saved prompt template sections.
@app.get("/api/prompt-templates")
async def get_prompt_templates():
    return {
        "ok": True,
        "preamble": _read_text(PROMPT_PREAMBLE_PATH),
        "closing": _read_text(PROMPT_CLOSING_PATH),
    }


# Persists an updated prompt template section.
@app.post("/api/prompt-templates/save")
async def save_prompt_section(payload: PromptSectionPayload):
    section = (payload.section or "").strip().lower()
    content = payload.content or ""
    if section not in ("preamble", "closing"):
        return {"ok": False, "message": "section must be 'preamble' or 'closing'"}

    path = PROMPT_PREAMBLE_PATH if section == "preamble" else PROMPT_CLOSING_PATH
    _write_text(path, content)
    return {"ok": True, "saved": section}


# Resets the prompt template files back to defaults.
@app.post("/api/prompt-templates/reset")
async def reset_prompt_templates(payload: PromptTemplatesResetPayload):
    _write_text(PROMPT_PREAMBLE_PATH, payload.preamble or "")
    _write_text(PROMPT_CLOSING_PATH, payload.closing or "")
    return {"ok": True, "saved": ["preamble", "closing"]}


# Returns the current normalized app configuration.
@app.get("/api/app-config")
async def get_app_config():
    return {
        "ok": True,
        "config": _get_current_app_config(),
        "defaults": dict(DEFAULT_APP_CONFIG),
    }


# Validates and saves updated app configuration values.
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


# Resets the app configuration file back to defaults.
@app.post("/api/app-config/reset")
async def reset_app_config():
    _write_json(APP_CONFIG_PATH, dict(DEFAULT_APP_CONFIG))
    return {"ok": True, "config": dict(DEFAULT_APP_CONFIG)}


# Returns lightweight usage stats for the menu and UI.
@app.get("/api/stats")
async def get_stats():
    return {
        "ok": True,
        "total_api_calls": _read_api_call_count(),
    }


# Static files last
app.mount("/", StaticFiles(directory="public", html=True), name="static")
