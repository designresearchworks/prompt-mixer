import os
import json
import threading
import time
import uuid
import mimetypes
import urllib.error
import urllib.request
import urllib.parse
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI

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
DEFAULT_REPLICATE_MODEL = os.getenv(
    "REPLICATE_MODEL",
    "prunaai/flux-fast",
)
REPLICATE_API_BASE = "https://api.replicate.com/v1"
PROMPT_STORE_DIR = os.path.join(BASE_DIR, "data")
PROMPT_PREAMBLE_PATH = os.path.join(PROMPT_STORE_DIR, "base_preamble.txt")
PROMPT_CLOSING_PATH = os.path.join(PROMPT_STORE_DIR, "base_closing.txt")
APP_CONFIG_PATH = os.path.join(PROMPT_STORE_DIR, "app_config.json")
API_COUNTER_PATH = os.path.join(PROMPT_STORE_DIR, "api_call_counter.json")
FISHEYE_IMAGE_DIR = os.path.join(PROMPT_STORE_DIR, "fisheye_images")
FISHEYE_IMAGE_INDEX_PATH = os.path.join(PROMPT_STORE_DIR, "fisheye_images_index.json")
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


def _write_json_list(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, indent=2, ensure_ascii=True)
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


def _store_fisheye_image(source_url: str) -> tuple[str, str]:
    os.makedirs(FISHEYE_IMAGE_DIR, exist_ok=True)
    payload, content_type = _download_binary(source_url)
    ext = _infer_image_extension(source_url, content_type)
    image_ref = uuid.uuid4().hex[:20]
    filename = f"{image_ref}{ext}"
    file_path = os.path.join(FISHEYE_IMAGE_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(payload)

    items = _read_json_list(FISHEYE_IMAGE_INDEX_PATH)
    items.append(
        {
            "ref": image_ref,
            "filename": filename,
            "created_at": int(time.time()),
            "source_url": source_url,
        }
    )

    # Keep only the most recent 20 images.
    if len(items) > 20:
        stale = items[:-20]
        items = items[-20:]
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


def _build_replicate_input(
    prompt: str, model_slug: str, image_seed: Optional[int] = None
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


def _replicate_generate_image(
    prompt: str, model_slug: str, image_seed: Optional[int] = None
) -> str:
    replicate_input = _build_replicate_input(prompt, model_slug, image_seed)
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


@app.post("/api/generate")
async def generate(request: Request):
    _increment_api_call_count()
    body = await request.json()
    prompt = body.get("prompt", "")
    model = body.get("model") or DEFAULT_MODEL

    # 🔁 Dynamic provider routing
    client, actual_model = _build_model_client(model)

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


@app.post("/api/visualise")
async def visualise(payload: VisualisePayload):
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

    # Use current output text directly as the image-generation prompt.
    image_prompt = output_text

    try:
        _increment_api_call_count()
        image_url = _replicate_generate_image(image_prompt, image_model, image_seed)
        image_ref, local_image_url = _store_fisheye_image(image_url)
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


@app.get("/api/fisheye-images/{image_ref}")
async def get_fisheye_image(image_ref: str):
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


@app.post("/api/save-keys")
async def save_keys(payload: EnvKeysPayload):
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
