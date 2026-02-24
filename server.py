import os
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


@app.post("/api/generate")
async def generate(request: Request):
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


# Static files last
app.mount("/", StaticFiles(directory="public", html=True), name="static")
