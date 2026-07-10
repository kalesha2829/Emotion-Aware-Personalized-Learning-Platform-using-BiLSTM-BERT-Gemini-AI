"""Google Gemini AI integration for personalized study guidance.

This module loads the Gemini API key from a .env file, generates a guidance prompt
from the final detected emotion and student input, and returns a structured
response containing learning guidance, encouragement, study tips, and next steps.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover
    find_dotenv = None
    load_dotenv = None

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

FALLBACK_RESPONSE = {
    "personalized_learning_guidance": (
        "I couldn't reach Gemini AI right now, but you can still move forward by "
        "focusing on one small topic at a time and applying active recall. "
        "Start with the concept that feels most familiar, then gradually expand "
        "into the harder material."
    ),
    "encouraging_response": (
        "You're doing a great job by reflecting on your learning and asking for help. "
        "Keep going — every step you take builds stronger understanding."
    ),
    "study_tips": (
        "Try short focused sessions, summarize what you learned in your own words, "
        "and test yourself with simple examples. If a concept feels confusing, "
        "rewrite it as if you're explaining it to a friend."
    ),
    "next_learning_steps": (
        "Review your notes, identify one clear goal for your next study session, "
        "and choose a related practice problem to apply what you just learned."
    ),
}

SECTION_KEY_MAP = {
    "personalized learning guidance": "personalized_learning_guidance",
    "learning guidance": "personalized_learning_guidance",
    "encouraging response": "encouraging_response",
    "encouragement": "encouraging_response",
    "study tips": "study_tips",
    "tips": "study_tips",
    "suggested next learning steps": "next_learning_steps",
    "next learning steps": "next_learning_steps",
    "next steps": "next_learning_steps",
}


@dataclass
class GeminiLearningSupport:
    personalized_learning_guidance: str
    encouraging_response: str
    study_tips: str
    next_learning_steps: str
    raw_response: str | None = None
    error: str | None = None


class GeminiServiceError(Exception):
    """Raised when the Gemini service is unavailable or misconfigured."""


def _load_local_env() -> None:
    """Parse a local .env file if python-dotenv is unavailable."""
    current = Path.cwd()
    candidates = [current] + list(current.parents)
    env_path = None
    for directory in candidates:
        candidate = directory / ".env"
        if candidate.exists():
            env_path = candidate
            break

    if not env_path:
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_api_key() -> str:
    """Load the Gemini API key from a .env file in the repository root."""
    if load_dotenv is not None and find_dotenv is not None:
        load_dotenv(find_dotenv())
    else:
        _load_local_env()

    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise GeminiServiceError(
            f"Gemini API key not found. Set {GEMINI_API_KEY_ENV} in a .env file."
        )
    return api_key


def _import_gemini() -> Any:
    """Import the Google Gemini SDK, if available."""
    try:
        import google.generativeai as genai

        return genai
    except ImportError as exc:
        raise GeminiServiceError(
            "google-generativeai is not installed. Install it to enable Gemini support."
        ) from exc


def _build_prompt(final_emotion: str, original_input: str) -> str:
    """Build a prompt that asks Gemini AI for tailored learning guidance."""
    return (
        "You are a supportive learning coach. A student just shared the following text: "
        f'"{original_input}". The detected emotion for this student is '
        f'"{final_emotion}". Based on that emotion and the student’s original input, '
        "provide a helpful response with four clearly labeled fields: "
        "personalized learning guidance, encouraging response, study tips, "
        "and suggested next learning steps. Return a valid JSON object with keys: "
        "personalized_learning_guidance, encouraging_response, study_tips, "
        "next_learning_steps. Keep the tone positive, actionable, and appropriate "
        "for a student who is learning."
    )


def _normalize_section_name(section_label: str) -> str:
    normalized = section_label.strip().lower()
    return SECTION_KEY_MAP.get(normalized, "")


def _parse_gemini_response(response_text: str) -> dict[str, str]:
    """Parse Gemini response text into the expected structured fields."""
    candidate = response_text.strip()

    # Clean markdown json code blocks if present
    match_code_block = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if match_code_block:
        candidate = match_code_block.group(1).strip()

    def clean_field(val: Any) -> str:
        if isinstance(val, list):
            return "\n".join(f"- {item}" for item in val).strip()
        return str(val or "").strip()

    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return {
                "personalized_learning_guidance": clean_field(
                    payload.get("personalized_learning_guidance", "")
                ),
                "encouraging_response": clean_field(
                    payload.get("encouraging_response", "")
                ),
                "study_tips": clean_field(payload.get("study_tips", "")),
                "next_learning_steps": clean_field(
                    payload.get("next_learning_steps", "")
                ),
            }
    except json.JSONDecodeError:
        pass

    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []
    pattern = re.compile(
        r"^(personalized learning guidance|learning guidance|encouraging response|encouragement|study tips|tips|suggested next learning steps|next learning steps|next steps)\s*[:\-]?\s*(.*)$",
        re.IGNORECASE,
    )

    for line in candidate.splitlines():
        match = pattern.match(line)
        if match:
            if current_key and buffer:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = _normalize_section_name(match.group(1))
            buffer = [match.group(2).strip()] if match.group(2).strip() else []
        elif current_key is not None:
            buffer.append(line)

    if current_key and buffer:
        sections[current_key] = "\n".join(buffer).strip()

    if not sections:
        return {}

    return {
        "personalized_learning_guidance": sections.get(
            "personalized_learning_guidance", ""
        ),
        "encouraging_response": sections.get("encouraging_response", ""),
        "study_tips": sections.get("study_tips", ""),
        "next_learning_steps": sections.get("next_learning_steps", ""),
    }


def _fallback_support(error: str | None = None) -> GeminiLearningSupport:
    result = GeminiLearningSupport(
        personalized_learning_guidance=FALLBACK_RESPONSE["personalized_learning_guidance"],
        encouraging_response=FALLBACK_RESPONSE["encouraging_response"],
        study_tips=FALLBACK_RESPONSE["study_tips"],
        next_learning_steps=FALLBACK_RESPONSE["next_learning_steps"],
        raw_response=None,
        error=error,
    )
    return result
DEBUG_DISABLE_FALLBACK = False


def generate_gemini_learning_support(
    final_emotion: str,
    original_input: str,
) -> GeminiLearningSupport:
    """Generate structured learning guidance from Gemini AI.

    Includes retry logic with exponential backoff and detailed diagnostics logging.
    """
    import time
    import random
    from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

    genai = _import_gemini()
    api_key = _load_api_key()
    
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "too_short"
    print(f"[GEMINI DEBUG] Loading API Key: {masked_key}")
    print(f"[GEMINI DEBUG] Using model: {GEMINI_MODEL_NAME}")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
    prompt = _build_prompt(final_emotion, original_input)
    
    max_retries = 5
    initial_backoff = 1.0
    timeout = 15.0
    backoff = initial_backoff
    last_exc = None

    for attempt in range(max_retries):
        try:
            print(f"[GEMINI DEBUG] Attempt {attempt + 1}/{max_retries} to call Gemini API...")
            response = model.generate_content(
                prompt,
                request_options={"timeout": timeout}
            )
            raw_text = getattr(response, "text", str(response)).strip()
            print(f"[GEMINI DEBUG] Raw response received: {raw_text}")

            parsed = _parse_gemini_response(raw_text)
            if parsed and any(parsed.values()):
                return GeminiLearningSupport(
                    personalized_learning_guidance=parsed["personalized_learning_guidance"],
                    encouraging_response=parsed["encouraging_response"],
                    study_tips=parsed["study_tips"],
                    next_learning_steps=parsed["next_learning_steps"],
                    raw_response=raw_text,
                    error=None,
                )

            raise GeminiServiceError(
                f"Unexpected format or empty response from Gemini. Raw response: {raw_text}"
            )
        except (ResourceExhausted, GoogleAPICallError) as exc:
            last_exc = exc
            status_code = getattr(exc, "code", "Unknown")
            error_message = getattr(exc, "message", str(exc))
            
            # Log detailed error using log_gemini_error
            try:
                from utils.logger import log_gemini_error
                import traceback
                log_gemini_error(
                    model_used=GEMINI_MODEL_NAME,
                    http_status=status_code,
                    exception_type=type(exc).__name__,
                    error_message=error_message,
                    complete_error=traceback.format_exc(),
                )
            except Exception as log_err:
                print(f"[GEMINI DEBUG] Failed to write error log: {log_err}")
                
            if status_code in [429, 500, 502, 503, 504] or isinstance(exc, ResourceExhausted):
                if attempt == max_retries - 1:
                    break
                sleep_time = backoff + random.uniform(0, 0.5)
                print(f"[GEMINI RETRY] Attempt {attempt + 1} failed (HTTP {status_code}). Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                backoff *= 2
            else:
                break
        except Exception as exc:
            last_exc = exc
            status_code = "Unknown"
            error_message = str(exc)
            
            # Log detailed error using log_gemini_error
            try:
                from utils.logger import log_gemini_error
                import traceback
                log_gemini_error(
                    model_used=GEMINI_MODEL_NAME,
                    http_status=status_code,
                    exception_type=type(exc).__name__,
                    error_message=error_message,
                    complete_error=traceback.format_exc(),
                )
            except Exception as log_err:
                print(f"[GEMINI DEBUG] Failed to write error log: {log_err}")
                
            if attempt == max_retries - 1:
                break
            sleep_time = backoff + random.uniform(0, 0.5)
            print(f"[GEMINI RETRY] Attempt {attempt + 1} failed ({type(exc).__name__}). Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            backoff *= 2

    # Handle final failure after retries
    status_code = getattr(last_exc, "code", "Unknown")
    error_message = getattr(last_exc, "message", str(last_exc))
    detail_msg = (
        f"Gemini API Error occurred (after retries):\n"
        f"- Model Name: {GEMINI_MODEL_NAME}\n"
        f"- API Key Loaded: Yes ({masked_key})\n"
        f"- HTTP Status Code: {status_code}\n"
        f"- Error Message: {error_message}"
    )
    print(f"[GEMINI ERROR DETAIL]\n{detail_msg}")
    
    if DEBUG_DISABLE_FALLBACK:
        raise Exception(detail_msg) from last_exc
        
    return _fallback_support(str(last_exc))
