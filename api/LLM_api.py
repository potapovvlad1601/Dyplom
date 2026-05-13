import base64
import json
import mimetypes
import os
import re
from functools import lru_cache

import anthropic
from MainServerv2.env import load_env_file

load_env_file()

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "3000"))


PROMPT_TEMPLATE = """You are an expert in livestock weight estimation.

Your task is to estimate the weight of a cow using:

numerical features
the image

Both sources must be used simultaneously and with equal importance.

NUMERICAL FEATURES

{feature_lines}

INSTRUCTIONS
Estimate the cow's weight by jointly interpreting numerical features and the image from the start.
Treat both sources as equally important signals.
Do NOT compute separate intermediate estimates.

While reasoning, consider:
- overall body size and proportions from numerical features
- body volume indicators
- girth measurements as indicators of mass
- visual body condition (thin / normal / fat)
- musculature and body density
- how well the visual appearance matches the numerical proportions

CONSISTENCY CHECK
Ensure that the final weight is consistent with BOTH:
- the scale suggested by numerical features
- the visual appearance of the cow

If there is a mismatch:
- adjust the estimate to better reflect both sources
- briefly explain the inconsistency

IMPORTANT CONSTRAINTS
- Do NOT use or mention formulas or equations
- Do NOT split reasoning into separate feature-only and image-only estimates
- Avoid unnecessary precision

OUTPUT FORMAT
Return valid JSON only with this schema:
{{
  "weight_kg": 0,
  "confidence": "low|medium|high",
}}
"""


def load_features_from_txt(features_path):
    features = {}

    with open(features_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            try:
                features[key] = float(value)
            except ValueError:
                continue

    if not features:
        raise ValueError(f"No features found in file: {features_path}")

    return features


def build_weight_prompt(features):
    feature_lines = "\n".join(
        f"{key}: {{{value:.6f}}}"
        for key, value in features.items()
    )
    return PROMPT_TEMPLATE.format(feature_lines=feature_lines)


def build_weight_prompt_from_file(features_path):
    features = load_features_from_txt(features_path)
    return build_weight_prompt(features)


def save_weight_prompt(prompt_path, prompt):
    with open(prompt_path, "w", encoding="utf-8") as file:
        file.write(prompt)


def _guess_media_type(image_path):
    media_type, _ = mimetypes.guess_type(image_path)
    if media_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        return media_type
    return "image/jpeg"


def _load_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def _extract_text_content(response_payload):
    content_blocks = response_payload.content
    text_parts = []

    for block in content_blocks:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            text_parts.append(block.text)

    return "\n".join(text_parts).strip()


def _extract_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Claude response does not contain JSON: {text}")

    return json.loads(match.group(0))


def parse_weight_response(text):
    payload = _extract_json_object(text)

    weight_kg = payload.get("weight_kg")
    confidence = payload.get("confidence")

    if weight_kg is None:
        raise ValueError(f"Claude response is missing weight_kg: {text}")

    try:
        weight_kg = float(weight_kg)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid weight_kg in Claude response: {text}") from error

    if confidence is not None:
        confidence = str(confidence).strip().lower()

    if confidence not in {"low", "medium", "high"}:
        confidence = None

    return {
        "weight": weight_kg,
        "confidence": confidence,
    }


@lru_cache(maxsize=None)
def _get_anthropic_client(api_key):
    return anthropic.Anthropic(api_key=api_key)


def request_claude_weight(image_path, prompt, api_key, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
    client = _get_anthropic_client(api_key)

    try:
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _guess_media_type(image_path),
                                "data": _load_image_base64(image_path),
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
    except anthropic.APIError as error:
        raise RuntimeError(f"Claude API request failed: {error}") from error


def estimate_weight_from_files(image_path, features_path, api_key=None, model=DEFAULT_MODEL):
    resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    prompt = build_weight_prompt_from_file(features_path)
    response_payload = request_claude_weight(
        image_path=image_path,
        prompt=prompt,
        api_key=resolved_api_key,
        model=model,
    )
    response_text = _extract_text_content(response_payload)
    parsed = parse_weight_response(response_text)

    return {
        "weight": parsed["weight"],
        "confidence": parsed["confidence"],
        "model": str(response_payload.model),
        "prompt": prompt,
        "raw_response": response_text,
        "usage": response_payload.usage.model_dump(),
    }
