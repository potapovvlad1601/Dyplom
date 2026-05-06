import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "300"))


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
  "reasoning": "short explanation referencing both the image and the numerical features"
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
    content_blocks = response_payload.get("content", [])
    text_parts = []

    for block in content_blocks:
        if block.get("type") == "text" and block.get("text"):
            text_parts.append(block["text"])

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
    reasoning = payload.get("reasoning")

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

    if reasoning is not None:
        reasoning = str(reasoning).strip()

    return {
        "weight": weight_kg,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def request_claude_weight(image_path, prompt, api_key, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
    media_type = _guess_media_type(image_path)
    image_base64 = _load_image_base64(image_path)

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }

    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Claude API request failed with status {error.code}: {error_body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Claude API request failed: {error}") from error

    return response_data


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
        "reasoning": parsed["reasoning"],
        "model": model,
        "prompt": prompt,
        "raw_response": response_text,
        "usage": response_payload.get("usage"),
    }
