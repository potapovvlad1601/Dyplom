import os
import threading

import torch

from .model_classes import CowClassifier, CowTracker, PoseEstimator

_MODELS = None
_MODELS_LOCK = threading.Lock()

MODEL_CONFIGS = {
    "gpu": {
        "device": "cuda",
        "tracker": "models/gpu/Yolo26l-seg.pt",
        "pose_model": "models/gpu/Yolo26n-poseB.pt",
        "classifier": "models/gpu/Yolo11m-cls.pt",
    },
    "cpu": {
        "device": "cpu",
        "tracker": "models/cpu/Yolo26l-seg.onnx",
        "pose_model": "models/cpu/Yolo26n-poseB.onnx",
        "classifier": "models/cpu/Yolo11m-cls.onnx",
    },
}


def _get_backend_preference():
    backend = os.getenv("MODEL_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("MODEL_BACKEND must be one of: auto, cpu, gpu")
    return backend


def _get_missing_models(model_config):
    return [
        model_path
        for key, model_path in model_config.items()
        if key != "device" and not os.path.exists(model_path)
    ]


def _validate_backend(backend):
    model_config = MODEL_CONFIGS[backend]

    if backend == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU backend requested, but CUDA is not available")

    missing_models = _get_missing_models(model_config)
    if missing_models:
        raise FileNotFoundError(
            f"Missing {backend.upper()} model files: {', '.join(missing_models)}"
        )

    return backend, model_config


def _select_model_config():
    backend_preference = _get_backend_preference()

    if backend_preference in {"cpu", "gpu"}:
        return _validate_backend(backend_preference)

    if torch.cuda.is_available():
        try:
            return _validate_backend("gpu")
        except (RuntimeError, FileNotFoundError) as error:
            print(f"GPU backend unavailable, falling back to CPU: {error}")

    return _validate_backend("cpu")


def _build_models():
    backend, model_config = _select_model_config()
    device = model_config["device"]

    print(
        "Loading cow models "
        f"backend={backend}, device={device}, "
        f"tracker={model_config['tracker']}, "
        f"pose={model_config['pose_model']}, "
        f"classifier={model_config['classifier']}"
    )

    return {
        "device": device,
        "backend": backend,
        "model_paths": {
            "tracker": model_config["tracker"],
            "pose_model": model_config["pose_model"],
            "classifier": model_config["classifier"],
        },
        "tracker": CowTracker(model_config["tracker"], "models/botsort_reid20.yaml", device),
        "pose_model": PoseEstimator(model_config["pose_model"], device),
        "classifier": CowClassifier(model_config["classifier"], device),
    }


def get_models():
    global _MODELS

    if _MODELS is None:
        with _MODELS_LOCK:
            if _MODELS is None:
                _MODELS = _build_models()

    return _MODELS


def preload_models():
    return get_models()
