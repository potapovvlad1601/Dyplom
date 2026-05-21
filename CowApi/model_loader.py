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


def _select_model_config():
    backend = "gpu" if torch.cuda.is_available() else "cpu"
    model_config = MODEL_CONFIGS[backend]
    missing_models = [
        model_path
        for key, model_path in model_config.items()
        if key != "device" and not os.path.exists(model_path)
    ]

    if missing_models:
        raise FileNotFoundError(
            f"Missing {backend.upper()} model files: {', '.join(missing_models)}"
        )

    return backend, model_config


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
