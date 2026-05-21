import threading
import uuid
import os
import json
from .pipeline import process_video

TASKS = {}


def _build_client_artifact_urls(task_id):
    return {
        "annotated_video_name": "annotated.mp4",
        "annotated_video_url": f"/CowApi/result/{task_id}/files/annotated.mp4",
        "weights_file_name": "weights.json",
        "weights_file_url": f"/CowApi/result/{task_id}/files/weights.json",
    }


def _build_client_result_payload(task_id):
    return {
        "task_id": task_id,
        "status": "done",
        "progress": 100,
        "stage": "done",
        **_build_client_artifact_urls(task_id),
    }


def _build_result_urls(task_id, result):
    for cow_result in result.values():
        image_path = cow_result.get("snapshot_image")
        features_path = cow_result.get("snapshot_features")
        prompt_path = cow_result.get("snapshot_prompt")

        if image_path:
            cow_result["snapshot_image_url"] = f"/media/results/{task_id}/{image_path}"

        if features_path:
            cow_result["snapshot_features_url"] = f"/media/results/{task_id}/{features_path}"

        if prompt_path:
            cow_result["snapshot_prompt_url"] = f"/media/results/{task_id}/{prompt_path}"

    return result


def _save_weight_results(client_dir, result):
    os.makedirs(client_dir, exist_ok=True)

    weights_payload = {}
    for cow_id, cow_result in result.items():
        weights_payload[str(cow_id)] = {
            "weight": cow_result.get("weight"),
            "weight_confidence": cow_result.get("weight_confidence"),
            "weight_model": cow_result.get("weight_model"),
            "weight_error": cow_result.get("weight_error"),
        }

    weights_path = os.path.join(client_dir, "weights.json")
    with open(weights_path, "w", encoding="utf-8") as file:
        json.dump(weights_payload, file, ensure_ascii=False, indent=2)

    return weights_path


def run_task(task_id, video_path):
    try:
        def progress_callback(progress, stage):
            update_progress(task_id, progress, stage)

        output_dir = os.path.join("media", "results", task_id)
        client_dir = os.path.join("client", task_id)
        result, annotated_video_path = process_video(
            video_path,
            output_dir,
            client_dir,
            progress_callback=progress_callback
        )
        weights_path = _save_weight_results(client_dir, result)
        result = _build_result_urls(task_id, result)

        TASKS[task_id] = {
            "task_id": task_id,
            "status": "done",
            "progress": 100,
            "result": result,
            "video_path": annotated_video_path,
            "client_dir": client_dir,
            "weights_path": weights_path,
            **_build_client_artifact_urls(task_id),
        }

    except Exception as e:
        TASKS[task_id] = {
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        }


def process_video_task(video_path, task_id=None):
    task_id = task_id or str(uuid.uuid4())

    TASKS[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "progress": 0,
        "stage": "starting"
    }

    thread = threading.Thread(
        target=run_task,
        args=(task_id, video_path),
        daemon=True
    )
    thread.start()

    return task_id

def update_progress(task_id, progress, stage):
    TASKS[task_id]["progress"] = progress
    TASKS[task_id]["stage"] = stage

def get_task_result(task_id):
    task = TASKS.get(task_id)
    if not task:
        return {"task_id": task_id, "status": "not_found", "error": "not_found"}

    if task.get("status") == "done":
        return _build_client_result_payload(task_id)

    if task.get("status") == "error":
        return {
            "task_id": task_id,
            "status": "error",
            "error": task.get("error", "unknown_error"),
        }

    return {
        "task_id": task_id,
        "status": "processing",
        "progress": task.get("progress", 0),
        "stage": task.get("stage", "starting"),
    }
