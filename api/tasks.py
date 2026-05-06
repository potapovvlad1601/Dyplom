import threading
import uuid
import os
from .cow_pipeline import process_video

TASKS = {}


def _build_result_urls(task_id, result):
    for cow_result in result.values():
        image_path = cow_result.get("snapshot_image")
        features_path = cow_result.get("snapshot_features")

        if image_path:
            cow_result["snapshot_image_url"] = f"/media/results/{task_id}/{image_path}"

        if features_path:
            cow_result["snapshot_features_url"] = f"/media/results/{task_id}/{features_path}"

    return result


def run_task(task_id, video_path):
    try:
        def progress_callback(progress, stage):
            update_progress(task_id, progress, stage)

        output_dir = os.path.join("media", "results", task_id)
        result, annotated_video_path = process_video(
            video_path,
            output_dir,
            progress_callback=progress_callback
        )
        result = _build_result_urls(task_id, result)

        TASKS[task_id] = {
            "status": "done",
            "progress": 100,
            "result": result,
            "video_path": annotated_video_path,
            "video_url": f"/media/results/{task_id}/annotated.mp4"
        }

    except Exception as e:
        TASKS[task_id] = {
            "status": "error",
            "error": str(e)
        }


def process_video_task(video_path):
    task_id = str(uuid.uuid4())

    TASKS[task_id] = {
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
    return TASKS.get(task_id, {"status": "not_found"})
