import threading
import uuid
from .cow_pipeline import process_video

TASKS = {}


def run_task(task_id, video_path):
    try:
        def progress_callback(progress, stage):
            update_progress(task_id, progress, stage)

        result = process_video(
            video_path,
            f"results/{task_id}",
            progress_callback=progress_callback
        )

        TASKS[task_id] = {
            "status": "done",
            "progress": 100,
            "result": result
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