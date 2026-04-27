import threading
import uuid
from .cow_pipeline import process_video

TASKS = {}


def run_task(task_id, video_path):
    try:
        result = process_video(video_path, f"results/{task_id}")

        TASKS[task_id] = {
            "status": "done",
            "result": result
        }

    except Exception as e:
        TASKS[task_id] = {
            "status": "error",
            "error": str(e)
        }


def process_video_task(video_path):
    task_id = str(uuid.uuid4())

    TASKS[task_id] = {"status": "processing"}

    thread = threading.Thread(
        target=run_task,
        args=(task_id, video_path),
        daemon=True
    )
    thread.start()

    return task_id


def get_task_result(task_id):
    return TASKS.get(task_id, {"status": "not_found"})