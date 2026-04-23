import threading
from app.services.video_processor import process_video
from app.models import VideoTask


def run_video_task(video_path, task_id):
    task = VideoTask.objects.get(task_id=task_id)

    try:
        output_path = f"media/output_{task_id}.mp4"

        txt_path = process_video(video_path, output_path)

        task.status = "done"
        task.result_video = output_path
        task.result_txt = txt_path
        task.save()

    except Exception as e:
        task.status = "error"
        task.save()


def start_video_processing(video_path, task_id):
    thread = threading.Thread(
        target=run_video_task,
        args=(video_path, task_id),
        daemon=True
    )
    thread.start()