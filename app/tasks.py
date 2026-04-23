from celery import shared_task
from .models import VideoTask
from .services.video_processor import process_video


@shared_task
def process_video_task(video_path, task_id):

    task = VideoTask.objects.get(task_id=task_id)

    try:
        output_path = f"media/output_{task_id}.mp4"

        process_video(video_path, output_path)

        task.status = "done"
        task.result_video = output_path
        task.save()

    except Exception as e:
        task.status = "error"
        task.save()