import uuid
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from app.models import VideoTask
from app.services.video_thread import start_video_processing


@csrf_exempt
def upload_video(request):
    if request.method == "POST":

        file = request.FILES["file"]

        task_id = str(uuid.uuid4())
        video_path = f"media/{task_id}.mp4"

        with open(video_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)

        VideoTask.objects.create(
            task_id=task_id,
            status="processing"
        )

        # 🔥 запуск потока
        start_video_processing(video_path, task_id)

        return JsonResponse({"task_id": task_id})

def get_result(request, task_id):
    task = VideoTask.objects.get(task_id=task_id)

    return JsonResponse({
        "status": task.status,
        "result_video": task.result_video,
        "result_txt": task.result_txt
    })