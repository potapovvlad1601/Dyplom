import os
import uuid

from django.http import FileResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings

from .tasks import enqueue_video_task, get_task_result


@csrf_exempt
def upload_video(request):
    if request.method == "POST":
        video = request.FILES["video"]
        task_id = str(uuid.uuid4())
        video_name = os.path.basename(video.name)

        path = default_storage.save(f"results/{task_id}/{video_name}", video)
        full_path = default_storage.path(path)

        task_id = enqueue_video_task(full_path, task_id=task_id)

        return JsonResponse({"task_id": task_id, "status": "queued"})


def get_result(request, task_id):
    return JsonResponse(get_task_result(task_id))


def get_result_file(request, task_id, filename):
    allowed_files = {"annotated.mp4", "weights.json"}
    if filename not in allowed_files:
        return JsonResponse({"status": "error", "error": "File not allowed"}, status=400)

    file_path = os.path.join(settings.BASE_DIR, "client", task_id, filename)
    if not os.path.exists(file_path):
        return JsonResponse({"status": "not_found"}, status=404)

    return FileResponse(open(file_path, "rb"), as_attachment=True, filename=filename)
