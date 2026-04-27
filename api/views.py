from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

from .tasks import process_video_task, get_task_result


@csrf_exempt
def upload_video(request):
    if request.method == "POST":
        video = request.FILES["video"]

        path = default_storage.save(f"uploads/{video.name}", video)
        full_path = default_storage.path(path)

        task_id = process_video_task(full_path)

        return JsonResponse({"task_id": task_id})


def get_result(request, task_id):
    return JsonResponse(get_task_result(task_id))