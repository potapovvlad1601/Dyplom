import uuid
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import VideoTask
from .tasks import process_video_task


@csrf_exempt
def upload_video(request):

    if request.method == "POST":
        file = request.FILES["file"]

        task_id = str(uuid.uuid4())
        path = f"media/{task_id}_{file.name}"

        with open(path, "wb+") as f:
            for chunk in file.chunks():
                f.write(chunk)

        VideoTask.objects.create(
            task_id=task_id,
            video_name=file.name,
            status="processing"
        )

        process_video_task.delay(path, task_id)

        return JsonResponse({"task_id": task_id})


def get_result(request, task_id):

    try:
        task = VideoTask.objects.get(task_id=task_id)

        return JsonResponse({
            "status": task.status,
            "result_video": task.result_video
        })

    except VideoTask.DoesNotExist:
        return JsonResponse({"error": "not found"}, status=404)