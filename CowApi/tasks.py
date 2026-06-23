import json
import os
import uuid

from celery import shared_task
from django.conf import settings
from django.db import transaction

from .models import CowResult, VideoTask
from .pipeline import process_video

STATUS_QUEUED = "queued"
STATUS_PROGRESSING = "progressing"
STATUS_LLM_WEIGHT = "llm_weight"
STATUS_ANNOTATING = "annotating"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"

CLIENT_STATUSES = {
    STATUS_QUEUED,
    STATUS_PROGRESSING,
    STATUS_LLM_WEIGHT,
    STATUS_ANNOTATING,
    STATUS_COMPLETED,
    STATUS_ERROR,
}

LEGACY_STAGE_TO_STATUS = {
    "queued": STATUS_QUEUED,
    "starting": STATUS_PROGRESSING,
    "tracking": STATUS_PROGRESSING,
    "pose": STATUS_PROGRESSING,
    "finalizing": STATUS_ANNOTATING,
    "claude_results": STATUS_LLM_WEIGHT,
    "done": STATUS_COMPLETED,
    "error": STATUS_ERROR,
}


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
        "status": STATUS_COMPLETED,
        "progress": 100,
        "stage": STATUS_COMPLETED,
        **_build_client_artifact_urls(task_id),
    }


def _save_weight_results(client_dir, result):
    os.makedirs(client_dir, exist_ok=True)

    weights_payload = {}
    for cow_id, cow_result in result.items():
        weights_payload[str(cow_id)] = {
            "weight": cow_result.get("weight"),
            "weight_confidence": cow_result.get("weight_confidence"),
            "weight_error": cow_result.get("weight_error"),
        }

    weights_path = os.path.join(client_dir, "weights.json")
    with open(weights_path, "w", encoding="utf-8") as file:
        json.dump(weights_payload, file, ensure_ascii=False, indent=2)

    return weights_path


def _update_task_state(task_id, **fields):
    VideoTask.objects.filter(task_id=task_id).update(**fields)


def _normalize_client_status(status, stage=""):
    if status in CLIENT_STATUSES:
        return status
    if status == "done":
        return STATUS_COMPLETED
    if status == "processing":
        return LEGACY_STAGE_TO_STATUS.get(stage or "", STATUS_PROGRESSING)
    return status


def update_progress(task_id, progress, status):
    normalized_progress = max(0, min(int(progress), 99))
    normalized_status = _normalize_client_status(status)
    _update_task_state(
        task_id,
        status=normalized_status,
        progress=normalized_progress,
        stage=normalized_status,
        error="",
    )


def _mark_task_error(task_id, video_path, error_message):
    VideoTask.objects.update_or_create(
        task_id=task_id,
        defaults={
            "videofile_name": os.path.basename(video_path),
            "status": STATUS_ERROR,
            "stage": STATUS_ERROR,
            "error": str(error_message),
        },
    )


@shared_task(name="CowApi.process_video_task")
def process_video_task(task_id, video_path):
    try:
        update_progress(task_id, 1, STATUS_PROGRESSING)

        def progress_callback(progress, status):
            update_progress(task_id, progress, status)

        output_dir = settings.MEDIA_ROOT / "results" / task_id
        client_dir = settings.BASE_DIR / "client" / task_id

        result, annotated_video_path = process_video(
            video_path,
            os.fspath(output_dir),
            os.fspath(client_dir),
            progress_callback=progress_callback,
        )
        weights_path = _save_weight_results(os.fspath(client_dir), result)
        save_results_to_db(task_id, video_path, result)

        return {
            "task_id": task_id,
            "annotated_video_path": annotated_video_path,
            "weights_path": weights_path,
        }
    except Exception as error:
        _mark_task_error(task_id, video_path, error)
        raise


def enqueue_video_task(video_path, task_id=None):
    task_id = task_id or str(uuid.uuid4())
    video_name = os.path.basename(video_path)

    VideoTask.objects.update_or_create(
        task_id=task_id,
        defaults={
            "videofile_name": video_name,
            "status": STATUS_QUEUED,
            "progress": 0,
            "stage": STATUS_QUEUED,
            "error": "",
        },
    )

    transaction.on_commit(
        lambda: process_video_task.apply_async(
            args=(task_id, video_path),
            task_id=task_id,
        )
    )

    return task_id


def get_task_result(task_id):
    video_task = VideoTask.objects.filter(task_id=task_id).first()
    if not video_task:
        return {"task_id": task_id, "status": "not_found", "error": "not_found"}

    client_status = _normalize_client_status(video_task.status, video_task.stage)

    if client_status == STATUS_COMPLETED:
        return _build_client_result_payload(video_task.task_id)

    if client_status == STATUS_ERROR:
        return {
            "task_id": video_task.task_id,
            "status": STATUS_ERROR,
            "error": video_task.error or "unknown_error",
        }

    return {
        "task_id": video_task.task_id,
        "status": client_status,
        "progress": video_task.progress,
        "stage": client_status,
    }


def save_results_to_db(task_id, video_path, result):
    video_name = os.path.basename(video_path)

    video_task, _ = VideoTask.objects.update_or_create(
        task_id=task_id,
        defaults={
            "videofile_name": video_name,
            "status": STATUS_COMPLETED,
            "progress": 100,
            "stage": STATUS_COMPLETED,
            "error": "",
        },
    )

    cow_ids = [int(cow_id) for cow_id in result.keys()]
    stale_results = CowResult.objects.filter(task=video_task)
    if cow_ids:
        stale_results.exclude(cow_id__in=cow_ids).delete()
    else:
        stale_results.delete()

    for cow_id, cow_data in result.items():
        CowResult.objects.update_or_create(
            task=video_task,
            cow_id=int(cow_id),
            defaults={
                "measurements": cow_data.get("measurements"),
                "breed": cow_data.get("breed"),
                "weight": cow_data.get("weight"),
                "weight_confidence": cow_data.get("weight_confidence"),
            },
        )
