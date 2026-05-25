import json
import os
import uuid

from celery import shared_task
from django.conf import settings
from django.db import transaction

from .models import CowResult, VideoTask
from .pipeline import process_video


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
        "status": "done",
        "progress": 100,
        "stage": "done",
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


def update_progress(task_id, progress, stage):
    normalized_progress = max(0, min(int(progress), 99))
    _update_task_state(
        task_id,
        status="processing",
        progress=normalized_progress,
        stage=stage,
        error="",
    )


def _mark_task_error(task_id, video_path, error_message):
    VideoTask.objects.update_or_create(
        task_id=task_id,
        defaults={
            "videofile_name": os.path.basename(video_path),
            "status": "error",
            "stage": "error",
            "error": str(error_message),
        },
    )


@shared_task(name="CowApi.process_video_task")
def process_video_task(task_id, video_path):
    try:
        update_progress(task_id, 1, "starting")

        def progress_callback(progress, stage):
            update_progress(task_id, progress, stage)

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
            "status": "processing",
            "progress": 0,
            "stage": "queued",
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

    if video_task.status == "done":
        return _build_client_result_payload(video_task.task_id)

    if video_task.status == "error":
        return {
            "task_id": video_task.task_id,
            "status": "error",
            "error": video_task.error or "unknown_error",
        }

    return {
        "task_id": video_task.task_id,
        "status": "processing",
        "progress": video_task.progress,
        "stage": video_task.stage or "processing",
    }


def save_results_to_db(task_id, video_path, result):
    video_name = os.path.basename(video_path)

    video_task, _ = VideoTask.objects.update_or_create(
        task_id=task_id,
        defaults={
            "videofile_name": video_name,
            "status": "done",
            "progress": 100,
            "stage": "done",
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
