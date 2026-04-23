from django.db import models

class VideoTask(models.Model):
    STATUS_CHOICES = [
        ("processing", "Processing"),
        ("done", "Done"),
        ("error", "Error"),
    ]

    task_id = models.CharField(max_length=255, unique=True)
    video_name = models.CharField(max_length=255)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="processing")
    result_video = models.CharField(max_length=255, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)