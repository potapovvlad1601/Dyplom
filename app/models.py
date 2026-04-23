from django.db import models

class VideoTask(models.Model):
    task_id = models.CharField(max_length=100)
    status = models.CharField(max_length=20)

    result_video = models.CharField(max_length=255, null=True, blank=True)
    result_txt = models.CharField(max_length=255, null=True, blank=True)