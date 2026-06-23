from django.db import models


class VideoTask(models.Model):
    task_id = models.CharField(
        primary_key=True,
        max_length=64
    )

    videofile_name = models.CharField(max_length=255)

    received_at = models.DateTimeField(
        auto_now_add=True
    )

    status = models.CharField(
        max_length=32,
        default="queued"
    )

    progress = models.PositiveSmallIntegerField(
        default=0
    )

    stage = models.CharField(
        max_length=64,
        default="queued",
        blank=True
    )

    error = models.TextField(
        default="",
        blank=True
    )

    def __str__(self):
        return self.task_id


class CowResult(models.Model):
    task = models.ForeignKey(
        VideoTask,
        on_delete=models.CASCADE,
        related_name="cows"
    )

    cow_id = models.IntegerField()

    measurements = models.JSONField(
        null=True,
        blank=True
    )

    breed = models.CharField(
        max_length=128,
        null=True,
        blank=True
    )

    weight = models.FloatField(
        null=True,
        blank=True
    )

    weight_confidence = models.CharField(
        max_length=32,
        null=True,
        blank=True
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("task", "cow_id"),
                name="unique_task_cow"
            )
        ]

    def __str__(self):
        return f"{self.task.task_id} - cow {self.cow_id}"
