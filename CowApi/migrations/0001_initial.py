import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name="VideoTask",
            fields=[
                ("task_id", models.CharField(max_length=64, primary_key=True, serialize=False)),
                ("videofile_name", models.CharField(max_length=255)),
                ("received_at", models.DateTimeField(auto_now_add=True)),
                ("status", models.CharField(default="processing", max_length=32)),
                ("progress", models.PositiveSmallIntegerField(default=0)),
                ("stage", models.CharField(blank=True, default="queued", max_length=64)),
                ("error", models.TextField(blank=True, default="")),
            ],
        ),
        migrations.CreateModel(
            name="CowResult",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("cow_id", models.IntegerField()),
                ("measurements", models.JSONField(blank=True, null=True)),
                ("breed", models.CharField(blank=True, max_length=128, null=True)),
                ("weight", models.FloatField(blank=True, null=True)),
                ("weight_confidence", models.CharField(blank=True, max_length=32, null=True)),
                ("task", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="cows", to="CowApi.videotask")),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(fields=("task", "cow_id"), name="unique_task_cow"),
                ],
            },
        ),
    ]
