import os

from celery import Celery
from celery.signals import worker_process_init

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MainServerv2.settings")

app = Celery("MainServerv2")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


@worker_process_init.connect
def preload_worker_models(**kwargs):
    from CowApi.model_loader import preload_models

    preload_models()
