import os

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "CowApi"

    def ready(self):
        if os.environ.get("RUN_MAIN") not in {None, "true"}:
            return

        from .model_loader import preload_models

        preload_models()
