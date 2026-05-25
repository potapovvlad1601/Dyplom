import os

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "CowApi"

    def ready(self):
        if os.environ.get("PRELOAD_COW_MODELS") != "1":
            return

        from .model_loader import preload_models

        preload_models()
