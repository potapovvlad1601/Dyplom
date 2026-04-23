from django.urls import path
from .views import upload_video, get_result

urlpatterns = [
    path("upload/", upload_video),
    path("result/<str:task_id>/", get_result),
]