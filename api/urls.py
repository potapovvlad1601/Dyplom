from django.urls import path
from .views import upload_video, get_result, get_result_file

urlpatterns = [
    path("upload/", upload_video),
    path("result/<str:task_id>/", get_result),
    path("result/<str:task_id>/files/<str:filename>", get_result_file),
]
