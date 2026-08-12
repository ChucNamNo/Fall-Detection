from django.urls import path
from . import views

app_name = "detector"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/health/", views.health, name="health"),
    path("api/predict/", views.predict_frame, name="predict"),
    path("api/reset/", views.reset_session, name="reset"),
]
