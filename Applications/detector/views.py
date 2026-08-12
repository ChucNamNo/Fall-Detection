"""HTTP endpoints for browser-camera fall detection."""
from __future__ import annotations

import uuid

import cv2
import numpy as np
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from .services.model_service import get_model_service


@ensure_csrf_cookie
def index(request: HttpRequest):
    return render(request, "detector/index.html")


@require_GET
def health(request: HttpRequest) -> JsonResponse:
    load_model = request.GET.get("load", "0") == "1"
    service = get_model_service()
    return JsonResponse(service.health(load=load_model), status=200)


@require_POST
def predict_frame(request: HttpRequest) -> JsonResponse:
    upload = request.FILES.get("image")
    session_id = (request.POST.get("session_id") or "").strip()

    if not session_id:
        session_id = str(uuid.uuid4())
    if len(session_id) > 128:
        return JsonResponse({"ok": False, "error": "session_id không hợp lệ."}, status=400)
    if upload is None:
        return JsonResponse({"ok": False, "error": "Thiếu ảnh camera."}, status=400)

    raw = np.frombuffer(upload.read(), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return JsonResponse({"ok": False, "error": "Không giải mã được ảnh."}, status=400)

    try:
        result = get_model_service().predict(frame, session_id)
        result["session_id"] = session_id
        return JsonResponse(result)
    except FileNotFoundError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)
    except RuntimeError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)
    except Exception as exc:  # Không làm sập server demo khi một frame lỗi.
        return JsonResponse(
            {"ok": False, "error": f"Lỗi xử lý frame: {type(exc).__name__}: {exc}"},
            status=500,
        )


@require_POST
def reset_session(request: HttpRequest) -> JsonResponse:
    session_id = (request.POST.get("session_id") or "").strip()
    if not session_id:
        return JsonResponse({"ok": False, "error": "Thiếu session_id."}, status=400)
    get_model_service().reset(session_id)
    return JsonResponse({"ok": True, "session_id": session_id})
