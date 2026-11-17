"""Background-removal endpoints."""

from __future__ import annotations

import asyncio
import base64
import functools
import logging
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status

from app import metrics
from app.config import Settings, get_settings
from app.errors import ApiError, ModelNotReady, PayloadTooLarge, UnsupportedMedia
from app.fetcher import fetch_image_bytes
from app.logging_config import request_id_var
from app.model import get_remover
from app.pipeline import ProcessResult, process_image
from app.ratelimit import enforce_rate_limit
from app.runner import get_runner
from app.schemas import (
    ProcessingOptions,
    RemoveJsonResponse,
    RemoveUrlRequest,
)
from app.security import require_api_key
from app.storage import StorageError, get_storage

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["background-removal"])

_READ_CHUNK = 1024 * 1024


async def _read_upload(upload: UploadFile, settings: Settings) -> bytes:
    """Read a multipart file with a hard byte cap.

    Content-Length is attacker-controlled, so the cap is enforced while
    streaming rather than trusted up front.
    """
    if upload.content_type and settings.allowed_content_types:
        base_type = upload.content_type.split(";")[0].strip().lower()
        if base_type not in settings.allowed_content_types:
            raise UnsupportedMedia(
                f"Content type {base_type!r} is not accepted. "
                f"Allowed: {', '.join(settings.allowed_content_types)}."
            )

    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_READ_CHUNK):
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise PayloadTooLarge(
                f"Upload exceeds the {settings.max_upload_bytes / 1e6:.1f} MB limit."
            )
        chunks.append(chunk)
    await upload.close()

    if total == 0:
        raise UnsupportedMedia("Uploaded file is empty.")
    return b"".join(chunks)


def _options_from_form(
    output_format: str | None,
    background: str | None,
    feather: float,
    threshold: float | None,
    edge_contrast: float,
    max_output_dimension: int | None,
    upload: bool,
) -> ProcessingOptions:
    """Validate form fields through the same model used for JSON bodies.

    model_validate(), not the constructor: multipart fields are plain strings,
    so letting pydantic coerce them is what turns `output_format=gif` into a
    clean 400 instead of a type error.
    """
    try:
        return ProcessingOptions.model_validate(
            {
                "output_format": output_format or None,
                "background": background or None,
                "feather": feather,
                "threshold": threshold,
                "edge_contrast": edge_contrast,
                "max_output_dimension": max_output_dimension,
                "upload": upload,
            }
        )
    except ValueError as exc:
        raise ApiError(
            f"Invalid processing options: {exc}", code="invalid_options"
        ) from None


async def _run_pipeline(
    data: bytes,
    options: ProcessingOptions,
    settings: Settings,
    declared_content_type: str | None = None,
) -> ProcessResult:
    remover = get_remover()
    if not remover.is_ready:
        raise ModelNotReady()

    # partial(), not positional args: process_image takes
    # declared_content_type as keyword-only.
    job = functools.partial(
        process_image,
        data,
        options,
        settings,
        remover,
        declared_content_type=declared_content_type,
    )
    result = await get_runner().run(job)
    metrics.inc("rmbg_images_processed_total", {"format": result.extension})
    return result


async def _maybe_upload(
    result: ProcessResult, filename: str, options: ProcessingOptions
) -> str | None:
    """Push to object storage off the inference slot, so network latency does
    not occupy an inference worker."""
    if not options.upload:
        return None

    storage = get_storage()
    if not storage.enabled:
        raise ApiError(
            "upload=true was requested but object storage is not configured.",
            code="storage_disabled",
        )

    key = storage.build_key(filename, result.extension)
    try:
        url = await asyncio.to_thread(
            storage.upload, result.payload, key, result.media_type
        )
    except StorageError as exc:
        raise ApiError(str(exc), status_code=502, code="storage_error") from None

    metrics.inc("rmbg_uploads_total")
    return url


def _output_filename(original: str | None, extension: str) -> str:
    stem = PurePosixPath(original or "image").stem or "image"
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in stem).strip("-")
    return f"{safe or 'image'}_cutout.{extension}"


def _binary_response(result: ProcessResult, filename: str) -> Response:
    return Response(
        content=result.payload,
        media_type=result.media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(result.payload)),
            "Cache-Control": "no-store",
            "X-Image-Width": str(result.meta.width),
            "X-Image-Height": str(result.meta.height),
            "X-Inference-Ms": str(result.timings_ms.get("inference", 0)),
        },
    )


# --------------------------------------------------------------------------- #
# Binary endpoint - the one a web front-end should call.
# --------------------------------------------------------------------------- #
@router.post(
    "/remove",
    summary="Remove the background and return the image bytes",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "Cutout image"},
        413: {"description": "Image too large"},
        415: {"description": "Unsupported media type"},
        429: {"description": "Rate limited"},
        503: {"description": "Model loading or server at capacity"},
    },
)
async def remove_background_binary(
    request: Request,
    file: UploadFile = File(..., description="Source image"),
    output_format: str | None = Form(None),
    background: str | None = Form(None),
    feather: float = Form(0.0),
    threshold: float | None = Form(None),
    edge_contrast: float = Form(0.0),
    max_output_dimension: int | None = Form(None),
    identity: str = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> Response:
    enforce_rate_limit(request, identity)
    options = _options_from_form(
        output_format, background, feather, threshold, edge_contrast,
        max_output_dimension, upload=False,
    )
    data = await _read_upload(file, settings)
    result = await _run_pipeline(data, options, settings, file.content_type)

    log.info(
        "processed image",
        extra={
            "identity": identity,
            "in_bytes": len(data),
            "out_bytes": len(result.payload),
            "format": result.extension,
            **{f"ms_{k}": v for k, v in result.timings_ms.items()},
        },
    )
    return _binary_response(result, _output_filename(file.filename, result.extension))


# --------------------------------------------------------------------------- #
# JSON endpoint - base64 and/or a storage URL.
# --------------------------------------------------------------------------- #
@router.post(
    "/remove/json",
    response_model=RemoveJsonResponse,
    summary="Remove the background and return JSON (base64 and/or storage URL)",
)
async def remove_background_json(
    request: Request,
    file: UploadFile = File(...),
    output_format: str | None = Form(None),
    background: str | None = Form(None),
    feather: float = Form(0.0),
    threshold: float | None = Form(None),
    edge_contrast: float = Form(0.0),
    max_output_dimension: int | None = Form(None),
    upload: bool = Form(False),
    include_base64: bool = Form(True),
    identity: str = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> RemoveJsonResponse:
    enforce_rate_limit(request, identity)
    options = _options_from_form(
        output_format, background, feather, threshold, edge_contrast,
        max_output_dimension, upload,
    )
    data = await _read_upload(file, settings)
    result = await _run_pipeline(data, options, settings, file.content_type)
    storage_url = await _maybe_upload(result, file.filename or "image", options)

    return RemoveJsonResponse(
        request_id=request_id_var.get() or "",
        filename=_output_filename(file.filename, result.extension),
        content_type=result.media_type,
        bytes=len(result.payload),
        image_base64=(
            base64.b64encode(result.payload).decode() if include_base64 else None
        ),
        storage_url=storage_url,
        meta=result.meta,
        timings_ms=result.timings_ms,
    )


# --------------------------------------------------------------------------- #
# Batch endpoint - the production form of the notebook's multi-file loop.
# --------------------------------------------------------------------------- #
@router.post(
    "/remove/batch",
    summary="Process several images in one request",
    status_code=status.HTTP_200_OK,
)
async def remove_background_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Up to 10 source images"),
    output_format: str | None = Form(None),
    background: str | None = Form(None),
    feather: float = Form(0.0),
    threshold: float | None = Form(None),
    edge_contrast: float = Form(0.0),
    max_output_dimension: int | None = Form(None),
    upload: bool = Form(False),
    include_base64: bool = Form(False),
    identity: str = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> dict:
    if len(files) > 10:
        raise ApiError("At most 10 files per batch request.", code="too_many_files")

    # One rate-limit token per image, so a batch cannot bypass the limit.
    for _ in files:
        enforce_rate_limit(request, identity)

    options = _options_from_form(
        output_format, background, feather, threshold, edge_contrast,
        max_output_dimension, upload,
    )

    results: list[dict] = []
    for upload_file in files:
        name = upload_file.filename or "image"
        try:
            data = await _read_upload(upload_file, settings)
            result = await _run_pipeline(data, options, settings, upload_file.content_type)
            storage_url = await _maybe_upload(result, name, options)
            results.append(
                {
                    "filename": name,
                    "ok": True,
                    "output_filename": _output_filename(name, result.extension),
                    "content_type": result.media_type,
                    "bytes": len(result.payload),
                    "image_base64": (
                        base64.b64encode(result.payload).decode()
                        if include_base64
                        else None
                    ),
                    "storage_url": storage_url,
                    "meta": result.meta.model_dump(),
                    "timings_ms": result.timings_ms,
                }
            )
        except ApiError as exc:
            # One bad file must not fail the whole batch.
            results.append(
                {"filename": name, "ok": False, "error": {"code": exc.code, "message": exc.message}}
            )

    return {
        "request_id": request_id_var.get() or "",
        "count": len(results),
        "succeeded": sum(1 for r in results if r["ok"]),
        "results": results,
    }


# --------------------------------------------------------------------------- #
# URL endpoint - disabled unless RMBG_ALLOW_REMOTE_FETCH=true.
# --------------------------------------------------------------------------- #
@router.post(
    "/remove/url",
    summary="Fetch an image by URL, remove the background, return the bytes",
    response_class=Response,
)
async def remove_background_from_url(
    request: Request,
    payload: RemoveUrlRequest,
    identity: str = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> Response:
    enforce_rate_limit(request, identity)

    data, content_type = await asyncio.to_thread(
        fetch_image_bytes, str(payload.image_url), settings
    )
    options = ProcessingOptions(**payload.model_dump(exclude={"image_url"}))
    result = await _run_pipeline(data, options, settings, content_type)

    filename = _output_filename(
        PurePosixPath(str(payload.image_url).split("?")[0]).name, result.extension
    )
    return _binary_response(result, filename)
