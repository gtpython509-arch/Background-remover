"""Optional S3-compatible upload (Backblaze B2, R2, Wasabi, MinIO, AWS S3).

This is the production form of the notebook's B2 cell. Credentials come from
the environment, never from the request. boto3 is imported lazily so the
service runs without it when storage is disabled.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime

from app.config import Settings

log = logging.getLogger(__name__)


class StorageError(RuntimeError):
    pass


class ObjectStorage:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._settings.storage_enabled

    def _get_client(self):
        # botocore clients are thread-safe once constructed, but construction
        # itself is not - hence the lock around first use only.
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:  # pragma: no cover
                raise StorageError(
                    "storage is enabled but boto3 is not installed "
                    "(pip install -r requirements-storage.txt)"
                ) from exc

            settings = self._settings
            self._client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint_url,
                region_name=settings.s3_region,
                aws_access_key_id=settings.s3_access_key_id,
                aws_secret_access_key=settings.s3_secret_access_key,
                config=Config(
                    retries={"max_attempts": 3, "mode": "standard"},
                    connect_timeout=5,
                    read_timeout=30,
                    signature_version="s3v4",
                ),
            )
            log.info(
                "s3 client ready",
                extra={
                    "endpoint": settings.s3_endpoint_url or "aws",
                    "bucket": settings.s3_bucket,
                },
            )
            return self._client

    def build_key(self, filename: str, extension: str) -> str:
        from pathlib import PurePosixPath

        stem = PurePosixPath(filename or "image").stem[:60] or "image"
        safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in stem).strip("-")
        stamp = datetime.now(UTC).strftime("%Y/%m/%d")
        unique = uuid.uuid4().hex[:12]
        return f"{self._settings.s3_key_prefix}{stamp}/{safe or 'image'}-{unique}.{extension}"

    def upload(
        self,
        payload: bytes,
        key: str,
        content_type: str,
        *,
        cache_seconds: int = 31_536_000,
    ) -> str:
        """Upload and return a URL. Blocking - call it from a worker thread."""
        settings = self._settings
        if not settings.storage_enabled:
            raise StorageError("storage is not enabled")

        client = self._get_client()
        try:
            client.put_object(
                Bucket=settings.s3_bucket,
                Key=key,
                Body=payload,
                ContentType=content_type,
                CacheControl=f"public, max-age={cache_seconds}, immutable",
            )
        except Exception as exc:
            log.exception("upload failed", extra={"key": key})
            raise StorageError(f"upload to object storage failed: {exc}") from exc

        if settings.s3_public_base_url:
            return f"{settings.s3_public_base_url.rstrip('/')}/{key}"

        if settings.s3_presign_seconds > 0:
            try:
                return client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": settings.s3_bucket, "Key": key},
                    ExpiresIn=settings.s3_presign_seconds,
                )
            except Exception as exc:  # pragma: no cover
                raise StorageError(f"could not presign {key}: {exc}") from exc

        base = (settings.s3_endpoint_url or "https://s3.amazonaws.com").rstrip("/")
        return f"{base}/{settings.s3_bucket}/{key}"

    def health(self) -> bool:
        """head_bucket round-trip, used by the readiness probe."""
        if not self._settings.storage_enabled:
            return True
        try:
            self._get_client().head_bucket(Bucket=self._settings.s3_bucket)
            return True
        except Exception as exc:
            log.warning("storage health check failed", extra={"error": str(exc)})
            return False


_storage: ObjectStorage | None = None


def get_storage() -> ObjectStorage:
    global _storage
    if _storage is None:
        from app.config import get_settings

        _storage = ObjectStorage(get_settings())
    return _storage


def set_storage(storage: ObjectStorage | None) -> None:
    global _storage
    _storage = storage
