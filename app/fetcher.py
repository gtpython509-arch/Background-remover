"""Guarded download of a user-supplied image URL.

Fetching an arbitrary URL on behalf of a caller is server-side request forgery
unless it is constrained. This module:

* allows only http/https,
* resolves the hostname and refuses private, loopback, link-local, multicast
  and reserved address space,
* re-checks after every redirect (so a public host cannot 302 to 169.254.x),
* streams with a hard byte cap instead of trusting Content-Length.

It stays disabled unless ``RMBG_ALLOW_REMOTE_FETCH=true``.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.errors import ApiError, PayloadTooLarge

log = logging.getLogger(__name__)

_MAX_REDIRECTS = 3


def _reject(message: str, code: str = "invalid_url") -> ApiError:
    return ApiError(message, status_code=400, code=code)


def _assert_public_host(hostname: str, allowed_hosts: list[str]) -> None:
    if allowed_hosts and hostname.lower() not in {h.lower() for h in allowed_hosts}:
        raise _reject(f"Host {hostname!r} is not in the allow-list.", "host_not_allowed")

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise _reject(f"Could not resolve host {hostname!r}: {exc}") from None

    if not infos:
        raise _reject(f"Host {hostname!r} resolved to no addresses.")

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise _reject(
                f"Host {hostname!r} resolves to a non-public address and is blocked.",
                "blocked_address",
            )


def _validate(url: str, settings: Settings) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise _reject("Only http and https URLs are accepted.")
    if not parsed.hostname:
        raise _reject("URL has no host.")
    _assert_public_host(parsed.hostname, settings.remote_fetch_allowed_hosts)


def fetch_image_bytes(url: str, settings: Settings) -> tuple[bytes, str | None]:
    """Download an image. Blocking - call from a worker thread."""
    if not settings.allow_remote_fetch:
        raise ApiError(
            "Fetching images by URL is disabled on this server.",
            status_code=403,
            code="remote_fetch_disabled",
        )

    limit = settings.max_upload_bytes
    current = url

    for _ in range(_MAX_REDIRECTS + 1):
        _validate(current, settings)
        with httpx.Client(
            timeout=settings.remote_fetch_timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": "rmbg-service/1.0", "Accept": "image/*"},
        ) as client, client.stream("GET", current) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise _reject("Redirect without a Location header.")
                # Loop back so the new host is validated before we follow it.
                current = str(response.url.join(location))
                continue

            if response.status_code >= 400:
                raise ApiError(
                    f"Source returned HTTP {response.status_code}.",
                    status_code=502,
                    code="fetch_failed",
                )

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise PayloadTooLarge(
                    f"Remote image is {int(declared)} bytes; limit is {limit}."
                )

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes(64 * 1024):
                total += len(chunk)
                if total > limit:
                    raise PayloadTooLarge(
                        f"Remote image exceeds the {limit} byte limit."
                    )
                chunks.append(chunk)

            return b"".join(chunks), response.headers.get("content-type")

    raise _reject("Too many redirects.", "too_many_redirects")
