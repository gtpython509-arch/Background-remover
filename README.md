# RMBG Background Removal Service

Production HTTP API for background removal, built on `briaai/RMBG-1.4`. This is
the Colab notebook rebuilt as a deployable service: the same model and the same
pre/post-processing, wrapped in a FastAPI app with bounded concurrency, input
validation, authentication, rate limiting, structured logs, metrics and
container/VPS deployment files.

```
POST /v1/remove   ->  multipart image in, transparent PNG out
```

---

## Read this before you deploy

**1. Hostinger shared/Business hosting will not run this.** PyTorch needs
~1.5 GB of disk and ~1.5 GB of resident RAM. Shared plans cannot install torch
and will kill the process on memory limits. You need a **Hostinger VPS**:

| Plan | vCPU | RAM | Verdict |
|---|---|---|---|
| Shared / Business / Cloud hosting | – | – | Will not work |
| VPS KVM 1 | 1 | 4 GB | Works, ~3–6 s per image. Tight; enable swap. |
| **VPS KVM 2** | **2** | **8 GB** | **Recommended minimum.** ~1.5–3 s per image. |
| VPS KVM 4 | 4 | 16 GB | ~1–1.5 s per image, comfortable headroom. |

<details>
<summary><strong>If you only have Hostinger shared hosting</strong> (click to expand)</summary>

You cannot run the model there, but you can still ship the feature: keep your
site on shared hosting and run this service somewhere that can hold a model.

1. Deploy this service on a separate box — a Hostinger VPS, or any small
   cloud instance — on its own subdomain, with `RMBG_API_KEYS` set.
2. On your shared-hosting site, add a thin PHP proxy that forwards the upload
   and injects the key server-side (see
   [Calling it from your front-end](#calling-it-from-your-front-end)).
3. Your front-end calls your own domain; the key never reaches the browser.

If you do not want to run a server at all, the same front-end can point at a
hosted inference provider instead — but then the per-image cost and the data
handling are theirs, not yours.

</details>

**2. `RMBG-1.4` is licensed CC BY-NC 4.0 — non-commercial use only.** If this
service will earn money, you need a commercial licence from BRIA, or a
permissively licensed model. Nothing in the code is model-specific beyond the
output-shape handling, so swap it with `RMBG_MODEL_ID` and no code changes.

**3. Set `RMBG_API_KEYS` before exposing the service.** Without it, anyone who
finds the URL gets free GPU/CPU time on your box. The app logs a warning at
startup when auth is off.

---

## Project layout

```
rmbg-service/
├── app/
│   ├── main.py              # app factory, lifespan (loads the model once)
│   ├── config.py            # every setting, env-driven, validated
│   ├── model.py             # the network: load, warmup, predict_mask
│   ├── pipeline.py          # decode -> segment -> refine -> encode
│   ├── imaging.py           # decode/validate/encode, bomb guards, compositing
│   ├── runner.py            # bounded thread pool; 503 instead of OOM
│   ├── schemas.py           # request/response models
│   ├── security.py          # API-key auth, real-client-IP resolution
│   ├── ratelimit.py         # sliding-window limiter
│   ├── storage.py           # optional S3/Backblaze B2 upload
│   ├── fetcher.py           # guarded fetch-by-URL (SSRF protections)
│   ├── middleware.py        # request ids, access logs, size cap, headers
│   ├── errors.py            # one JSON error envelope for every failure
│   ├── metrics.py           # Prometheus text endpoint, no extra deps
│   ├── logging_config.py    # JSON logs
│   └── static/index.html    # drag-and-drop test page
├── postman/
│   ├── rmbg-api.postman_collection.json   # 14 requests, examples + tests
│   ├── rmbg-local.postman_environment.json
│   └── openapi.json         # regenerate with `make openapi`
├── scripts/
│   ├── download_model.py    # pre-fetch weights (build step)
│   ├── export_openapi.py    # dump the schema without starting the server
│   └── batch.py             # offline folder processing (notebook cells 4-5)
├── deploy/
│   ├── Dockerfile           # multi-stage, CPU torch, non-root, weights baked in
│   ├── docker-compose.yml
│   ├── gunicorn.conf.py
│   ├── nginx.conf           # TLS, limits, timeouts for slow inference
│   ├── rmbg.service         # systemd unit (non-Docker path)
│   └── setup-vps.sh         # provisions a fresh Ubuntu VPS
├── tests/
├── requirements.txt         # pinned
├── .env.example
└── Makefile
```

### What changed from the notebook, and why

| Notebook | Service | Reason |
|---|---|---|
| Model loaded in a cell | Loaded once in the app lifespan + warmed up | A 20–60 s load per request is not an API |
| `files.upload()` | Multipart endpoint with a streamed byte cap | Colab's picker does not exist on a server |
| Unbounded image size | Pixel/dimension/byte limits, bomb guard | A 100 MP upload is a free OOM kill |
| Per-pixel Python checkerboard | Tiled, and only in the UI | It was O(w·h) in Python per preview |
| `torch.no_grad()` | `inference_mode()` + a lock + a bounded pool | Keeps peak RAM at one graph; sheds load as 503 |
| No auth | API keys + rate limiting | It is on the public internet now |
| Hardcoded B2 keys in a cell | Env-driven, lazily constructed client | Keys do not belong in source |
| `print()` | JSON logs with request ids | So you can debug a 3 a.m. incident |
| Crash = traceback | Typed errors, one JSON envelope | Clients need to branch on a code |

The maths is unchanged: resize to 1024×1024 bilinear, scale to 0–1, subtract
mean 0.5, forward pass, take `d1`, bilinear resize back to the source size,
min–max normalise, write as 8-bit alpha. With default options the output is
byte-identical to the notebook's.

---

## Local development (VS Code)

**Python 3.11 or 3.12** is the tested target (the Docker image uses 3.11).
3.13 works — `requirements.txt` selects a numpy 2.x wheel for it — but prefer
3.11/3.12 locally so your machine matches production.

```bash
cd rmbg-service
cp .env.example .env
```

Edit `.env` — for local work set:

```
RMBG_ENVIRONMENT=development
RMBG_LOG_JSON=false
RMBG_MODEL_CACHE_DIR=./models
RMBG_API_KEYS=
```

Then:

```bash
make install-dev
```

```bash
make dev
```

Open <http://127.0.0.1:8000> for the drag-and-drop page, or
<http://127.0.0.1:8000/docs> for the OpenAPI UI (both are disabled when
`RMBG_ENVIRONMENT=production`).

Without `make`:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/bin/uvicorn app.main:app --reload
```

In VS Code: pick `.venv` as the interpreter, then **Run and Debug → "Run API
(reload)"**. The launch config reads your `.env` automatically.

The first start downloads ~180 MB of weights. Pre-fetch them with
`make model`, then set `RMBG_LOCAL_FILES_ONLY=true` so startup never touches
the network.

---

## Testing with Postman

```bash
make dev
```

`make dev` starts the server with **auth off, `/docs` on and rate limiting
disabled** — the right setup for poking at it. It prints a warning that auth is
disabled; that is expected locally.

Then in Postman, **Import** both files from `postman/`:

| File | What it is |
|---|---|
| `rmbg-api.postman_collection.json` | 14 ready-made requests with example bodies and tests |
| `rmbg-local.postman_environment.json` | Sets `baseUrl` to `http://127.0.0.1:8000` |

Select the **RMBG - Local** environment (top-right), open
**Background removal → Remove background (PNG)**, go to the **Body** tab, click
**Select Files** next to `file`, pick an image, and Send. The cutout renders in
Postman's preview pane; use **Save Response → Save to a file** to keep the PNG.

The collection is organised as:

* **Ops** — `/healthz`, `/readyz`, `/v1/info`, `/metrics`
* **Background removal** — PNG, JPEG-on-white, mask-only, JSON+base64, batch, by-URL
* **Error cases** — 415, 400, 422 and 401, so you can see the error envelope

Optional parameters (`background`, `feather`, `threshold`, `edge_contrast`,
`max_output_dimension`) are present but **switched off**, so defaults match the
original notebook. Tick a checkbox in the Body tab to use one.

### About the API key

Auth is off whenever `RMBG_API_KEYS` is empty — which is the default. While it
is off, a wrong or empty `X-API-Key` header is simply ignored, so the collection
works untouched. When you set keys on the server, paste one into the `apiKey`
environment variable and every request in the collection picks it up from the
collection-level auth; nothing else changes.

> Leaving auth off is fine on `127.0.0.1`. It is not fine on a public host —
> anyone who finds the URL gets free CPU time on your VPS.

### Prefer the raw OpenAPI schema?

```bash
make openapi
```

Writes `postman/openapi.json`, which Postman can import directly
(**Import → File**). It is generated from the route signatures, so it is always
in sync with the code — but it has no example bodies or tests, unlike the
collection. The live schema is also at `/openapi.json` and the Swagger UI at
`/docs` whenever docs are enabled.

`/docs` and `/openapi.json` are **off in production** by default. To reach them
on a deployed host, set `RMBG_ENABLE_DOCS=true` — it publishes your full
endpoint and parameter list, so importing the committed collection is usually
the better option.

---

## API

All endpoints accept the key as `X-API-Key: <key>` or
`Authorization: Bearer <key>` when `RMBG_API_KEYS` is set.

### `POST /v1/remove` — image in, image out

`multipart/form-data`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | required | PNG, JPEG, WebP, BMP, TIFF |
| `output_format` | string | `png` | `png`, `webp`, `jpeg`, `mask` |
| `background` | string | – | `#ffffff`, `white`, … Forced opaque behind the subject |
| `feather` | float | `0` | Gaussian edge softening, 0–25 px |
| `threshold` | float | – | Hard-cut the matte to binary at this alpha, 0–1 |
| `edge_contrast` | float | `0` | Pushes mid-grey alpha to 0/255. Try `2`–`4` for hair halo |
| `max_output_dimension` | int | – | Downscale the result's longest edge |

```bash
curl -X POST https://api.yourdomain.com/v1/remove \
  -H "X-API-Key: $RMBG_KEY" \
  -F "file=@photo.jpg" \
  -o cutout.png
```

Response headers include `X-Request-ID`, `X-Image-Width`, `X-Image-Height` and
`X-Inference-Ms`.

### `POST /v1/remove/json` — base64 and/or a storage URL

Same fields, plus `upload=true` (requires object storage) and
`include_base64=false`.

```json
{
  "request_id": "0f3c…",
  "filename": "photo_cutout.png",
  "content_type": "image/png",
  "bytes": 284113,
  "image_base64": "iVBORw0KG…",
  "storage_url": null,
  "meta": {"width": 1200, "height": 800, "original_width": 1200,
           "original_height": 800, "source_format": "JPEG", "was_downscaled": false},
  "timings_ms": {"decode": 18.4, "inference": 1430.2, "postprocess": 41.9, "encode": 102.6}
}
```

### `POST /v1/remove/batch` — up to 10 files

Returns a JSON array. One bad file fails only its own entry; each image costs
one rate-limit token.

### `POST /v1/remove/url` — fetch by URL

Disabled unless `RMBG_ALLOW_REMOTE_FETCH=true`. When on, private, loopback,
link-local and reserved addresses are refused, every redirect is re-checked,
and the download is capped mid-stream. Restrict it further with
`RMBG_REMOTE_FETCH_ALLOWED_HOSTS`.

### Ops endpoints

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness. 200 while the process lives |
| `GET /readyz` | Readiness. 503 until the model is loaded |
| `GET /v1/info` | Model, device, limits, which features are on |
| `GET /metrics` | Prometheus text. Restricted to localhost in `nginx.conf` |

### Errors

Every failure uses one shape:

```json
{"error": {"code": "payload_too_large",
           "message": "Image is 9000x9000 (81.0 MP); the limit is 50.0 MP.",
           "request_id": "0f3c…"}}
```

| Status | `code` | Meaning |
|---|---|---|
| 400 | `invalid_options`, `invalid_url`, `storage_disabled` | Fix the request |
| 401 | `unauthorized` | Missing or wrong API key |
| 413 | `payload_too_large` | Over the byte or pixel limit |
| 415 | `unsupported_media_type` | Not a decodable image |
| 422 | `validation_error` | Missing/malformed field |
| 429 | `rate_limited` | Back off; honour `Retry-After` |
| 503 | `model_not_ready`, `overloaded` | Retry; the queue is full or still booting |
| 504 | `inference_timeout` | Image too large for the time budget |

---

## Deploying to a Hostinger VPS

### 1. Point DNS at the server

In Hostinger's DNS panel add an `A` record for `api` → your VPS IPv4. Wait for
it to resolve before requesting a certificate.

### 2. Provision the host

SSH in as root, then:

```bash
apt-get update && apt-get install -y git
```

```bash
git clone <your-repo-url> /srv/rmbg-service && cd /srv/rmbg-service
```

```bash
bash deploy/setup-vps.sh api.yourdomain.com
```

That installs Docker and nginx, creates a 2 GB swap file (torch's peak
allocation during load can exceed free RAM on a 4 GB box — swap turns an OOM
kill into a slow start), configures `ufw`, and gets a Let's Encrypt
certificate. Read it before you run it.

### 3. Configure

```bash
cp .env.example .env && nano .env
```

Set at minimum:

```
RMBG_ENVIRONMENT=production
RMBG_API_KEYS=<paste a generated key>
RMBG_CORS_ORIGINS=https://yourdomain.com
RMBG_TORCH_THREADS=2
RMBG_TRUSTED_PROXY_COUNT=1
```

Generate a key with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4. Build and start

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

The build downloads torch and bakes the weights into the image — **expect
10–20 minutes** on a 2-vCPU VPS. Watch it:

```bash
docker compose -f deploy/docker-compose.yml logs -f
```

### 5. Verify

```bash
curl https://api.yourdomain.com/healthz
```

```bash
curl -X POST https://api.yourdomain.com/v1/remove -H "X-API-Key: $KEY" -F "file=@test.jpg" -o out.png
```

`/readyz` returns 503 until the model finishes loading — that is correct, not a
failure. Give it ~60 s after the container starts.

### Without Docker

```bash
sudo useradd -r -s /usr/sbin/nologin rmbg && sudo mkdir -p /var/lib/rmbg && sudo chown rmbg:rmbg /var/lib/rmbg
```

```bash
cd /srv/rmbg-service && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

```bash
sudo -u rmbg .venv/bin/python scripts/download_model.py
```

```bash
sudo cp deploy/rmbg.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now rmbg
```

Then install `deploy/nginx.conf` as described in its header comment.

---

## Calling it from your front-end

```js
async function removeBackground(file) {
  const body = new FormData();
  body.append("file", file);
  body.append("output_format", "png");

  const res = await fetch("https://api.yourdomain.com/v1/remove", {
    method: "POST",
    headers: { "X-API-Key": API_KEY },   // see the warning below
    body,
  });

  if (!res.ok) {
    const { error } = await res.json();
    throw new Error(error.message);
  }
  return URL.createObjectURL(await res.blob());
}
```

**Do not put the API key in browser JavaScript** — anyone can read it in
DevTools. Either proxy through your own backend (PHP/Node on Hostinger) and
keep the key server-side, or issue short-lived per-user keys. The bundled test
page asks for the key by hand precisely because it is a test page.

PHP proxy sketch, if your site is PHP on Hostinger:

```php
<?php
$ch = curl_init('http://127.0.0.1:8000/v1/remove');
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_HTTPHEADER => ['X-API-Key: ' . getenv('RMBG_API_KEY')],
    CURLOPT_POSTFIELDS => ['file' => new CURLFile($_FILES['image']['tmp_name'])],
    CURLOPT_RETURNTRANSFER => true,
]);
$png = curl_exec($ch);
header('Content-Type: image/png');
echo $png;
```

---

## Batch processing offline

The notebook's upload-loop-and-zip flow, as a CLI:

```bash
.venv/bin/python scripts/batch.py ./input -o ./output --zip cutouts.zip
```

```bash
.venv/bin/python scripts/batch.py ./photos -o ./out --format webp --feather 1.5 --recursive
```

It loads the model once for the whole run, skips existing outputs unless you
pass `--overwrite`, and reports per-image timings.

---

## Operations

### Tuning

| Symptom | Change |
|---|---|
| Slow single requests | Raise `RMBG_TORCH_THREADS` to your vCPU count |
| 503 `overloaded` under load | Raise `RMBG_MAX_QUEUED_REQUESTS`, or add a replica |
| Memory pressure | Lower `RMBG_MAX_INPUT_DIMENSION` (2048 is plenty for web) |
| Output too large | `output_format=webp`, or set `max_output_dimension` |
| Halo around hair | `edge_contrast=3`, or `feather=1` |

Leave `RMBG_INFERENCE_WORKERS=1` and gunicorn at one worker unless you have
measured spare RAM: each extra copy of the model is ~1.5 GB and torch already
uses every core inside one process.

### Monitoring

```bash
curl -s localhost:8000/metrics | grep rmbg_
```

Exposed: `rmbg_http_requests_total`, `rmbg_http_request_seconds`,
`rmbg_images_processed_total`, `rmbg_inference_seconds`,
`rmbg_inference_in_flight`, `rmbg_requests_rejected_total`,
`rmbg_model_loaded`, `rmbg_uptime_seconds`.

Counters are per-process; with multiple replicas, scrape each one.

### Logs

```bash
docker compose -f deploy/docker-compose.yml logs -f --tail 100
```

```bash
sudo journalctl -u rmbg -f          # systemd install
```

Every line is JSON with a `request_id` that matches the `X-Request-ID` header,
so a user's bug report maps to exact log lines.

### Updating

```bash
cd /srv/rmbg-service && git pull && docker compose -f deploy/docker-compose.yml up -d --build
```

Compose keeps the old container until the new one is healthy. Rolling back is
`git checkout <previous-tag>` and the same command.

---

## Tests

```bash
make test
```

```bash
make lint typecheck
```

The suite stubs the model, so it runs in about a second and needs no weights —
it covers the validation and HTTP contract, not segmentation quality.

---

## Troubleshooting

**`Killed` during startup, or the container restarts in a loop.** Out of
memory. Check `free -h`; add swap (`setup-vps.sh` does this) or move up a VPS
tier.

**`No space left on device` while building.** Torch plus weights needs ~4 GB
free. `docker system prune -af` then rebuild.

**`/readyz` stays 503.** Read the logs. Usually the weights could not be
downloaded (no network, or `RMBG_LOCAL_FILES_ONLY=true` with an empty cache).

**504 from nginx on big images.** Raise `proxy_read_timeout` in
`deploy/nginx.conf` and `RMBG_INFERENCE_TIMEOUT_SECONDS` together.

**413 on an upload you think is fine.** Both limits must allow it:
`client_max_body_size` in nginx and `RMBG_MAX_UPLOAD_BYTES` in `.env`.

**CORS errors in the browser.** `RMBG_CORS_ORIGINS` must list the exact scheme
and host of your site, comma-separated, no trailing slash.

**Fuzzy or wrong cutouts.** The model is the limit, not the code. Confirm by
running the same image through the notebook; if they match, you need a
different model or a matting refinement pass.
