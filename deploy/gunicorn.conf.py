"""Gunicorn configuration.

Read by `gunicorn app.main:app -c deploy/gunicorn.conf.py`.
"""

import os

bind = f"{os.getenv('RMBG_HOST', '127.0.0.1')}:{os.getenv('RMBG_PORT', '8000')}"

# ONE worker. Each one loads its own copy of the model (~1.5 GB resident), so
# a second worker doubles memory for no throughput gain - torch already uses
# every core inside a single process. Scale out with replicas instead.
workers = int(os.getenv("RMBG_GUNICORN_WORKERS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# Model load + warmup can take ~60s on a cold cache; don't let the arbiter
# kill the worker before it has finished booting.
timeout = int(os.getenv("RMBG_WORKER_TIMEOUT", "180"))
graceful_timeout = 60
keepalive = 10

# Recycling a worker means reloading the model, so never do it on a timer.
max_requests = 0

accesslog = None       # the app logs structured access lines itself
errorlog = "-"
loglevel = os.getenv("RMBG_LOG_LEVEL", "info").lower()
preload_app = False    # must stay False: the model loads in the app lifespan

if workers > 1:
    print(
        f"WARNING: {workers} gunicorn workers x ~1.5GB model = "
        f"~{workers * 1.5:.1f}GB RAM. Verify your VPS has the headroom."
    )
