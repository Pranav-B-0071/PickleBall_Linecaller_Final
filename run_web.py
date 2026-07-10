"""Entry point for the Pickleball Linecaller web app.

    python run_web.py            # http://127.0.0.1:5001  (host/port from config.yaml)

Everything else is configured under the ``webapp:`` section of ``config.yaml``.
"""

from __future__ import annotations

import os
import shutil
import threading
import webbrowser

from webapp import create_app
from webapp.config import WebConfig


def main() -> None:
    cfg = WebConfig.load()

    # Fresh start on every launch: drop previous session uploads + state so the
    # app behaves like a first run. Guarded by WERKZEUG_RUN_MAIN so it only runs
    # on the initial start, NOT on the debug reloader's hot-restarts (which would
    # wipe your session mid-edit).
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        shutil.rmtree(cfg.data_root, ignore_errors=True)

    app = create_app(cfg)

    # Auto-open the app in the default browser. Guarded by WERKZEUG_RUN_MAIN so
    # it fires exactly once (in the reloader's supervisor process, not the
    # worker it respawns on every code change), after a short delay so the
    # server is already listening. 0.0.0.0 isn't browsable -> use localhost.
    browse_host = "127.0.0.1" if cfg.host in ("0.0.0.0", "") else cfg.host
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        url = f"http://{browse_host}:{cfg.port}/"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    # threaded: the real Page-3 analysis is a long (~minutes on CPU) blocking
    # request; without this it would freeze video streaming + session polling.
    app.run(host=cfg.host, port=cfg.port, debug=True, threaded=True)


if __name__ == "__main__":
    main()
