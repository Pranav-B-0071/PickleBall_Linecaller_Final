"""JSON error handlers so the frontend always gets a consistent shape."""

from __future__ import annotations

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        if request.path.startswith("/api") or request.path.startswith("/media"):
            return jsonify({"ok": False, "error": exc.description,
                            "status": exc.code}), exc.code
        return exc

    @app.errorhandler(413)
    def _too_large(exc):
        cfg = app.config["WEB"]
        return jsonify({"ok": False,
                        "error": f"file exceeds the {cfg.max_upload_mb} MB limit"}), 413

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        if isinstance(exc, HTTPException):
            return exc
        app.logger.exception("unhandled error")
        if request.path.startswith(("/api", "/media")):
            return jsonify({"ok": False, "error": f"server error: {exc}"}), 500
        raise exc
