"""Shared request/response helpers for the API blueprints."""

from __future__ import annotations

from functools import wraps

from flask import current_app, jsonify, request

from ..config import WebConfig
from ..services.storage import SessionStore, StorageError


def cfg() -> WebConfig:
    return current_app.config["WEB"]


def store() -> SessionStore:
    return SessionStore(cfg())


def session_id() -> str | None:
    """Resolve the session id from header, JSON body, form, or query."""
    sid = request.headers.get("X-Session-Id")
    if not sid and request.is_json:
        sid = (request.get_json(silent=True) or {}).get("session_id")
    if not sid:
        sid = request.form.get("session_id") or request.args.get("session_id")
    return sid


def ok(**payload):
    return jsonify({"ok": True, **payload})


def err(message: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), status


def with_session(fn):
    """Inject a validated (sid, SessionStore) into the view; 404 if unknown."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        sid = session_id()
        st = store()
        if not sid or not st.exists(sid):
            return err("no active session - reload the page", 404)
        try:
            return fn(sid, st, *args, **kwargs)
        except StorageError as exc:
            return err(str(exc), 400)

    return wrapper
