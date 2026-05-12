from __future__ import annotations

from typing import Optional

from flask import Request

from auth_store import AuthStore


def _get_form_value(req: Request, key: str) -> str:
    return (req.form.get(key) or "").strip()


def try_signup(req: Request, store: AuthStore) -> tuple[bool, str]:
    username = _get_form_value(req, "username")
    password = _get_form_value(req, "password")

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    ok = store.create_user(username, password)
    if not ok:
        return False, "User already exists."
    return True, "Signup successful."


def try_login(req: Request, store: AuthStore) -> tuple[bool, str]:
    username = _get_form_value(req, "username")
    password = _get_form_value(req, "password")

    ok = store.verify_user(username, password)
    if not ok:
        return False, "Invalid username or password."
    return True, "Login successful."

