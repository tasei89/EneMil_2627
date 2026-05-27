# -*- coding: utf-8 -*-
"""認証・ユーザー管理モジュール"""
import sqlite3
import hashlib
import hmac
import os
import secrets
import string
from datetime import datetime, timedelta
import streamlit as st

DB_PATH = os.environ.get("DB_PATH", "db/baseball_stats.db")


# ──────────────────────────────────────────
# DB 初期化
# ──────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            role        TEXT NOT NULL DEFAULT 'user',
            position    TEXT DEFAULT '',
            pw_hash     TEXT NOT NULL,
            is_temp_pw  INTEGER NOT NULL DEFAULT 1,
            is_active   INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL,
            last_login  TEXT
        );

        CREATE TABLE IF NOT EXISTS event_log (
            rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT,
            event_type  TEXT,
            detail      TEXT,
            ts          TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            user_id     TEXT,
            expires_at  TEXT
        );
    """)
    # 管理者が1人もいなければ初期アカウントを作成
    # 初期ID: 000 / 初期PW: 123
    cur.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?)",
            ("000", "管理者", "admin", "",
             _hash_pw("123"), 1, 1,
             datetime.now().isoformat(), None)
        )
        con.commit()
    con.commit()
    con.close()


# ──────────────────────────────────────────
# パスワードユーティリティ
# ──────────────────────────────────────────
def _hash_pw(pw: str) -> str:
    salt = os.environ.get("SECRET_KEY", "baseball-secret-salt-2026")
    return hashlib.sha256((salt + pw).encode()).hexdigest()


def _gen_password(length: int = 12) -> str:
    chars = string.ascii_letters + string.digits + "!@#$"
    return "".join(secrets.choice(chars) for _ in range(length))


def validate_password(pw: str) -> bool:
    if len(pw) < 8:
        return False
    has_upper = any(c.isupper() for c in pw)
    has_lower = any(c.islower() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    return has_upper and has_lower and has_digit


# ──────────────────────────────────────────
# ログイン / ログアウト
# ──────────────────────────────────────────
def login(user_id: str, password: str) -> tuple[bool, str]:
    """(success, message)"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT pw_hash, is_active, is_temp_pw, role, display_name "
                "FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        return False, "IDまたはパスワードが正しくありません"
    pw_hash, is_active, is_temp, role, name = row
    if not is_active:
        con.close()
        return False, "このアカウントは無効化されています"
    if not hmac.compare_digest(pw_hash, _hash_pw(password)):
        con.close()
        return False, "IDまたはパスワードが正しくありません"

    cur.execute("UPDATE users SET last_login=? WHERE id=?",
                (datetime.now().isoformat(), user_id))
    _log_event(cur, user_id, "login", "")
    con.commit()
    con.close()

    st.session_state["auth_user_id"]   = user_id
    st.session_state["auth_role"]      = role
    st.session_state["auth_name"]      = name
    st.session_state["auth_is_temp"]   = bool(is_temp)
    st.session_state["auth_logged_in"] = True
    return True, "ok"


def logout():
    for k in ["auth_user_id", "auth_role", "auth_name", "auth_is_temp", "auth_logged_in"]:
        st.session_state.pop(k, None)


def is_logged_in() -> bool:
    return st.session_state.get("auth_logged_in", False)


def current_role() -> str:
    return st.session_state.get("auth_role", "")


def current_user() -> str:
    return st.session_state.get("auth_user_id", "")


def is_temp_pw() -> bool:
    return st.session_state.get("auth_is_temp", False)


# ──────────────────────────────────────────
# ユーザー CRUD
# ──────────────────────────────────────────
def get_all_users() -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT id, display_name, role, position, is_active, last_login FROM users")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def add_user(user_id: str, display_name: str, role: str,
             position: str, password: str | None = None) -> tuple[bool, str]:
    """(success, plain_password)"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id FROM users WHERE id=?", (user_id,))
    if cur.fetchone():
        con.close()
        return False, "このIDは既に使われています"
    pw = password or _gen_password()
    cur.execute(
        "INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?)",
        (user_id, display_name, role, position,
         _hash_pw(pw), 1, 1, datetime.now().isoformat(), None)
    )
    con.commit()
    con.close()
    return True, pw


def reset_password(user_id: str) -> str:
    new_pw = _gen_password()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE users SET pw_hash=?, is_temp_pw=1 WHERE id=?",
                (_hash_pw(new_pw), user_id))
    con.commit()
    con.close()
    return new_pw


def change_password(user_id: str, new_pw: str) -> bool:
    if not validate_password(new_pw):
        return False
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE users SET pw_hash=?, is_temp_pw=0 WHERE id=?",
                (_hash_pw(new_pw), user_id))
    con.commit()
    con.close()
    st.session_state["auth_is_temp"] = False
    return True


def update_user(user_id: str, display_name: str, role: str,
                position: str, is_active: bool):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE users SET display_name=?, role=?, position=?, is_active=? WHERE id=?",
                (display_name, role, position, int(is_active), user_id))
    con.commit()
    con.close()


def delete_user(user_id: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    con.commit()
    con.close()


# ──────────────────────────────────────────
# イベントログ
# ──────────────────────────────────────────
def _log_event(cur, user_id: str, event_type: str, detail: str):
    cur.execute("INSERT INTO event_log(user_id,event_type,detail,ts) VALUES(?,?,?,?)",
                (user_id, event_type, detail, datetime.now().isoformat()))


def log_event(event_type: str, detail: str = ""):
    uid = current_user() or "anonymous"
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    _log_event(cur, uid, event_type, detail)
    con.commit()
    con.close()


def get_event_log(limit: int = 500) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM event_log ORDER BY rowid DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows
