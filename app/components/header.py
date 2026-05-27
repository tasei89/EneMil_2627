# -*- coding: utf-8 -*-
"""共通ヘッダーコンポーネント。
画面上部に EneMil テキストロゴと、その下にページタイトル
（例：投手分析）を表示する。
サイドバーが畳まれていても、メインエリア側にロゴが残るのが特徴。"""
import os
import base64
from functools import lru_cache
import streamlit as st


_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
_LOGO_TEXT_CANDIDATES = ["logo_text.png", "logo_text.jpg"]
_LOGO_FULL_CANDIDATES = ["logo_full.png", "logo_full.jpg"]


def _resolve_logo(candidates: list) -> tuple:
    """候補リストから最初に存在するファイルパスと MIME タイプを返す。"""
    for fname in candidates:
        p = os.path.normpath(os.path.join(_ASSETS_DIR, fname))
        if os.path.isfile(p):
            mime = "image/png" if fname.lower().endswith(".png") else "image/jpeg"
            return p, mime
    return "", ""


@lru_cache(maxsize=4)
def _img_b64(path: str) -> str:
    """画像ファイルを base64 エンコードした文字列を返す（キャッシュ済）。
    ファイルが存在しないときは空文字列。"""
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return ""


def render_page_header(title: str, icon: str = ""):
    """ページ最上部に EneMil テキストロゴ + ページタイトルを描画。
    title:  例 "投手分析"
    icon:   例 "🏟️" （省略可）"""
    path, mime = _resolve_logo(_LOGO_TEXT_CANDIDATES)
    text_b64 = _img_b64(path)
    title_html = f"{icon}&nbsp;{title}" if icon else title

    if text_b64:
        # ロゴ画像あり：画像 + タイトル
        html = f"""
<div style="margin: 0 0 1.2rem 0;">
    <div style="text-align:center; padding: 6px 0 10px 0;">
        <img src="data:{mime};base64,{text_b64}"
             alt="EneMil"
             style="max-width: 340px; width: 60%; min-width: 220px;
                    height: auto; display: inline-block;" />
    </div>
    <div style="background: linear-gradient(135deg,
                #2C5F7C 0%, #3A6B85 50%, #4A7A93 100%);
                padding: 0.7rem 1.4rem; border-radius: 6px;
                color: white;
                box-shadow: 0 1px 3px rgba(44, 95, 124, 0.15);
                border-left: 4px solid #6FA8C9;">
        <h2 style="color: white !important; margin: 0;
                   border: none !important; font-weight: 600;
                   font-size: 1.4rem;">{title_html}</h2>
    </div>
</div>
"""
    else:
        # フォールバック（ロゴ画像が見つからないとき）：テキストのみ
        html = f"""
<div style="margin: 0 0 1.2rem 0;">
    <div style="text-align:center; padding: 6px 0;">
        <span style="font-family: -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif;
                     font-size: 2.6rem; font-weight: 800;
                     color: #1F3A5F; letter-spacing: -0.02em;">EneMil</span>
        <div style="font-size: 0.85rem; color: #4A7A93;
                    letter-spacing: 0.18em; margin-top: -4px;">
            エネミル&nbsp;&nbsp;BASEBALL ANALYTICS
        </div>
    </div>
    <div style="background: linear-gradient(135deg,
                #2C5F7C 0%, #3A6B85 50%, #4A7A93 100%);
                padding: 0.7rem 1.4rem; border-radius: 6px;
                color: white; margin-top: 8px;
                box-shadow: 0 1px 3px rgba(44, 95, 124, 0.15);
                border-left: 4px solid #6FA8C9;">
        <h2 style="color: white !important; margin: 0;
                   border: none !important; font-weight: 600;
                   font-size: 1.4rem;">{title_html}</h2>
    </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_logo():
    """サイドバー上部にフル EneMil ロゴ（球＋テキスト）を描画。"""
    path, mime = _resolve_logo(_LOGO_FULL_CANDIDATES)
    full_b64 = _img_b64(path)
    if full_b64:
        html = f"""
<div style="text-align:center; padding: 4px 0 12px 0;">
    <img src="data:{mime};base64,{full_b64}"
         alt="EneMil"
         style="max-width: 180px; width: 90%; height: auto;
                display: inline-block;" />
</div>
"""
    else:
        html = """
<div style="text-align:center; padding: 6px 0 10px 0;">
    <span style="font-family: -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif;
                 font-size: 1.8rem; font-weight: 800;
                 color: #1F3A5F; letter-spacing: -0.02em;">EneMil</span>
    <div style="font-size: 0.7rem; color: #4A7A93;
                letter-spacing: 0.15em; margin-top: -2px;">
        BASEBALL ANALYTICS
    </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_login_logo():
    """ログイン画面用のロゴ表示（フルロゴを中央大きく）。"""
    path, mime = _resolve_logo(_LOGO_FULL_CANDIDATES)
    full_b64 = _img_b64(path)
    if full_b64:
        html = f"""
<div style="text-align:center; padding: 20px 0 12px 0;">
    <img src="data:{mime};base64,{full_b64}"
         alt="EneMil"
         style="max-width: 260px; width: 60%; height: auto;
                display: inline-block;" />
</div>
"""
    else:
        html = """
<div style="text-align:center; padding: 20px 0 12px 0;">
    <span style="font-family: -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif;
                 font-size: 3rem; font-weight: 800;
                 color: #1F3A5F; letter-spacing: -0.02em;">EneMil</span>
    <div style="font-size: 0.9rem; color: #4A7A93;
                letter-spacing: 0.18em; margin-top: -4px;">
        エネミル&nbsp;&nbsp;BASEBALL ANALYTICS
    </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
