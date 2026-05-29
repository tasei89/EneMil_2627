# -*- coding: utf-8 -*-
"""
野球スタッツ可視化アプリ — エントリポイント
"""
import os
import streamlit as st
import pandas as pd

# ページ設定（最初に呼ぶ）
st.set_page_config(
    page_title="EneMil",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="auto",  # PC: expanded, モバイル: collapsed
)

# パス設定
import sys
sys.path.insert(0, os.path.dirname(__file__))

from app.auth import init_db, login, logout, is_logged_in, current_role, is_temp_pw, change_password, validate_password

# カスタム CSS（Baseball Savant 風ブルーグレー基調）
st.markdown("""
<style>
/* ====================================
   Baseball Savant 風 カラーパレット
   ・プライマリ:  #2C5F7C  (深ブルーグレー)
   ・セカンダリ:  #4A7A93  (中ブルーグレー)
   ・アクセント:  #6FA8C9  (淡ブルー)
   ・背景:        #F4F6F8  (極淡グレー)
   ・テキスト:    #1F2A37  (濃スレート)
   ・罫線:        #D6DEE5  (淡ブルーグレー)
==================================== */

/* ダークモード強制無効化（iPad/iPhone のダークモード対策） */
:root { color-scheme: light !important; }
html, body {
    color-scheme: light !important;
    background-color: #FFFFFF !important;
    color: #1F2A37 !important;
}
@media (prefers-color-scheme: dark) {
    html, body, .stApp,
    section[data-testid="stSidebar"],
    .main, .main .block-container {
        background-color: #FFFFFF !important;
        color: #1F2A37 !important;
    }
    p, span, div, label, h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p, .stCaption,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    [data-testid="stText"],
    .stRadio label, .stSelectbox label,
    .stMultiSelect label, .stTextInput label,
    .stDateInput label, .stNumberInput label,
    .stCheckbox label {
        color: #1F2A37 !important;
    }
    .js-plotly-plot, .plot-container, .svg-container {
        background-color: #FFFFFF !important;
    }
}

/* ─── 全体背景 ─── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
.stMain {
    background-color: #FFFFFF !important;
    color: #1F2A37 !important;
}
.main .block-container {
    padding-top: 1.2rem;
    max-width: 1280px;
    background-color: #FFFFFF !important;
}

/* ─── Plotly チャート背景を常に白に強制（iPadダークモード対策） ─── */
/* ★ SVG 要素 (.main-svg / .svg-container) には触らない。
   そこに CSS を当てると Plotly の auto-sizing が壊れることがある。
   コンテナ div のみ白指定。 */
div[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF !important;
}
.js-plotly-plot .modebar { background: transparent !important; }

/* ─── 既定のテキスト色（全モードで明示） ─── */
body, p, span, div, label,
.stMarkdown, .stMarkdown p,
[data-testid="stMarkdownContainer"],
[data-testid="stText"] {
    color: #1F2A37;
}

/* ─── サイドバー ─── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #D6DEE5;
}
/* サイドバー内のコラプス時のフッターやヘッダー部分も白に統一 */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #FFFFFF;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-weight: 600;
    color: #2C5F7C;
    font-size: 13px;
}

/* ─── タイトル・見出し ─── */
h1, h2, h3, h4 {
    color: #1F2A37;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Helvetica Neue", "Hiragino Sans", "Yu Gothic UI", sans-serif;
}
h2 { color: #1F2A37 !important;
     border-bottom: 2px solid #2C5F7C; padding-bottom: 4px;
     margin-bottom: 1rem; }
h3 { color: #2C5F7C !important; margin-top: 1.2rem; }
h4 { color: #4A7A93 !important; font-weight: 600; margin-top: 0.8rem; }

/* ─── サイドバーのラジオボタン（ナビメニュー）テキスト色 ─── */
/* iPad/iPhone のダークモードで透けて見えない事象を防ぐ */
section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label > div,
section[data-testid="stSidebar"] [role="radiogroup"] label p,
section[data-testid="stSidebar"] [role="radiogroup"] label span,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio label p,
section[data-testid="stSidebar"] .stRadio label span,
section[data-testid="stSidebar"] .stRadio label > div {
    color: #1F2A37 !important;
    font-weight: 500;
}
/* セレクトボックスやテキスト入力のテキストも明示 */
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-baseweb="input"] *,
section[data-testid="stSidebar"] [data-baseweb="select"] div[role="button"],
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] *,
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] *,
section[data-testid="stSidebar"] [data-testid="stTextInput"] *,
section[data-testid="stSidebar"] [data-testid="stDateInput"] *,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] * {
    color: #1F2A37 !important;
}
/* メイン側のセレクトボックスも同様 */
.main [data-baseweb="select"] *,
.main [data-baseweb="input"] *,
.main [data-baseweb="select"] div[role="button"],
.main input,
.main [data-testid="stSelectbox"] *,
.main [data-testid="stMultiSelect"] *,
.main [data-testid="stTextInput"] *,
.main [data-testid="stDateInput"] *,
.main [data-testid="stNumberInput"] * {
    color: #1F2A37 !important;
}
/* ─── multiselect の「選択済みチップ」だけは白文字に上書き ─── */
/* チップ背景がブルーグレーになっているため、上の一括ルールに上書きして白文字にする */
[data-testid="stMultiSelect"] [data-baseweb="tag"],
[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
[data-testid="stMultiSelect"] span[data-baseweb="tag"] span,
[data-testid="stMultiSelect"] div[data-baseweb="tag"] span,
.stMultiSelect [data-baseweb="tag"],
.stMultiSelect [data-baseweb="tag"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
/* チップの×ボタン（svg）も白に */
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg path,
.stMultiSelect [data-baseweb="tag"] svg,
.stMultiSelect [data-baseweb="tag"] svg path {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
/* セレクトボックスの背景は白で固定 */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: #FFFFFF !important;
}
/* セレクトのドロップダウンのテキスト */
[data-baseweb="popover"] li,
[data-baseweb="popover"] li * {
    color: #1F2A37 !important;
    background-color: #FFFFFF !important;
}
[data-baseweb="popover"] li:hover {
    background-color: #EEF2F5 !important;
}

/* ─── ラジオボタン（メインエリア横並びオプション）も統一 ─── */
.main [role="radiogroup"] label,
.main .stRadio label,
.main .stRadio label > div,
.main .stRadio label p,
.main .stRadio label span {
    color: #1F2A37 !important;
}

/* ─── レスポンシブ（タブレット・スマホ対応） ─── */

/* タブレット（768px〜1024px）：本文のサイドパディングを絞る */
@media (max-width: 1024px) {
    .main .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 100% !important;
    }
}

/* スマホ（768px 以下）：レイアウト・フォントサイズ・テーブルを大幅最適化 */
@media (max-width: 768px) {
    /* 本文余白をさらに切り詰める */
    .main .block-container {
        padding: 0.6rem 0.4rem !important;
        max-width: 100% !important;
    }

    /* サイドバーは閉じる前提だが、開いた時に画面いっぱい使えるように */
    section[data-testid="stSidebar"] {
        min-width: 75vw !important;
        max-width: 90vw !important;
    }
    /* サイドバートグル（折りたたみボタン）を必ず表示 */
    button[kind="header"], button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stBaseButton-headerNoPadding"],
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        z-index: 999999 !important;
    }

    /* 見出し縮小 */
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.05rem !important; margin-top: 0.8rem !important; }
    h4 { font-size: 0.95rem !important; }

    /* アプリヘッダー帯のパディング縮小 */
    .app-header {
        padding: 0.55rem 0.9rem !important;
    }

    /* 表は横スクロール、最低文字サイズを確保 */
    .main table {
        font-size: 11px !important;
        min-width: 100%;
    }
    .main table th, .main table td {
        padding: 4px 5px !important;
        white-space: nowrap;
    }
    /* 表のラッパーをスクロール可能に */
    div[data-testid="stMarkdownContainer"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    /* selectbox / multiselect の最小幅を抑制（スマホで折返し対応） */
    div[data-baseweb="select"] { min-width: 100% !important; }

    /* multiselect で選んだチップが重ならないように */
    div[data-baseweb="select"] > div { flex-wrap: wrap !important; }

    /* st.columns のスタック化（横並びを縦並びに切替） */
    /* "horizontal" な radio はそのまま、columns は flex-wrap で折返し */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.4rem !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }

    /* iPhone での縦仕切り線を非表示（横並びが縦並びに変わるため不要） */
    .enemil-vert-divider {
        display: none !important;
    }

    /* テキスト入力 / 数値入力の上下パディングを切り詰める */
    div[data-baseweb="input"] input { font-size: 14px !important; }

    /* divider のマージン圧縮 */
    hr { margin: 0.7rem 0 !important; }

    /* ロゴ画像のサイズを抑える */
    img[alt="EneMil"] {
        max-width: 180px !important;
    }
}

/* タップしやすいようにボタンサイズを確保 */
@media (pointer: coarse) {
    .stButton > button, div[data-baseweb="select"] {
        min-height: 38px !important;
    }
    /* タッチデバイスでホバー hover-only スタイルを無効化 */
    button:hover { transform: none !important; }
}

/* テーブル横スクロール基本対応（PCでも） */
div[data-testid="stDataFrame"] { overflow-x: auto; }

/* ─── アプリヘッダー (Savant 風横長バー) ─── */
.app-header {
    background: linear-gradient(135deg, #2C5F7C 0%, #3A6B85 50%, #4A7A93 100%);
    padding: 0.9rem 1.4rem;
    border-radius: 6px;
    margin-bottom: 1.2rem;
    color: white;
    box-shadow: 0 1px 3px rgba(44, 95, 124, 0.15);
    border-left: 4px solid #6FA8C9;
}
/* アプリヘッダー内のテキストはすべて白で固定（h2/h3/spanなど全部） */
.app-header, .app-header *,
.app-header h1, .app-header h2, .app-header h3,
.app-header h1 *, .app-header h2 *, .app-header h3 *,
.app-header p, .app-header span, .app-header div {
    color: white !important;
    border-color: white !important;
}
.app-header h1, .app-header h2, .app-header h3 {
    border: none !important;
    margin: 0 !important;
}
/* ページヘッダー帯（render_page_header の動的HTML）も同様に白固定。
   インライン linear-gradient を含む div 内のすべてのテキスト要素。 */
div[style*="linear-gradient(135deg"],
div[style*="linear-gradient(135deg"] *,
div[style*="linear-gradient(135deg"] h1,
div[style*="linear-gradient(135deg"] h2,
div[style*="linear-gradient(135deg"] h3,
div[style*="linear-gradient(135deg"] p,
div[style*="linear-gradient(135deg"] span {
    color: white !important;
}

/* ─── サイドバーのナビメニュー（選択中の行）の文字をオレンジに ─── */
/* Streamlit のラジオで選択中の項目はテキストをアクセントオレンジに */
section[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"]
    input[type="radio"]:checked + div,
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) *,
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) span,
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) div {
    color: #E55A4C !important;   /* アクセントオレンジ */
    font-weight: 700 !important;
}

/* ─── メトリックカード ─── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #D6DEE5;
    border-radius: 6px;
    padding: 0.75rem;
    border-left: 4px solid #2C5F7C;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* ─── ボタン ─── */
.stButton > button {
    background-color: #2C5F7C !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px;
    font-weight: 600 !important;
    padding: 0.35rem 1.2rem;
    transition: background-color 0.15s;
}
.stButton > button *,
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
}
.stButton > button:hover,
.stButton > button:hover * {
    background-color: #3A6B85 !important;
    color: #FFFFFF !important;
}
/* primary タイプ (Streamlit kind="primary") も同様に白文字 */
button[kind="primary"], button[kind="primary"] *,
button[data-baseweb="button"], button[data-baseweb="button"] * {
    color: #FFFFFF !important;
}

/* ─── タブ (st.tabs) ─── */
div[data-baseweb="tab-list"] {
    border-bottom: 2px solid #D6DEE5;
    gap: 2px;
}
button[data-baseweb="tab"] {
    color: #4A7A93;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 4px 4px 0 0 !important;
}
/* 選択中のタブはブルーグレー背景＋アクセントオレンジの文字 */
button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] *,
button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span,
button[data-baseweb="tab"][aria-selected="true"] div {
    background-color: #2C5F7C !important;
    color: #E55A4C !important;
    font-weight: 700 !important;
}

/* ─── 区切り線 ─── */
hr {
    border: none;
    border-top: 1px solid #D6DEE5;
    margin: 1.2rem 0;
}

/* ─── HTMLテーブル（着色テーブル） ─── */
.main table {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Helvetica Neue", "Hiragino Sans", sans-serif;
}
.main table th {
    background-color: #E7ECF0 !important;
    color: #2C5F7C !important;
    border-bottom: 2px solid #2C5F7C !important;
    font-weight: 600 !important;
}

/* ─── Streamlit デフォルト DataFrame の薄ブルーアクセント ─── */
div[data-testid="stDataFrame"] thead tr th {
    background-color: #E7ECF0 !important;
    color: #2C5F7C !important;
    font-weight: 600 !important;
}

/* ─── キャプション ─── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #6B7785 !important;
    font-size: 12px !important;
}

/* ─── 警告 / 情報メッセージ ─── */
.stAlert {
    border-radius: 6px;
    border-left-width: 4px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# DB 初期化
# ──────────────────────────────────────────
init_db()

# ──────────────────────────────────────────
# サイトパスワード（第1層）
# ──────────────────────────────────────────
SITE_PASSWORD = os.environ.get("SITE_PASSWORD", "")

def check_site_password():
    if not SITE_PASSWORD:
        return True  # 環境変数未設定なら第1層スキップ
    if st.session_state.get("site_auth"):
        return True
    from app.components.header import render_login_logo
    render_login_logo()
    st.subheader("サイトアクセス認証")
    pw_input = st.text_input("サイトパスワード", type="password", key="site_pw_input")
    if st.button("確認"):
        import hmac
        if hmac.compare_digest(pw_input, SITE_PASSWORD):
            st.session_state["site_auth"] = True
            st.rerun()
        else:
            st.error("サイトパスワードが正しくありません")
    return False

if not check_site_password():
    st.stop()

# ──────────────────────────────────────────
# ログイン画面（第2層）
# ──────────────────────────────────────────
def render_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        from app.components.header import render_login_logo
        render_login_logo()
        st.write("")
        user_id  = st.text_input("ログインID", placeholder="ID を入力")
        password = st.text_input("パスワード", type="password", placeholder="パスワードを入力")
        if st.button("ログイン", use_container_width=True, type="primary"):
            if not user_id or not password:
                st.error("IDとパスワードを入力してください")
            else:
                ok, msg = login(user_id, password)
                if ok:
                    st.rerun()
                else:
                    st.error(msg)
        st.caption("※ アカウントはチーム管理者にお問い合わせください")


# ──────────────────────────────────────────
# 仮パスワード変更画面
# ──────────────────────────────────────────
def render_change_pw():
    st.warning("⚠️ 初期パスワードのままです。パスワードを変更してください。")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        new_pw  = st.text_input("新しいパスワード", type="password")
        new_pw2 = st.text_input("新しいパスワード（確認）", type="password")
        if st.button("変更する", type="primary", use_container_width=True):
            if new_pw != new_pw2:
                st.error("パスワードが一致しません")
            elif not validate_password(new_pw):
                st.error("パスワードは8文字以上で英大文字・英小文字・数字を含む必要があります")
            else:
                from app.auth import change_password, current_user
                if change_password(current_user(), new_pw):
                    st.success("パスワードを変更しました")
                    st.rerun()


# ──────────────────────────────────────────
# ルーティング
# ──────────────────────────────────────────
if not is_logged_in():
    render_login()
    st.stop()

# ※ 初回ログイン時のパスワード強制変更は撤廃しました
#   （旧仕様では is_temp_pw() == True のときにパスワード変更画面に飛んでいました）
#   パスワード変更が必要な場合は今後別途プロフィール画面等で行う想定。

# ──────────────────────────────────────────
# サイドバー・ナビゲーション
# ──────────────────────────────────────────
from app.auth import current_user

with st.sidebar:
    from app.components.header import render_sidebar_logo
    render_sidebar_logo()
    st.caption(f"ログイン中: {st.session_state.get('auth_name', '')} ({current_user()})")
    st.divider()

    # ── 対象年セレクタ（全ページ共通） ──
    # CSV から年候補を抽出（fingerprint をキーにモジュールレベルでキャッシュ）
    from app.stats.calculations import (
        load_csvs as _load_csvs,
        _normalize_date_series,
        get_csv_fingerprint as _gfp,
    )
    import os as _os
    DATA_DIR_FOR_YEARS = _os.environ.get("DATA_DIR", "data")

    # モジュールレベル年候補キャッシュ（fingerprint がキー）
    global _YEAR_OPTS_CACHE
    try:
        _YEAR_OPTS_CACHE
    except NameError:
        _YEAR_OPTS_CACHE = {"fp": None, "opts": None}

    _fp_now = _gfp(DATA_DIR_FOR_YEARS)
    if _YEAR_OPTS_CACHE["fp"] == _fp_now and _YEAR_OPTS_CACHE["opts"] is not None:
        year_opts = _YEAR_OPTS_CACHE["opts"]
    else:
        _df_for_years = _load_csvs(DATA_DIR_FOR_YEARS)
        year_opts = ["すべて"]
        if not _df_for_years.empty and "_year" in _df_for_years.columns:
            _yrs = _df_for_years["_year"].dropna().astype(int).unique()
            for _y in sorted(_yrs, reverse=True):
                year_opts.append(f"{_y}年")
        elif not _df_for_years.empty and "日付" in _df_for_years.columns:
            _nd = _normalize_date_series(_df_for_years["日付"])
            _yrs = (pd.to_datetime(_nd, errors="coerce")
                    .dt.year.dropna().astype(int).unique())
            for _y in sorted(_yrs, reverse=True):
                year_opts.append(f"{_y}年")
        _YEAR_OPTS_CACHE["fp"] = _fp_now
        _YEAR_OPTS_CACHE["opts"] = year_opts

    # デフォルトを「2026年」に（候補にあれば、無ければ最新年）
    default_year_idx = 0
    if "2026年" in year_opts:
        default_year_idx = year_opts.index("2026年")
    elif len(year_opts) > 1:
        default_year_idx = 1  # 「すべて」の次（最新年）

    target_year = st.selectbox(
        "対象年", year_opts, index=default_year_idx, key="target_year",
        help="表示するデータの年。",
    )
    st.divider()

    pages = {
        "⚾️ 投手分析":      "pitcher",
        "🦍 打者分析":      "batter",
        "📊 指標順位検索": "leaderboard",
        "📋 投手試合データ":  "game_summary",
        "📈 重要指標推移":  "kpi_trend",
    }
    if current_role() == "admin":
        pages["⚙️ 管理者ダッシュボード"] = "admin"

    page_key = st.radio("ページ選択", list(pages.keys()), label_visibility="collapsed")
    selected = pages[page_key]

    st.divider()
    if st.button("ログアウト", use_container_width=True):
        logout()
        st.rerun()

# ──────────────────────────────────────────
# ページレンダリング
# ──────────────────────────────────────────
from app.auth import log_event
log_event("page_view", selected)

if selected == "pitcher":
    from app.pages.pitcher_analysis import render
    render()
elif selected == "batter":
    from app.pages.batter_analysis import render
    render()
elif selected == "leaderboard":
    from app.pages.leaderboard import render
    render()
elif selected == "game_summary":
    from app.pages.game_summary import render
    render()
elif selected == "kpi_trend":
    from app.pages.kpi_trend import render
    render()
elif selected == "admin":
    if current_role() == "admin":
        from app.pages.admin.dashboard import render
        render()
    else:
        st.error("アクセス権限がありません")
