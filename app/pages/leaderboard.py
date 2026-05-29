# -*- coding: utf-8 -*-
"""指標順位検索ページ"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from ..stats.calculations import (
    load_csvs, filter_by_year,
    calculate_pitcher_stats_combined,
    calculate_game_metrics, calculate_pitchtype_metrics,
    calculate_pitcher_overall_pitch_metrics,
    calculate_baseball_stats, calculate_sabermetrics,
    _pick_col, _extract_final_pitches_per_pa,
)
from ..components.charts import render_colored_stats_table
from ..auth import log_event

DATA_DIR = os.environ.get("DATA_DIR", "data")

# 表示候補列（投手・全球種モード）。「投球回」は文字列のためソート可能だが特別処理する。
PITCHER_COLS = [
    "投球回", "球数", "打席", "平均球速", "防御率", "被打率", "WHIP",
    "K%", "BB%", "K-BB%", "K/BB",
    "Strike%", "1st-Strike%", "3球追い込み%", "Whiff%",
    "Zone%", "Chase%", "SwStr%", "PutAway%", "GB%", "FB%",
]
# 表示候補列（投手・特定球種モード = calculate_pitchtype_metrics の列）
PITCHTYPE_COLS = [
    "投球数", "投球割合", "球速比（％）",
    "Zone%", "Chase%", "Whiff%", "SwStr%", "PutAway%",
    "GB%", "FB%", "被打率",
]
BATTER_COLS = [
    "打席", "打数", "打率", "出塁率", "長打率", "OPS",
    "ISO", "wOBA",
    "K%", "BB%",
    "Swing%", "Whiff%", "O-Swing%", "Z-Swing%",
    "Contact%", "O-Contact%", "Z-Contact%",
    "Pull-air%",
]

PITCHER_DIRECTIONS = {
    "防御率": "low", "被打率": "low", "WHIP": "low", "BB%": "low",
    "K%": "high", "K-BB%": "high", "K/BB": "high",
    "Strike%": "high", "1st-Strike%": "high",
    "3球追い込み%": "high", "Whiff%": "high",
    # 球種別 / 全球種共通
    "Zone%": "high", "Chase%": "high", "SwStr%": "high",
    "PutAway%": "high", "GB%": "high", "FB%": "low",
    "投球割合": "high", "球速比（％）": "high", "投球数": "high",
    "平均球速": "high",
}
BATTER_DIRECTIONS = {
    "打率": "high", "出塁率": "high", "長打率": "high", "OPS": "high",
    "ISO": "high", "wOBA": "high", "BB%": "high",
    "K%": "low", "Whiff%": "low", "O-Swing%": "low",
    "Swing%": "high", "Z-Swing%": "high",
    "Contact%": "high", "O-Contact%": "high", "Z-Contact%": "high",
    "Pull-air%": "high",
}


def _innings_to_outs(ip_str) -> float:
    """'5 2/3' のような文字列を数値（アウト数）に変換。ソート用。"""
    if pd.isna(ip_str):
        return 0.0
    s = str(ip_str).strip()
    if not s or s == "0":
        return 0.0
    try:
        if " " in s:
            whole, frac = s.split(" ", 1)
            num, den = frac.split("/")
            return float(whole) * 3 + float(num) / float(den) * 3
        if "/" in s:
            num, den = s.split("/")
            return float(num) / float(den) * 3
        return float(s) * 3
    except Exception:
        return 0.0


# ──────────────────────────────────────────
# キャッシュ：集計結果（投手・打者・球種別）
# ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_pitcher_board_df(target_year: str, n: int,
                            _df: pd.DataFrame) -> pd.DataFrame:
    """投手・全球種モードの集計表（重い処理）をキャッシュ。
    target_year, n をキャッシュキーに使う（_df はハッシュskip）。
    ★ n は通常引数（非アンダースコア）。アンダースコア接頭辞だと
       Streamlit がキーから除外し、CSV更新時に古いキャッシュが返るバグになる。"""
    df_all = _df
    pitcher_col = _pick_col(df_all, ["投手名"])
    team_col = _pick_col(df_all, ["投手チーム", "PitcherTeam"])
    if not pitcher_col:
        return pd.DataFrame()

    rows = []
    for name, sub in df_all.groupby(pitcher_col, sort=False):
        if pd.isna(name):
            continue
        row = {"投手名": name}
        if team_col:
            row["チーム"] = (sub[team_col].dropna().iloc[0]
                            if sub[team_col].notna().any() else "")
        s = calculate_pitcher_stats_combined(sub)
        m = calculate_game_metrics(sub)
        # 全球種合計レベルの Zone% / Chase% / SwStr% / PutAway% / GB% / FB%
        m2 = calculate_pitcher_overall_pitch_metrics(sub)
        row.update(s)
        row.update(m)
        row.update(m2)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df_lb = pd.DataFrame(rows)
    str_cols = ("投手名", "チーム", "投球回")
    for c in df_lb.columns:
        if c not in str_cols:
            df_lb[c] = pd.to_numeric(df_lb[c], errors="coerce")
    if "投球回" in df_lb.columns:
        df_lb["_アウト数"] = df_lb["投球回"].apply(_innings_to_outs)
    return df_lb


@st.cache_data(show_spinner=False)
def _build_pitchtype_board_df(target_year: str, n: int, _df: pd.DataFrame,
                              target_pitch: str) -> pd.DataFrame:
    """投手・特定球種モード（calculate_pitchtype_metrics ベース）の集計表。"""
    df_all = _df
    pitcher_col = _pick_col(df_all, ["投手名"])
    team_col = _pick_col(df_all, ["投手チーム", "PitcherTeam"])
    if not pitcher_col:
        return pd.DataFrame()

    rows = []
    for name, sub in df_all.groupby(pitcher_col, sort=False):
        if pd.isna(name):
            continue
        pt_df = calculate_pitchtype_metrics(sub)
        if pt_df.empty:
            continue
        hit = pt_df[pt_df["球種"] == target_pitch]
        if hit.empty:
            continue
        row = {"投手名": name}
        if team_col:
            row["チーム"] = (sub[team_col].dropna().iloc[0]
                            if sub[team_col].notna().any() else "")
        row.update(hit.iloc[0].to_dict())
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df_lb = pd.DataFrame(rows)
    str_cols = ("投手名", "チーム", "球種", "平均球速")
    for c in df_lb.columns:
        if c not in str_cols:
            df_lb[c] = pd.to_numeric(df_lb[c], errors="coerce")
    return df_lb


@st.cache_data(show_spinner=False)
def _build_batter_board_df(target_year: str, n: int,
                           _df: pd.DataFrame) -> pd.DataFrame:
    """打者集計表（重い処理）をキャッシュ。
    ★ target_year と n をキーに含める。n を非アンダースコアにすることで、
       CSV更新時に行数が変わればキャッシュが自動 invalidate される。"""
    df_all = _df
    batter_col = _pick_col(df_all, ["打者名", "BatterName", "打者"])
    team_col = _pick_col(df_all, ["打者チーム", "BatterTeam"])
    if not batter_col:
        return pd.DataFrame()

    rows = []
    for name, sub in df_all.groupby(batter_col, sort=False):
        if pd.isna(name):
            continue
        final = _extract_final_pitches_per_pa(sub)
        s = calculate_baseball_stats(final)
        m = calculate_sabermetrics(sub)
        row = {"打者名": name}
        if team_col:
            row["チーム"] = (sub[team_col].dropna().iloc[0]
                            if sub[team_col].notna().any() else "")
        row.update({k: v for k, v in s.items() if k != "状況"})
        row.update(m)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df_lb = pd.DataFrame(rows)
    for c in df_lb.columns:
        if c not in ("打者名", "チーム"):
            df_lb[c] = pd.to_numeric(df_lb[c], errors="coerce")
    return df_lb


# ──────────────────────────────────────────
# UI ヘルパ
# ──────────────────────────────────────────
def _default_asc_for(col_name: str, directions: dict) -> bool:
    """指標の方向に応じてデフォルトの昇順/降順を返す。
    'low' (低いほど良い) → 昇順 True。
    'high' (高いほど良い) → 降順 False。"""
    return directions.get(col_name, "high") == "low"


# ──────────────────────────────────────────
# render
# ──────────────────────────────────────────
def render():
    from ..components.header import render_page_header
    render_page_header("指標順位検索", icon="📊")

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.warning("data/ フォルダにCSVファイルがありません")
        return

    df_all = load_csvs(DATA_DIR)
    if df_all.empty:
        st.error("CSVの読み込みに失敗しました")
        return

    # 対象年フィルタ
    target_year = st.session_state.get("target_year", "すべて")
    df_year = filter_by_year(df_all, target_year)
    if df_year.empty:
        st.warning(f"{target_year} のデータがありません")
        return

    with st.sidebar:
        st.subheader("指標順位検索設定")
        tab_type = st.radio("種別", ["投手", "打者"])
        if tab_type == "投手":
            min_pa = st.number_input("最低球数", min_value=0,
                                     value=100, step=10, key="lb_min_p")
            pitch_opts = ["全球種"]
            if "球種" in df_year.columns:
                pitch_opts += sorted(
                    df_year["球種"].dropna().astype(str).unique().tolist())
            target_pitch = st.selectbox("対象球種", pitch_opts,
                                        index=0, key="lb_p_pitch")
        else:
            min_pa = st.number_input("最低打席数", min_value=0,
                                     value=50, step=5, key="lb_min_b")
            target_pitch = "全球種"

    log_event("leaderboard_view", tab_type)

    if tab_type == "投手":
        _render_pitcher_board(df_year, int(min_pa), target_pitch, target_year)
    else:
        _render_batter_board(df_year, int(min_pa), target_year)


def _render_pitcher_board(df_year: pd.DataFrame, min_pa: int,
                          target_pitch: str = "全球種",
                          target_year: str = "すべて"):
    is_pitch_filtered = (target_pitch != "全球種") and ("球種" in df_year.columns)

    # ★ キャッシュされた集計表を取得（重い処理はここでスキップされる）
    #    target_year をキーに含めて、年ごとに別キャッシュにする
    if is_pitch_filtered:
        df_lb = _build_pitchtype_board_df(target_year, len(df_year),
                                          df_year, target_pitch)
    else:
        df_lb = _build_pitcher_board_df(target_year, len(df_year), df_year)

    if df_lb.empty:
        if is_pitch_filtered:
            st.info(f"「{target_pitch}」を投げている投手がいません。")
        else:
            st.info("表示できる投手データがありません")
        return

    # 最低球数フィルタ（集計後の軽い処理）
    count_col_for_min = "球数" if "球数" in df_lb.columns else (
        "投球数" if "投球数" in df_lb.columns else None)
    if count_col_for_min:
        df_lb = df_lb[df_lb[count_col_for_min].fillna(0) >= min_pa]

    if df_lb.empty:
        st.info(f"最低球数 {min_pa} を満たす投手がいません。"
                "サイドバーでしきい値を下げてください。")
        return

    # 列選択
    if is_pitch_filtered:
        avail_cols = [c for c in PITCHTYPE_COLS if c in df_lb.columns]
        # 球種別モードの既定はこれまで通り先頭から
        default_cnt = min(8, len(avail_cols))
        default_cols = avail_cols[:default_cnt]
    else:
        avail_cols = [c for c in PITCHER_COLS if c in df_lb.columns]
        # 全球種モードの既定：要望指標を優先で並べ、不足分は avail_cols 先頭から補う
        requested_defaults = ["Whiff%", "SwStr%", "Chase%", "Zone%"]
        # ★ デフォルト初期表示から除外する指標（ユーザーがmultiselectで後から追加可能）
        excluded_from_default = {"投球回", "打席", "防御率"}
        default_cols = [c for c in requested_defaults if c in avail_cols]
        # 足りない場合は avail_cols の頭から重複しないように補う（最大8列、除外指標はスキップ）
        for c in avail_cols:
            if len(default_cols) >= 8:
                break
            if c in default_cols or c in excluded_from_default:
                continue
            default_cols.append(c)
    sel_cols = st.multiselect(
        "表示する指標", avail_cols,
        default=default_cols,
        key="lb_p_cols",
    )
    if not sel_cols:
        st.info("表示する指標を1つ以上選択してください")
        return

    # ソート
    col1, col2 = st.columns([3, 1])
    with col1:
        # ★ 「順位を知りたい指標」が変わったら sort_asc キーをリセット
        prev_sort = st.session_state.get("_lb_p_prev_sort")
        # 初期表示時のデフォルトソート列：全球種モードでは Whiff% を優先
        if not is_pitch_filtered and "Whiff%" in sel_cols:
            default_sort_index = sel_cols.index("Whiff%")
        else:
            default_sort_index = 0
        sort_col = st.selectbox("順位を知りたい指標", sel_cols,
                                index=default_sort_index,
                                key="lb_p_sort")
        # ★ デフォルト：指標の方向に応じて自動で昇順/降順を切替
        if sort_col != prev_sort:
            st.session_state["lb_p_asc"] = _default_asc_for(
                sort_col, PITCHER_DIRECTIONS)
            st.session_state["_lb_p_prev_sort"] = sort_col
    with col2:
        sort_asc = st.checkbox("昇順", value=st.session_state.get(
            "lb_p_asc", _default_asc_for(sort_col, PITCHER_DIRECTIONS)),
            key="lb_p_asc")

    id_cols = ["投手名"] + (["チーム"] if "チーム" in df_lb.columns else [])
    disp = df_lb[id_cols + sel_cols].copy()

    if sort_col == "投球回" and "_アウト数" in df_lb.columns:
        disp["_アウト数"] = df_lb["_アウト数"].values
        disp = disp.sort_values("_アウト数", ascending=sort_asc,
                                na_position="last").drop(columns="_アウト数")
    else:
        disp = disp.sort_values(sort_col, ascending=sort_asc,
                                na_position="last")

    disp = disp.reset_index(drop=True)
    disp.insert(0, "#", range(1, len(disp) + 1))

    log_event("leaderboard_sort", f"P:{sort_col}:{sort_asc}:{target_pitch}")

    label_suffix = "" if not is_pitch_filtered else f" — 対象球種: {target_pitch}"
    st.markdown(f"**投手ランキング（n={len(disp)}）{label_suffix}**")
    render_colored_stats_table(
        disp, metric_directions=PITCHER_DIRECTIONS,
        key="lb_pitcher_tbl",
        int_cols=["#", "球数", "打席", "投球数"],
        reverse_color_cols=["#"],
        no_color_cols=["投球数", "投球割合", "球速比（％）"],
    )
    st.caption("青＝下位、赤＝上位（指標の方向に応じて自動反転）。"
               "「BB%」「防御率」「WHIP」「被打率」は低いほど良い指標として着色。"
               "「#（順位）」列は 1 位が赤、下位ほど青で表示。")


def _render_batter_board(df_year: pd.DataFrame, min_pa: int,
                         target_year: str = "すべて"):
    df_lb = _build_batter_board_df(target_year, len(df_year), df_year)

    if df_lb.empty:
        st.info("表示できる打者データがありません")
        return

    if "打席" in df_lb.columns:
        df_lb = df_lb[df_lb["打席"].fillna(0) >= min_pa]

    if df_lb.empty:
        st.info(f"最低打席数 {min_pa} を満たす打者がいません。"
                "サイドバーでしきい値を下げてください。")
        return

    avail_cols = [c for c in BATTER_COLS if c in df_lb.columns]
    # 既定：要望指標を優先で並べ、不足分は avail_cols 先頭から補う
    requested_defaults = ["Whiff%", "O-Swing%", "Z-Swing%", "Pull-air%", "wOBA"]
    # ★ デフォルト初期表示から除外する指標（ユーザーがmultiselectで後から追加可能）
    excluded_from_default = {"打席", "打数", "打率"}
    default_cols = [c for c in requested_defaults if c in avail_cols]
    for c in avail_cols:
        if len(default_cols) >= 8:
            break
        if c in default_cols or c in excluded_from_default:
            continue
        default_cols.append(c)
    sel_cols = st.multiselect(
        "表示する指標", avail_cols,
        default=default_cols,
        key="lb_b_cols",
    )
    if not sel_cols:
        st.info("表示する指標を1つ以上選択してください")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        prev_sort = st.session_state.get("_lb_b_prev_sort")
        # 初期表示時のデフォルトソート列：Whiff% を優先
        if "Whiff%" in sel_cols:
            default_sort_index = sel_cols.index("Whiff%")
        else:
            default_sort_index = 0
        sort_col = st.selectbox("順位を知りたい指標", sel_cols,
                                index=default_sort_index,
                                key="lb_b_sort")
        if sort_col != prev_sort:
            st.session_state["lb_b_asc"] = _default_asc_for(
                sort_col, BATTER_DIRECTIONS)
            st.session_state["_lb_b_prev_sort"] = sort_col
    with col2:
        sort_asc = st.checkbox("昇順", value=st.session_state.get(
            "lb_b_asc", _default_asc_for(sort_col, BATTER_DIRECTIONS)),
            key="lb_b_asc")

    id_cols = ["打者名"] + (["チーム"] if "チーム" in df_lb.columns else [])
    disp = df_lb[id_cols + sel_cols].copy()
    disp = disp.sort_values(sort_col, ascending=sort_asc,
                            na_position="last").reset_index(drop=True)
    disp.insert(0, "#", range(1, len(disp) + 1))

    log_event("leaderboard_sort", f"B:{sort_col}:{sort_asc}")

    st.markdown(f"**打者ランキング（n={len(disp)}）**")
    render_colored_stats_table(
        disp, metric_directions=BATTER_DIRECTIONS,
        key="lb_batter_tbl",
        int_cols=["#", "打席", "打数"],
        decimals_2_cols=["打率", "出塁率", "長打率", "OPS", "ISO", "wOBA"],
        reverse_color_cols=["#"],
    )
    st.caption("青＝下位、赤＝上位（指標の方向に応じて自動反転）。"
               "「K%」「Whiff%」「O-Swing%」は低いほど良い指標として着色。"
               "「#（順位）」列は 1 位が赤、下位ほど青で表示。")
