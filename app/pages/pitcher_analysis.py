# -*- coding: utf-8 -*-
"""投手分析ページ"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from ..stats.calculations import (
    load_csvs, filter_by_player_period, filter_by_year, get_year_default_start,
    calculate_pitcher_stats_combined, calculate_game_metrics,
    calculate_pitchtype_metrics, calculate_pitchtype_metrics_by_side,
    _pick_col, _normalize_date_series,
)
from ..stats.constants import METRIC_DESCRIPTIONS
from ..components.charts import (
    draw_zone_heatmap_by_side, draw_velocity_histogram,
    draw_pitch_donut_split, render_colored_stats_table,
    draw_count_based_pitch_ratio_by_side,
)
from ..auth import log_event

DATA_DIR = os.environ.get("DATA_DIR", "data")

# 着色する数値カラム（high = 高いほど良い、low = 低いほど良い）
PITCHER_METRIC_DIRECTIONS = {
    # 基本成績
    "被打率": "low", "防御率": "low", "WHIP": "low",
    "K%": "high", "BB%": "low", "K-BB%": "high", "K/BB": "high",
    # 投手詳細指標
    "Strike%": "high", "1st-Strike%": "high",
    "Whiff%": "high", "3球追い込み%": "high",
    # 球種別
    "投球割合": "high", "球速比（％）": "high",
    "Zone%": "high", "Chase%": "high", "SwStr%": "high",
    "PutAway%": "high", "GB%": "high", "FB%": "low",
}

# 着色対象（基本成績は count 系除く）
PITCHER_BASIC_COLOR_COLS = [
    "被打率", "防御率", "WHIP", "K%", "BB%", "K-BB%", "K/BB",
]
PITCHER_GAME_COLOR_COLS = [
    "Strike%", "1st-Strike%", "Whiff%", "3球追い込み%",
]
# 球種別の着色対象（参考ノートブック準拠）
PITCHTYPE_COLOR_COLS = [
    "Zone%", "Chase%", "Whiff%", "SwStr%", "PutAway%", "GB%", "FB%", "被打率",
]


@st.cache_data(show_spinner=False)
def _list_teams_and_pitchers(target_year: str, _n: int,
                             _df_year: pd.DataFrame,
                             team_col: str | None,
                             pitcher_col: str) -> dict:
    """対象年フィルタ済みデータから、チーム一覧と各チームの投手一覧を一度だけ抽出してキャッシュ。
    キーは (target_year, _n) なので同一データなら 1 回しか走らない。"""
    if team_col and team_col in _df_year.columns:
        teams = sorted(_df_year[team_col].dropna().unique().tolist())
        by_team = {
            t: sorted(_df_year[_df_year[team_col] == t][pitcher_col]
                      .dropna().unique().tolist())
            for t in teams
        }
    else:
        teams = []
        by_team = {None: sorted(_df_year[pitcher_col].dropna().unique().tolist())}
    return {"teams": teams, "by_team": by_team}


@st.cache_data(show_spinner=False)
def _player_date_range(_n: int, _df_all: pd.DataFrame, player_col: str,
                        player: str, target_year: str) -> tuple:
    """特定選手の日付範囲（start, end）をキャッシュ。
    キャッシュキー: (_n, player_col, player, target_year)
    _df_all はハッシュskip。_norm_date 列があれば高速。"""
    sub = _df_all[_df_all[player_col] == player]
    if sub.empty:
        return (None, None)
    if "_norm_date" in sub.columns:
        dts = pd.to_datetime(sub["_norm_date"], errors="coerce").dropna()
    else:
        return (None, None)
    if not len(dts):
        return (None, None)
    if target_year != "すべて":
        try:
            yr = int(str(target_year).replace("年", "").strip())
            in_yr = dts[dts.dt.year == yr]
            if len(in_yr):
                dts = in_yr
        except Exception:
            pass
    return (dts.min().date(), dts.max().date())


@st.cache_data(show_spinner=False)
def _build_pitcher_distributions(_n: int, _df_all: pd.DataFrame,
                                 min_pitches: int = 20) -> dict:
    """全投手の基本成績・投手詳細指標から各指標の分布を作る。
    _df_all は先頭アンダースコアで Streamlit のハッシュ計算を skip。
    キャッシュキーは _n（行数）。"""
    df_all = _df_all
    pitcher_col = _pick_col(df_all, ["投手名"])
    if not pitcher_col:
        return {}
    dist = {c: [] for c in PITCHER_BASIC_COLOR_COLS + PITCHER_GAME_COLOR_COLS}
    # ★ groupby を使って何度もブールマスクを作るのを避ける
    for name, sub in df_all.groupby(pitcher_col, sort=False):
        if pd.isna(name):
            continue
        if "球種" in sub.columns:
            n_p = int(sub["球種"].notna().sum())
        else:
            n_p = len(sub)
        if n_p < min_pitches:
            continue
        s = calculate_pitcher_stats_combined(sub)
        m = calculate_game_metrics(sub)
        merged = {**s, **m}
        for c in dist:
            v = merged.get(c)
            try:
                vf = float(v)
                if np.isfinite(vf):
                    dist[c].append(vf)
            except Exception:
                pass
    return {c: np.array(arr, dtype=float) for c, arr in dist.items()}


@st.cache_data(show_spinner=False)
def _build_pitchtype_distributions(_n: int, _df_all: pd.DataFrame,
                                   min_pitches: int = 25) -> dict:
    """全投手の球種別の成績から、各「球種×指標」の分布を作る。"""
    df_all = _df_all
    pitcher_col = _pick_col(df_all, ["投手名"])
    if not pitcher_col:
        return {}
    dist = {}
    for name, sub in df_all.groupby(pitcher_col, sort=False):
        if pd.isna(name):
            continue
        pt_df = calculate_pitchtype_metrics(sub)
        if pt_df.empty:
            continue
        for _, row in pt_df.iterrows():
            pt = str(row.get("球種"))
            if pd.to_numeric(row.get("投球数", 0), errors="coerce") < min_pitches:
                continue
            for c in PITCHTYPE_COLOR_COLS:
                if c not in row:
                    continue
                try:
                    vf = float(row[c])
                    if np.isfinite(vf):
                        dist.setdefault((pt, c), []).append(vf)
                        dist.setdefault(("ALL", c), []).append(vf)
                except Exception:
                    pass
    return {k: np.array(arr, dtype=float) for k, arr in dist.items()}


def render():
    from ..components.header import render_page_header
    render_page_header("投手分析", icon="⚾️")

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.warning("data/ フォルダにCSVファイルがありません。"
                   "管理者ダッシュボードからアップロードしてください。")
        return

    df_all = load_csvs(DATA_DIR)
    if df_all.empty:
        st.error("CSVの読み込みに失敗しました")
        return

    pitcher_col = _pick_col(df_all, ["投手名"])
    team_col = _pick_col(df_all, ["投手チーム", "PitcherTeam"])
    if not pitcher_col:
        st.error("投手名列が見つかりません")
        return

    date_col = _pick_col(df_all, ["日付", "試合日", "ゲーム日", "実施日"])

    # 対象年フィルタ（main.py のサイドバーで設定）
    target_year = st.session_state.get("target_year", "すべて")
    df_year = filter_by_year(df_all, target_year)
    if df_year.empty:
        st.warning(f"{target_year} のデータがありません")
        return

    # ──── フィルターパネル（チーム → 投手 の二段階） ────
    with st.sidebar:
        st.subheader("フィルター")

        # ★ チーム一覧と各チームの投手一覧をキャッシュから取得
        tp = _list_teams_and_pitchers(target_year, len(df_year), df_year,
                                       team_col, pitcher_col)

        if team_col:
            teams = tp["teams"]
            team = st.selectbox("① チーム選択", teams, key="p_team")
            pitchers = tp["by_team"].get(team, [])
        else:
            team = None
            pitchers = tp["by_team"].get(None, [])

        if not pitchers:
            st.warning("選択チームに投手がいません")
            return
        pitcher = st.selectbox("② 投手選択", pitchers, key="p_player")

        # ──── 開始日/終了日のデフォルト ────
        # 選択された選手の日付範囲をキャッシュから取得（_norm_date を活用）
        default_start, default_end = _player_date_range(
            len(df_all), df_all, pitcher_col, pitcher, target_year)

        if date_col:
            # 開始日と終了日のセッション key は、選手＋年で個別化して、
            # 選手を変えたらデフォルトに戻るようにする。
            # ★ min_value/max_value を選手のデータ範囲に設定し、
            #    終了日は常にその選手の最新データ日が初期値になるようにする。
            if default_start and default_end:
                start_d = st.date_input(
                    "開始日", value=default_start,
                    min_value=default_start, max_value=default_end,
                    key=f"p_start_{pitcher}_{target_year}",
                )
                end_d = st.date_input(
                    "終了日", value=default_end,
                    min_value=default_start, max_value=default_end,
                    key=f"p_end_{pitcher}_{target_year}",
                )
            else:
                start_d = st.date_input(
                    "開始日", value=default_start,
                    key=f"p_start_{pitcher}_{target_year}",
                )
                end_d = st.date_input(
                    "終了日", value=default_end,
                    key=f"p_end_{pitcher}_{target_year}",
                )
            start_str = str(start_d) if start_d else None
            end_str = str(end_d) if end_d else None
        else:
            start_str = end_str = None

        view = st.radio("ヒートマップ視点", ["捕手目線", "投手目線"],
                        horizontal=True)
        view_key = "catcher" if view == "捕手目線" else "pitcher"

    df = filter_by_player_period(df_all, pitcher_col, pitcher,
                                 start_str, end_str)
    if df.empty:
        st.warning("該当データがありません")
        return

    log_event("pitcher_stats_view", pitcher)

    # ──── ヘッダー ────
    header = f"⚾ {pitcher}"
    if team:
        header += f"（{team}）"
    st.subheader(header)

    # ──── 分布データ事前計算（リーグ全体） ────
    dist_pitcher = _build_pitcher_distributions(len(df_all), df_all)
    dist_pitchtype = _build_pitchtype_distributions(len(df_all), df_all)

    # ──── 基本成績 ────
    st.markdown("#### 基本成績")
    stats = calculate_pitcher_stats_combined(df)
    render_colored_stats_table(
        pd.DataFrame([stats]),
        metric_directions=PITCHER_METRIC_DIRECTIONS,
        key="pitcher_basic_tbl",
        distributions={c: dist_pitcher[c] for c in PITCHER_BASIC_COLOR_COLS
                       if c in dist_pitcher},
        int_cols=["球数", "打席", "打数", "自責点", "被安打",
                  "被本塁打", "奪三振", "四球", "死球"],
    )

    # ──── 投手詳細指標 ────
    st.markdown("#### 投手詳細指標")
    metrics = calculate_game_metrics(df)
    render_colored_stats_table(
        pd.DataFrame([metrics]),
        metric_directions=PITCHER_METRIC_DIRECTIONS,
        key="pitcher_game_tbl",
        distributions={c: dist_pitcher[c] for c in PITCHER_GAME_COLOR_COLS
                       if c in dist_pitcher},
    )
    st.caption(
        "🟦青＝下位 / ⚪白＝中位 / 🟥赤＝上位（リーグ内パーセンタイル比較）。"
        "「BB%・WHIP・ERA・被打率」など低い方が良い指標は色を反転しています。"
    )

    with st.expander("📖 指標の説明"):
        seen = set()
        for col in list(stats.keys()) + list(metrics.keys()):
            if col in seen or col not in METRIC_DESCRIPTIONS:
                continue
            seen.add(col)
            info = METRIC_DESCRIPTIONS[col]
            st.markdown(f"**{col}**: {info['desc']}  \n"
                        f"`計算式: {info['formula']}`")

    st.divider()

    # ──── 球種割合（対右・対左を左右に並べて表示） ────
    st.markdown("### 球種割合")
    draw_pitch_donut_split(df, key_prefix="pitcher_donut")

    st.divider()

    # ──── カウント別投球割合（対右 / 対左）────
    draw_count_based_pitch_ratio_by_side(df, key_prefix="pitcher_cnt")

    st.divider()

    # ──── 球種別の成績（対右・対左で分けて表示） ────
    st.markdown("### 球種別の成績")

    pt_by_side = calculate_pitchtype_metrics_by_side(df)

    # 球種別の分布（行ごとに球種が異なるので per-row 分布を作って描画）
    def _make_pt_distributions(pt_df):
        """球種別 DataFrame の各行に対し「その球種の分布」を割り当てる。"""
        if pt_df is None or pt_df.empty:
            return {}
        # 全列共通の「球種ごとの分布」を作るのは難しいので、render_colored_stats_table は
        # 列単位の分布を扱う想定。ここでは「ALL 全球種混合」の分布を渡す方式にする。
        out = {}
        for c in PITCHTYPE_COLOR_COLS:
            key = ("ALL", c)
            if key in dist_pitchtype:
                out[c] = dist_pitchtype[key]
        return out

    col_R, col_L = st.columns(2)
    with col_R:
        st.markdown("**対右打者**")
        df_R = pt_by_side.get("R", pd.DataFrame())
        if df_R.empty:
            st.caption("対右打者のデータがありません")
        else:
            render_colored_stats_table(
                df_R, metric_directions=PITCHER_METRIC_DIRECTIONS,
                key="pt_R_tbl",
                min_count=5, count_col="投球数",
                distributions=_make_pt_distributions(df_R),
                int_cols=["投球数"],
                no_color_cols=["投球数", "投球割合", "球速比（％）"],
            )
    with col_L:
        st.markdown("**対左打者**")
        df_L = pt_by_side.get("L", pd.DataFrame())
        if df_L.empty:
            st.caption("対左打者のデータがありません")
        else:
            render_colored_stats_table(
                df_L, metric_directions=PITCHER_METRIC_DIRECTIONS,
                key="pt_L_tbl",
                min_count=5, count_col="投球数",
                distributions=_make_pt_distributions(df_L),
                int_cols=["投球数"],
                no_color_cols=["投球数", "投球割合", "球速比（％）"],
            )

    # 全体（左右合計）— 常に表示（折りたたみ無し）
    st.markdown("#### 球種別の成績（左右合計）")
    pt_all = calculate_pitchtype_metrics(df)
    if not pt_all.empty:
        render_colored_stats_table(
            pt_all, metric_directions=PITCHER_METRIC_DIRECTIONS,
            key="pt_all_tbl",
            distributions=_make_pt_distributions(pt_all),
            int_cols=["投球数"],
            no_color_cols=["投球数", "投球割合", "球速比（％）"],
        )

    st.divider()

    # ──── ヒートマップ（球種別 KDE、対右・対左で分割） ────
    draw_zone_heatmap_by_side(df, title="投球コース",
                              view=view_key, top_n=4,
                              key_prefix="pitcher_hm")

    st.divider()

    # ──── 球速分布 ────
    draw_velocity_histogram(df, pitcher_name=pitcher,
                            key="pitcher_velo_hist")
