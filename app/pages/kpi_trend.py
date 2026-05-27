# -*- coding: utf-8 -*-
"""KPI推移ページ"""
import os, streamlit as st, pandas as pd, numpy as np
from ..stats.calculations import (load_csvs, filter_by_player_period,
    calculate_game_metrics, calculate_pitchtype_metrics,
    calculate_baseball_stats, calculate_sabermetrics,
    _pick_col, _normalize_date_series, _extract_final_pitches_per_pa)
from ..stats.constants import PITCHER_KPI, BATTER_KPI
from ..components.charts import draw_kpi_trend
from ..auth import log_event

DATA_DIR = os.environ.get("DATA_DIR", "data")


def _get_pitcher_kpi_by_game(df_all, pitcher_col, pitcher) -> list[dict]:
    """試合ごとの投手KPIを計算して返す"""
    date_col = _pick_col(df_all, ["日付", "試合日", "ゲーム日", "実施日"])
    if not date_col:
        return []
    df_p = df_all[df_all[pitcher_col].astype(str) == str(pitcher)].copy()
    df_p["_nd"] = _normalize_date_series(df_p[date_col])
    results = []
    for date, grp in df_p.groupby("_nd"):
        if pd.isna(date) or date == "" or date is None:
            continue
        m  = calculate_game_metrics(grp)
        pt = calculate_pitchtype_metrics(grp)
        row = {"date": date}
        row["Strike%"]       = m.get("Strike%", None)
        row["1st-Strike%"]   = m.get("1st-Strike%", None)
        row["3球追い込み%"]  = m.get("3球追い込み%", None)
        # K%, BB%
        from ..stats.calculations import calculate_pitcher_stats_combined
        base = calculate_pitcher_stats_combined(grp)
        row["K%"]  = base.get("K%",  None)
        row["BB%"] = base.get("BB%", None)

        # 球種別KPI（新しい calculate_pitchtype_metrics の列名に対応）
        # ・平均球速: 文字列 "136.9 (140.0)" → 先頭の数値を抜き出す
        # ・Whiff%: 数値そのまま
        # ・球種名: "チェンジ", "カット" 等の略称（古い"チェンジアップ"は新では"チェンジ"）
        pitch_aliases = {
            "ストレート": ["ストレート"],
            "スプリット": ["スプリット"],
            "チェンジ":   ["チェンジ", "チェンジアップ"],
            "スライダー": ["スライダー"],
        }
        def _row_for(pt_name):
            if pt.empty or "球種" not in pt.columns:
                return None
            for alias in pitch_aliases.get(pt_name, [pt_name]):
                hit = pt[pt["球種"] == alias]
                if not hit.empty:
                    return hit.iloc[0]
            return None

        # ストレート平均球速
        ff_row = _row_for("ストレート")
        if ff_row is not None:
            v = ff_row.get("平均球速")
            # "136.9 (140.0)" → 136.9 / 数値ならそのまま
            try:
                if isinstance(v, str):
                    head = v.strip().split()[0]
                    row["ストレート平均球速"] = float(head)
                else:
                    f = float(v)
                    row["ストレート平均球速"] = f if pd.notna(f) else None
            except Exception:
                row["ストレート平均球速"] = None
        else:
            row["ストレート平均球速"] = None

        # 各球種の Whiff%
        for pt_name, kpi_key in [
            ("ストレート", "ストレートWhiff%"),
            ("スプリット", "スプリットWhiff%"),
            ("チェンジ",   "チェンジアップWhiff%"),
            ("スライダー", "スライダーWhiff%"),
        ]:
            r = _row_for(pt_name)
            if r is not None:
                try:
                    v = float(r.get("Whiff%"))
                    row[kpi_key] = v if pd.notna(v) else None
                except Exception:
                    row[kpi_key] = None
            else:
                row[kpi_key] = None

        results.append(row)
    return results


def _get_batter_kpi_by_game(df_all, batter_col, batter) -> list[dict]:
    date_col = _pick_col(df_all, ["日付", "試合日", "ゲーム日", "実施日"])
    if not date_col:
        return []
    df_b = df_all[df_all[batter_col].astype(str) == str(batter)].copy()
    df_b["_nd"] = _normalize_date_series(df_b[date_col])
    results = []
    for date, grp in df_b.groupby("_nd"):
        if pd.isna(date) or date == "" or date is None:
            continue
        final = _extract_final_pitches_per_pa(grp)
        basic = calculate_baseball_stats(final)
        saber = calculate_sabermetrics(grp)
        row = {"date": date}
        row["wOBA"]      = basic.get("wOBA", None)
        row["ISO"]       = basic.get("ISO",  None)
        row["Pull-air%"] = saber.get("Pull-air%", None)
        row["O-Swing%"]  = saber.get("O-Swing%", None)
        row["Whiff%"]    = saber.get("Whiff%",   None)
        row["K%"]        = basic.get("K%",  None) or saber.get("K%", None)
        row["BB%"]       = basic.get("BB%", None) or saber.get("BB%", None)
        results.append(row)
    return results


def render():
    from ..components.header import render_page_header
    render_page_header("重要指標推移", icon="📈")

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.warning("data/ フォルダにCSVファイルがありません"); return

    df_all = load_csvs(DATA_DIR)
    if df_all.empty:
        st.error("CSVの読み込みに失敗しました"); return

    pitcher_col = _pick_col(df_all, ["投手名"])
    batter_col  = _pick_col(df_all, ["打者名", "BatterName", "打者"])
    p_team_col  = _pick_col(df_all, ["投手チーム", "PitcherTeam"])
    b_team_col  = _pick_col(df_all, ["打者チーム", "BatterTeam"])

    with st.sidebar:
        st.subheader("KPI設定")
        player_type = st.radio("種別", ["投手", "打者"])

        if player_type == "投手" and pitcher_col:
            player_col = pitcher_col
            team_col = p_team_col
            kpi_defs = PITCHER_KPI
        elif batter_col:
            player_col = batter_col
            team_col = b_team_col
            kpi_defs = BATTER_KPI
        else:
            st.error("選手名列が見つかりません"); return

        # チーム → 選手 の二段階選択
        if team_col:
            teams = sorted(df_all[team_col].dropna().unique().tolist())
            team = st.selectbox("① チーム選択", teams, key=f"kpi_team_{player_type}")
            player_pool = df_all[df_all[team_col] == team]
        else:
            team = None
            player_pool = df_all
        players = sorted(player_pool[player_col].dropna().unique().tolist())
        if not players:
            st.warning("選択チームに選手がいません"); return
        player = st.selectbox("② 選手選択", players, key=f"kpi_player_{player_type}")

        window_opts = {"1試合": 1, "3試合": 3, "5試合": 5, "10試合": 10}
        window_label = st.select_slider("集計単位", options=list(window_opts.keys()), value="1試合")
        window = window_opts[window_label]

        range_opts = {"直近5試合": 5, "直近10試合": 10, "直近20試合": 20, "全期間": 9999}
        range_label = st.selectbox("表示範囲", list(range_opts.keys()), index=1)
        n_games = range_opts[range_label]

        kpi_keys = [k["key"] for k in kpi_defs]
        kpi_labels = [k["label"] for k in kpi_defs]
        sel_kpis = st.multiselect("表示するKPI", kpi_labels,
                                   default=kpi_labels[:4])

    log_event("kpi_trend_view", f"{player} {window_label}")

    # データ取得
    if player_type == "投手":
        game_data = _get_pitcher_kpi_by_game(df_all, player_col, player)
    else:
        game_data = _get_batter_kpi_by_game(df_all, player_col, player)

    if not game_data:
        st.info("試合データがありません"); return

    df_kpi = pd.DataFrame(game_data).sort_values("date").reset_index(drop=True)

    # 表示範囲スライス
    if n_games < 9999:
        df_kpi = df_kpi.tail(n_games).reset_index(drop=True)

    st.subheader(f"{player}　{window_label}単位　{range_label}")

    # 選択KPIのグラフを表示
    for kpi_def in kpi_defs:
        if kpi_def["label"] not in sel_kpis:
            continue
        key = kpi_def["key"]
        if key not in df_kpi.columns:
            continue
        col_data = df_kpi[["date", key]].rename(columns={key: "value"})
        kpi_data = col_data.dropna(subset=["value"]).to_dict("records")
        if not kpi_data:
            continue
        draw_kpi_trend(
            kpi_data,
            kpi_key=key,
            kpi_label=kpi_def["label"],
            unit=kpi_def["unit"],
            better=kpi_def["better"],
            window=window,
            key=f"kpi_trend_{key}",
        )
