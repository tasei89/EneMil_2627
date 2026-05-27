# -*- coding: utf-8 -*-
"""投手試合データページ（特定日・特定チームの投手スタッツ）"""
import os, streamlit as st, pandas as pd
from ..stats.calculations import (load_csvs, filter_by_team_date,
    calculate_pitcher_stats_combined, calculate_game_metrics,
    calculate_pitchtype_metrics, _pick_col, _normalize_date_series)
from ..components.charts import (draw_zone_heatmap_by_side, draw_velocity_histogram,
    draw_pitch_donut_split, render_colored_stats_table)
from ..auth import log_event

DATA_DIR = os.environ.get("DATA_DIR", "data")


def _build_game_options(df_all: pd.DataFrame, team: str,
                        team_col: str, date_col: str) -> list:
    """[(game_key, display_label)] のリストを返す。
    game_key = "YYYY-MM-DD||対戦相手" の形式（同日複数試合の区別用）。
    label = "YYYY-MM-DD（vs 対戦相手）"。"""
    if not date_col:
        return []
    sub = df_all[df_all[team_col] == team].copy() if team_col else df_all.copy()
    if sub.empty:
        return []
    sub["_nd"] = _normalize_date_series(sub[date_col])
    opp_col = _pick_col(sub, ["打者チーム", "BatterTeam",
                              "team_batter", "相手チーム"])

    out = []
    if opp_col:
        # (日付, 相手) ごとに 1 試合として扱う
        for (d, opp), grp in sub.groupby(["_nd", opp_col]):
            if pd.isna(d) or d == "" or pd.isna(opp):
                continue
            if str(opp) == team:
                continue  # 自チームは相手にカウントしない
            game_key = f"{d}||{opp}"
            label = f"{d}（vs {opp}）"
            out.append((game_key, label, d))
    else:
        # 相手列がない場合は日付のみで集約（旧挙動）
        for d, grp in sub.groupby("_nd"):
            if pd.isna(d) or d == "":
                continue
            game_key = f"{d}||"
            label = str(d)
            out.append((game_key, label, d))

    # 新しい順
    out.sort(key=lambda x: x[2], reverse=True)
    return [(k, lbl) for k, lbl, _ in out]


def render():
    from ..components.header import render_page_header
    render_page_header("投手試合データ", icon="📋")

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.warning("data/ フォルダにCSVファイルがありません"); return

    df_all = load_csvs(DATA_DIR)
    if df_all.empty:
        st.error("CSVの読み込みに失敗しました"); return

    # 対象年フィルタ
    from ..stats.calculations import filter_by_year
    target_year = st.session_state.get("target_year", "すべて")
    df_year = filter_by_year(df_all, target_year)
    if df_year.empty:
        st.warning(f"{target_year} のデータがありません")
        return

    team_col = _pick_col(df_year, ["投手チーム", "PitcherTeam", "team_pitcher"])
    date_col = _pick_col(df_year, ["日付", "試合日", "ゲーム日", "実施日"])

    teams = sorted(df_year[team_col].dropna().unique()) if team_col else []

    with st.sidebar:
        st.subheader("試合選択")
        team = st.selectbox("チーム", teams) if teams else st.text_input("チーム名")

        # 試合セレクタ：同日複数試合は (日付, 対戦相手) で区別
        game_opts = _build_game_options(df_year, team, team_col, date_col) if team else []
        sel_game_key = None
        if game_opts:
            display_labels = [lbl for _, lbl in game_opts]
            label2key = {lbl: k for k, lbl in game_opts}
            sel_label = st.selectbox("試合", display_labels, key="gs_game")
            sel_game_key = label2key[sel_label]
        else:
            sel_game_key = None

        view = st.radio("ヒートマップ視点", ["捕手目線", "投手目線"],
                        horizontal=True, key="gs_view")
        view_key = "catcher" if view == "捕手目線" else "pitcher"

    if not team or not sel_game_key:
        st.info("チームと試合を選択してください")
        return

    # game_key を分解
    if "||" in sel_game_key:
        date, opp_from_key = sel_game_key.split("||", 1)
    else:
        date, opp_from_key = sel_game_key, ""

    # (チーム, 日付) でまず絞り、さらに対戦相手で絞り込む
    df_game = filter_by_team_date(df_year, team, date)
    if opp_from_key:
        opp_col = _pick_col(df_game, ["打者チーム", "BatterTeam",
                                      "team_batter", "相手チーム"])
        if opp_col:
            df_game = df_game[df_game[opp_col].astype(str) == opp_from_key]

    if df_game.empty:
        st.warning(f"{team} / {date} / vs {opp_from_key} のデータが見つかりません"); return

    # 対戦相手
    opp_col = _pick_col(df_game, ["打者チーム", "BatterTeam", "team_batter", "相手チーム"])
    opp = (df_game[opp_col].dropna().iloc[0]
           if (opp_col and df_game[opp_col].notna().any())
           else (opp_from_key or "Unknown"))

    log_event("game_summary_view", f"{team} vs {opp} {date}")
    st.subheader(f"{team}  vs  {opp}　　{date}")

    pitcher_col = _pick_col(df_game, ["投手名"])
    if not pitcher_col:
        st.error("投手名列が見つかりません"); return

    pitchers = df_game[pitcher_col].dropna().unique().tolist()

    tabs = st.tabs(pitchers)
    for ti, (tab, name) in enumerate(zip(tabs, pitchers)):
        with tab:
            viz = df_game[df_game[pitcher_col] == name].copy()

            # ① 基本成績
            st.markdown("**① 基本成績**")
            render_colored_stats_table(
                pd.DataFrame([calculate_pitcher_stats_combined(viz)]),
                key=f"gs_basic_{ti}",
                int_cols=["球数", "打席", "打数", "自責点", "被安打",
                          "被本塁打", "奪三振", "四球", "死球"],
            )

            # ② 投手詳細指標
            st.markdown("**② 投手詳細指標**")
            render_colored_stats_table(
                pd.DataFrame([calculate_game_metrics(viz)]),
                key=f"gs_metrics_{ti}",
            )

            # ③ 球種別の成績（色付けなし）
            st.markdown("**③ 球種別の成績**")
            pt_df = calculate_pitchtype_metrics(viz)
            if not pt_df.empty:
                # 球種別の成績は色付けしない（投球数も含めて全部）
                no_color_cols = ["投球数", "投球割合", "球速比（％）",
                                 "Zone%", "Chase%", "Whiff%", "SwStr%",
                                 "PutAway%", "GB%", "FB%", "被打率"]
                render_colored_stats_table(
                    pt_df, key=f"gs_pt_{ti}",
                    int_cols=["投球数"],
                    decimals_2_cols=["被打率"],
                    no_color_cols=no_color_cols,
                )

            # ④ 可視化
            st.markdown("**④ 球種割合（対右 / 対左）**")
            draw_pitch_donut_split(viz, key_prefix=f"gs_donut_{ti}")

            st.markdown("**⑤ 投球コース（球種別ヒートマップ）**")
            draw_zone_heatmap_by_side(viz, title="投球コース",
                                      view=view_key, top_n=4,
                                      key_prefix=f"gs_hm_{ti}")

            st.markdown("**⑥ 球速分布**")
            draw_velocity_histogram(viz, pitcher_name=name,
                                    key=f"gs_velo_{ti}")
