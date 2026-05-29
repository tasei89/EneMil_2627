# -*- coding: utf-8 -*-
"""打者分析ページ"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from ..stats.calculations import (
    load_csvs, filter_by_player_period, filter_by_year, get_year_default_start,
    calculate_baseball_stats, calculate_sabermetrics,
    calculate_batter_pitchtype_metrics,
    calculate_first_strike_swing_rate,
    calculate_two_strike_metrics,
    _pick_col, _extract_final_pitches_per_pa, _normalize_date_series,
)
# 新しい calculations.py のみに存在するため防御的に import
try:
    from ..stats.calculations import get_csv_fingerprint
except ImportError:
    import hashlib as _hashlib
    def get_csv_fingerprint(data_dir: str) -> str:
        if not os.path.isdir(data_dir):
            return "empty"
        sig = []
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith(".csv"):
                continue
            p = os.path.join(data_dir, f)
            try:
                st_ = os.stat(p)
                sig.append(f"{f}:{int(st_.st_mtime)}:{st_.st_size}")
            except Exception:
                sig.append(f"{f}:0:0")
        if not sig:
            return "empty"
        return _hashlib.sha1("|".join(sig).encode("utf-8")).hexdigest()[:16]
from ..stats.constants import METRIC_DESCRIPTIONS
from ..components.charts import (
    draw_spray_chart, draw_bullet_chart_percentile,
    draw_pitch_outcome_scatter_by_side,
    render_colored_stats_table, _percentile_of_value,
)
from ..auth import log_event

DATA_DIR = os.environ.get("DATA_DIR", "data")

BATTER_METRIC_DIRECTIONS = {
    # 基本成績
    "打率": "high", "出塁率": "high", "長打率": "high",
    "OPS": "high", "ISO": "high", "wOBA": "high",
    "BB%": "high", "K%": "low",
    # セイバー
    "Whiff%": "low", "SwStr%": "low",
    "O-Swing%": "low",
    "Contact%": "high", "Z-Contact%": "high", "O-Contact%": "high",
    "Swing%": "high", "Z-Swing%": "high",
    "Pull-air%": "high",
    # 球種別
    "Z-Contact%": "high",
    # 2ストライク
    "2s後の平均投球数": "high",     # 投球数を稼げる方が良い
    "2s後の平均ファウル数": "high", # ファウルで粘れる方が良い
    "2s後のO-Swing%": "low",
    # 初球Swing率はカラーリングしない（高低の評価が文脈依存）
}

BULLET_METRICS = ["wOBA", "ISO", "Pull-air%", "O-Swing%", "Whiff%", "K%", "BB%"]
LOWER_IS_BETTER = {"K%", "Whiff%", "O-Swing%"}

# 基本成績で着色する列
BATTER_BASIC_COLOR_COLS = [
    "打率", "出塁率", "長打率", "OPS", "ISO", "wOBA", "BB%", "K%",
]
# セイバーで着色する列
BATTER_SABER_COLOR_COLS = [
    "Swing%", "Z-Swing%", "O-Swing%",
    "Contact%", "Z-Contact%", "O-Contact%",
    "Whiff%", "SwStr%",
    "Pull-air%",
    # wOBA / ISO / K% / BB% は calculate_sabermetrics に含まれる場合も着色
    "wOBA", "ISO", "K%", "BB%",
]
# 球種別で着色する列
PITCHTYPE_COLOR_COLS = [
    "打率", "長打率", "Whiff%", "Swing%", "Contact%",
    "O-Swing%", "Z-Contact%",
]
# 初球スイング率列
FIRST_STRIKE_COLS = ["全打席", "ストレート", "変化球", "カーブ"]
# 2ストライク列
TWO_STRIKE_COLS = ["2s後の平均投球数", "2s後の平均ファウル数", "2s後のO-Swing%"]


@st.cache_data(show_spinner=False)
def _list_teams_and_batters(target_year: str, n: int,
                             _df_year: pd.DataFrame,
                             team_col: str | None,
                             batter_col: str) -> dict:
    """対象年フィルタ済みデータから、チーム一覧と各チームの打者一覧をキャッシュ。
    ★ n を通常引数にしてCSV更新時にキャッシュ invalidate。"""
    if team_col and team_col in _df_year.columns:
        teams = sorted(_df_year[team_col].dropna().unique().tolist())
        by_team = {
            t: sorted(_df_year[_df_year[team_col] == t][batter_col]
                      .dropna().unique().tolist())
            for t in teams
        }
    else:
        teams = []
        by_team = {None: sorted(_df_year[batter_col].dropna().unique().tolist())}
    return {"teams": teams, "by_team": by_team}


@st.cache_data(show_spinner=False)
def _batter_date_range(n: int, _df_all: pd.DataFrame, batter_col: str,
                        batter: str, target_year: str) -> tuple:
    """特定打者の日付範囲（start, end）をキャッシュ。
    ★ n を通常引数にしてCSV更新時にキャッシュ invalidate。"""
    sub = _df_all[_df_all[batter_col] == batter]
    if sub.empty or "_norm_date" not in sub.columns:
        return (None, None)
    dts = pd.to_datetime(sub["_norm_date"], errors="coerce").dropna()
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




# ──────────────────────────────────────────
# 分布データの計算（重い）— 多段キャッシュ機構
# ──────────────────────────────────────────
# L1: モジュールレベル global (_LEAGUE_CACHE) … 全ページ共通 / 数十μs
# L2: ディスク (pickle)                       … アプリ再起動後も復元 / 数百ms
# L3: 計算 (純粋関数 _compute_*)              … 数十秒〜数分
# CSV ファイル群の fingerprint をキーに使うので、CSV を更新すると自動的に
# 新しいファイル名で再計算される。古いファイルは残るが無視される。
_LEAGUE_CACHE: dict = {}


def _get_or_build_dist(data_dir: str, name: str, builder,
                        df_all: pd.DataFrame, *args) -> dict:
    """L1(module) → L2(disk) → L3(compute) の順にフォールバック。"""
    import pickle
    fp = get_csv_fingerprint(data_dir)
    mkey = (fp, name)
    if mkey in _LEAGUE_CACHE:
        return _LEAGUE_CACHE[mkey]
    cache_dir = os.path.join(data_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{name}_{fp}.pkl")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            _LEAGUE_CACHE[mkey] = result
            return result
        except Exception:
            pass
    result = builder(df_all, *args)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    _LEAGUE_CACHE[mkey] = result
    return result


def _compute_batter_distributions(df_all: pd.DataFrame,
                                   min_pa: int = 5) -> dict:
    """全打者の打者詳細指標・基本成績・初球・2ストライクの分布を作る（純粋計算）。"""
    batter_col = _pick_col(df_all, ["打者名", "BatterName", "打者"])
    if not batter_col:
        return {}

    all_cols = list(set(BATTER_BASIC_COLOR_COLS + BATTER_SABER_COLOR_COLS
                        + FIRST_STRIKE_COLS + TWO_STRIKE_COLS
                        + BULLET_METRICS))
    dist = {c: [] for c in all_cols}

    if "打者打席左右" in df_all.columns:
        groups = df_all.groupby([batter_col, "打者打席左右"], sort=False)
    else:
        groups = df_all.groupby(batter_col, sort=False)

    for key, sub in groups:
        if isinstance(key, tuple):
            if pd.isna(key[0]):
                continue
        else:
            if pd.isna(key):
                continue
        if sub.empty:
            continue
        final = _extract_final_pitches_per_pa(sub)
        basic = calculate_baseball_stats(final)
        try:
            pa = int(basic.get("打席", 0))
        except Exception:
            pa = 0
        if pa < min_pa:
            continue
        saber = calculate_sabermetrics(sub)
        first = calculate_first_strike_swing_rate(sub)
        two   = calculate_two_strike_metrics(sub)
        merged = {**basic, **saber, **first, **two}
        for c in dist:
            v = merged.get(c)
            if v is None:
                continue
            try:
                vf = float(v)
                if np.isfinite(vf):
                    dist[c].append(vf)
            except Exception:
                pass
    return {c: np.array(arr, dtype=float) for c, arr in dist.items()}


def _compute_batter_pitchtype_distributions(df_all: pd.DataFrame,
                                             min_pitches: int = 15) -> dict:
    """全打者の球種別セイバーから「球種×指標」分布を作る（純粋計算）。"""
    batter_col = _pick_col(df_all, ["打者名", "BatterName", "打者"])
    if not batter_col:
        return {}
    dist = {}
    for name, sub in df_all.groupby(batter_col, sort=False):
        if pd.isna(name):
            continue
        pt_df = calculate_batter_pitchtype_metrics(sub)
        if pt_df.empty:
            continue
        for _, row in pt_df.iterrows():
            pt = str(row.get("球種"))
            try:
                np_ = float(row.get("球数", 0))
            except Exception:
                np_ = 0
            if np_ < min_pitches:
                continue
            for c in PITCHTYPE_COLOR_COLS:
                if c not in row:
                    continue
                try:
                    vf = float(row[c])
                    if np.isfinite(vf):
                        dist.setdefault(("ALL", c), []).append(vf)
                        dist.setdefault((pt, c), []).append(vf)
                except Exception:
                    pass
    return {k: np.array(arr, dtype=float) for k, arr in dist.items()}


def render():
    from ..components.header import render_page_header
    render_page_header("打者分析", icon="🦍")

    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.warning("data/ フォルダにCSVファイルがありません")
        return

    df_all = load_csvs(DATA_DIR)
    if df_all.empty:
        st.error("CSVの読み込みに失敗しました")
        return

    batter_col = _pick_col(df_all, ["打者名", "BatterName", "打者"])
    team_col = _pick_col(df_all, ["打者チーム", "BatterTeam"])
    if not batter_col:
        st.error("打者名列が見つかりません")
        return

    date_col = _pick_col(df_all, ["日付", "試合日", "ゲーム日", "実施日"])

    # 対象年フィルタ（main.py のサイドバーで設定）
    target_year = st.session_state.get("target_year", "すべて")
    df_year = filter_by_year(df_all, target_year)
    if df_year.empty:
        st.warning(f"{target_year} のデータがありません")
        return

    with st.sidebar:
        st.subheader("フィルター")

        # ★ チーム一覧と各チームの打者一覧をキャッシュから取得
        tb = _list_teams_and_batters(target_year, len(df_year), df_year,
                                       team_col, batter_col)

        # ★ デフォルト選手「姫木陸斗」が所属するチームを探す（初回表示用）
        DEFAULT_BATTER = "姫木陸斗"
        default_team_for_batter = None
        for _t, _members in tb["by_team"].items():
            if DEFAULT_BATTER in _members:
                default_team_for_batter = _t
                break

        if team_col:
            teams = tb["teams"]
            # 初回のみ、姫木陸斗の所属チームを初期選択に（session_state が無いとき）
            team_index = 0
            if (default_team_for_batter is not None
                    and default_team_for_batter in teams
                    and "bat_team" not in st.session_state):
                team_index = teams.index(default_team_for_batter)
            team = st.selectbox("① チーム選択", teams, index=team_index,
                                key="bat_team")
            batters = tb["by_team"].get(team, [])
        else:
            team = None
            batters = tb["by_team"].get(None, [])

        if not batters:
            st.warning("選択チームに打者がいません")
            return
        # 初回のみ、打者リストに姫木陸斗があれば初期選択に
        batter_index = 0
        if (DEFAULT_BATTER in batters
                and "bat_player" not in st.session_state):
            batter_index = batters.index(DEFAULT_BATTER)
        batter = st.selectbox("② 打者選択", batters, index=batter_index,
                              key="bat_player")

        # ──── 開始日/終了日のデフォルト ────
        # 選択された打者の日付範囲をキャッシュから取得
        default_start, default_end = _batter_date_range(
            len(df_all), df_all, batter_col, batter, target_year)

        if date_col:
            # ★ min_value/max_value を打者のデータ範囲に設定し、
            #    終了日は常にその打者の最新データ日が初期値になるようにする。
            if default_start and default_end:
                start_d = st.date_input(
                    "開始日", value=default_start,
                    min_value=default_start, max_value=default_end,
                    key=f"bat_start_{batter}_{target_year}",
                )
                end_d = st.date_input(
                    "終了日", value=default_end,
                    min_value=default_start, max_value=default_end,
                    key=f"bat_end_{batter}_{target_year}",
                )
            else:
                start_d = st.date_input(
                    "開始日", value=default_start,
                    key=f"bat_start_{batter}_{target_year}",
                )
                end_d = st.date_input(
                    "終了日", value=default_end,
                    key=f"bat_end_{batter}_{target_year}",
                )
            start_str = str(start_d) if start_d else None
            end_str = str(end_d) if end_d else None
        else:
            start_str = end_str = None

        sit_opts = ["全体", "ランナーなし", "得点圏", "vs左投手", "vs右投手"]
        situation = st.selectbox("シチュエーション", sit_opts)

    df = filter_by_player_period(df_all, batter_col, batter,
                                 start_str, end_str)
    if df.empty:
        # ★ フォールバック：日付フィルタが原因で空になった可能性があるので、
        #    日付フィルタを外して打者データだけで再試行
        df_no_date = filter_by_player_period(df_all, batter_col, batter,
                                              None, None)
        if df_no_date.empty:
            st.warning(f"「{batter}」のデータが見つかりません。")
            return
        st.warning(
            f"指定された期間（{start_str} 〜 {end_str}）に「{batter}」のデータがありません。"
            f"\n\nこの選手のデータは {df_no_date['_norm_date'].dropna().min()} 〜 "
            f"{df_no_date['_norm_date'].dropna().max()} の範囲に存在します。"
            f"\n\n→ 全期間のデータを使って表示します。"
        )
        df = df_no_date

    df_sit = _apply_situation(df, situation)
    log_event("batter_stats_view", batter)

    header = f"🦍 {batter}"
    if team:
        header += f"（{team}）"
    header += f"　— {situation}"
    st.subheader(header)

    # ──── 分布データの事前計算（リーグ全体・全期間データ） ────
    # ★ 対象年フィルタとは独立に、必ず df_all（全期間）から計算する
    # 多段キャッシュ: モジュールlevel global → ディスク pickle → 計算
    dist = _get_or_build_dist(
        DATA_DIR, "dist_batter", _compute_batter_distributions, df_all)
    dist_pt = _get_or_build_dist(
        DATA_DIR, "dist_batter_pt", _compute_batter_pitchtype_distributions, df_all)

    # 当選手の基本＋打者詳細を計算（バレットチャートと基本成績で共有）
    final_pa = _extract_final_pitches_per_pa(df_sit)
    basic = calculate_baseball_stats(final_pa, situation)
    basic_disp = {k: v for k, v in basic.items() if k != "状況"}
    saber = calculate_sabermetrics(df_sit)

    # ──── バレットチャート（最上部に配置）────
    st.markdown("### JABAパーセンタイル")
    merged_for_bullet = {**basic_disp, **saber}
    bullet_items = []
    for m in BULLET_METRICS:
        v = merged_for_bullet.get(m)
        try:
            vf = float(v) if v is not None and not pd.isna(v) else float("nan")
        except Exception:
            vf = float("nan")
        arr = dist.get(m, np.array([], dtype=float))
        p_raw = _percentile_of_value(vf, arr) if np.isfinite(vf) else float("nan")
        p = (1.0 - p_raw) if (m in LOWER_IS_BETTER and np.isfinite(p_raw)) else p_raw
        if not np.isfinite(vf):
            vtxt = "—"
        elif m in ("wOBA", "ISO"):
            vtxt = f"{vf:.3f}"
        else:
            vtxt = f"{vf:.1f}"
        bullet_items.append({
            "label": m, "value": vf, "percentile": p, "value_text": vtxt,
        })
    draw_bullet_chart_percentile(bullet_items, title="", key="bat_bullet")

    st.divider()

    # ──── 基本成績 ────
    st.markdown("#### 基本成績")
    basic_dist = {c: dist[c] for c in BATTER_BASIC_COLOR_COLS if c in dist}
    render_colored_stats_table(
        pd.DataFrame([basic_disp]),
        metric_directions=BATTER_METRIC_DIRECTIONS,
        key="bat_basic_tbl",
        distributions=basic_dist,
        int_cols=["打席", "打数", "安打", "二塁打", "三塁打",
                  "本塁打", "四死球", "三振", "打点"],
        decimals_2_cols=["打率", "出塁率", "長打率", "OPS", "ISO", "wOBA"],
    )

    # ──── 打者詳細指標 ────
    st.markdown("#### 打者詳細指標")
    if saber:
        saber_dist = {c: dist[c] for c in BATTER_SABER_COLOR_COLS if c in dist}
        render_colored_stats_table(
            pd.DataFrame([saber]),
            metric_directions=BATTER_METRIC_DIRECTIONS,
            key="bat_saber_tbl",
            distributions=saber_dist,
            decimals_2_cols=list(saber.keys()),
        )

    st.caption(
        "🟦青＝下位 / ⚪白＝中位 / 🟥赤＝上位（リーグ内パーセンタイル比較）。"
        "「K%・Whiff%・O-Swing%」など低い方が良い指標は色を反転しています。"
    )

    with st.expander("📖 指標の説明"):
        seen = set()
        for col in list(basic_disp.keys()) + list(saber.keys()):
            if col in seen or col not in METRIC_DESCRIPTIONS:
                continue
            seen.add(col)
            info = METRIC_DESCRIPTIONS[col]
            st.markdown(f"**{col}**: {info['desc']}  \n"
                        f"`計算式: {info['formula']}`")

    st.divider()

    # ──── シチュエーション別セイバー比較（4行） ────
    st.markdown("### シチュエーション別セイバー比較")
    cmp_df = _build_situation_comparison(df)
    if cmp_df is None or cmp_df.empty:
        st.info("比較データを作成できませんでした")
    else:
        # 比較表は行間で着色（min/max が同行内で複数値ある）
        render_colored_stats_table(
            cmp_df,
            metric_directions=BATTER_METRIC_DIRECTIONS,
            key="bat_sit_cmp",
            min_count=1, count_col="打席",
            int_cols=["打席", "打数"],
            decimals_2_cols=["打率", "出塁率", "長打率", "OPS",
                             "wOBA", "ISO", "BB%", "K%",
                             "Pull-air%", "O-Swing%", "Whiff%", "Contact%"],
        )

    st.divider()

    # ──── 球種別セイバー ────
    # 見出しの右に「すべて / 対右投手 / 対左投手」のトグル
    head_l, head_r = st.columns([4, 6])
    with head_l:
        st.markdown("### 球種別の成績")
    with head_r:
        pt_split = st.radio(
            "対投手", ["すべて", "対右投手", "対左投手"],
            horizontal=True, label_visibility="collapsed",
            key="bat_pt_split",
        )

    # トグル適用：対投手で df_sit をさらに絞り込む
    if pt_split == "対右投手":
        pt_source = _apply_situation(df_sit, "vs右投手")
    elif pt_split == "対左投手":
        pt_source = _apply_situation(df_sit, "vs左投手")
    else:
        pt_source = df_sit

    pt_df = calculate_batter_pitchtype_metrics(pt_source)
    if pt_df.empty:
        st.info(f"{pt_split}の球種別データがありません")
    else:
        # 各列ごとに「ALL 全打者×全球種」の分布で着色
        pt_dist_for_table = {}
        for c in PITCHTYPE_COLOR_COLS:
            key = ("ALL", c)
            if key in dist_pt:
                pt_dist_for_table[c] = dist_pt[key]
        render_colored_stats_table(
            pt_df,
            metric_directions=BATTER_METRIC_DIRECTIONS,
            key=f"bat_pt_tbl_{pt_split}",
            min_count=5, count_col="球数",
            distributions=pt_dist_for_table,
            int_cols=["球数", "打数", "安打"],
            decimals_2_cols=["打率", "長打率",
                             "Whiff%", "Swing%", "Contact%",
                             "O-Swing%", "Z-Contact%"],
        )

    st.divider()

    # ──── 初球スイング率 & 2ストライク時メトリクス（横並び） ────
    cn_l, cn_r = st.columns([4, 6])
    with cn_l:
        st.markdown("### カウント別の成績")
    with cn_r:
        cnt_split = st.radio(
            "対投手cnt", ["すべて", "対右投手", "対左投手"],
            horizontal=True, label_visibility="collapsed",
            key="bat_cnt_split",
        )
    if cnt_split == "対右投手":
        cnt_source = _apply_situation(df_sit, "vs右投手")
    elif cnt_split == "対左投手":
        cnt_source = _apply_situation(df_sit, "vs左投手")
    else:
        cnt_source = df_sit

    col_fs, col_ts = st.columns([4, 6])
    with col_fs:
        st.markdown("**初球（カウント0-0時）スイング率**")
        fs_dict = calculate_first_strike_swing_rate(cnt_source)
        fs_df = pd.DataFrame([{c: fs_dict.get(c, np.nan)
                               for c in FIRST_STRIKE_COLS}])
        fs_dist = {c: dist[c] for c in FIRST_STRIKE_COLS if c in dist}
        render_colored_stats_table(
            fs_df,
            metric_directions={c: "high" for c in FIRST_STRIKE_COLS},
            key=f"bat_fs_tbl_{cnt_split}",
            distributions=fs_dist,
            decimals_2_cols=FIRST_STRIKE_COLS,
        )
    with col_ts:
        st.markdown("**2ストライク到達後の成績**")
        ts_dict = calculate_two_strike_metrics(cnt_source)
        ts_df = pd.DataFrame([{c: ts_dict.get(c, np.nan)
                               for c in TWO_STRIKE_COLS}])
        ts_dist = {c: dist[c] for c in TWO_STRIKE_COLS if c in dist}
        render_colored_stats_table(
            ts_df,
            metric_directions=BATTER_METRIC_DIRECTIONS,
            key=f"bat_ts_tbl_{cnt_split}",
            distributions=ts_dist,
            decimals_2_cols=TWO_STRIKE_COLS,
        )
    st.caption(
        "・初球スイング率：カウント 0-0 時にバッターがスイングした割合。"
        "・2s後のO-Swing%：2ストライク後にボールゾーンに対してスイングした割合（低い方が良い）。"
        "・2s後の平均投球数/ファウル数：2ストライク到達後の粘り。多いほど良い。"
    )

    st.divider()

    # ──── 投球コース（投球到達位置：安打・凡打・空振り・ファウル） ────
    pc_l, pc_r = st.columns([4, 6])
    with pc_l:
        st.markdown("### 投球コース")
    with pc_r:
        pc_view = st.radio(
            "ヒートマップ視点",
            ["捕手目線", "投手目線"],
            horizontal=True, label_visibility="collapsed",
            key="bat_pc_view",
        )
    pc_view_key = "catcher" if pc_view == "捕手目線" else "pitcher"
    draw_pitch_outcome_scatter_by_side(
        df_sit, view=pc_view_key, key_prefix="bat_pco",
    )

    st.divider()

    # ──── スプレーチャート（対右投手・対左投手 横並び） ────
    sp_l, sp_r = st.columns([4, 6])
    with sp_l:
        st.markdown("### 打球方向")
    with sp_r:
        # 球種選択（"すべて" + 当該打者が見た球種）
        pt_opts = ["すべて"]
        if "球種" in df_sit.columns:
            seen_pts = (df_sit["球種"].dropna().astype(str).unique().tolist())
            pt_opts += sorted(seen_pts)
        sel_pt = st.selectbox(
            "球種", pt_opts, index=0,
            key="bat_spray_pt", label_visibility="collapsed",
        )

    # 球種フィルタを適用
    if sel_pt != "すべて" and "球種" in df_sit.columns:
        df_for_spray = df_sit[df_sit["球種"].astype(str) == sel_pt]
    else:
        df_for_spray = df_sit
    _draw_spray_by_pitcher_hand(df_for_spray, key_suffix=sel_pt)


def _build_situation_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """全体・対右投手・対左投手・得点圏 の 4 行のセイバー比較表を作る。"""
    rows = []
    seq = [
        ("全体",       df),
        ("対右投手",   _apply_situation(df, "vs右投手")),
        ("対左投手",   _apply_situation(df, "vs左投手")),
        ("得点圏",     _apply_situation(df, "得点圏")),
    ]
    cmp_cols = [
        "打席", "打数", "打率", "出塁率", "長打率", "OPS",
        "wOBA", "ISO", "BB%", "K%",
        "Pull-air%", "O-Swing%", "Whiff%", "Contact%",
    ]
    for label, sub in seq:
        if sub is None or sub.empty:
            row = {"シチュエーション": label}
            for c in cmp_cols:
                row[c] = np.nan
            rows.append(row)
            continue
        final = _extract_final_pitches_per_pa(sub)
        basic = calculate_baseball_stats(final)
        saber = calculate_sabermetrics(sub)
        merged = {**basic, **saber}
        row = {"シチュエーション": label}
        for c in cmp_cols:
            row[c] = merged.get(c, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_spray_by_pitcher_hand(df: pd.DataFrame, key_suffix: str = ""):
    """対右投手・対左投手のスプレーチャートを横並び表示。"""
    hc = _pick_col(df, ["投手投げ手左右", "投手投球左右", "PitcherHand"])
    if not hc:
        st.info("投手の利き腕列が見つかりません")
        draw_spray_chart(df, key=f"bat_spray_all_{key_suffix}")
        return

    df_R = df[df[hc].astype(str).str.contains("右|R", na=False)]
    df_L = df[df[hc].astype(str).str.contains("左|L", na=False)]
    col_R, col_L = st.columns(2)
    with col_R:
        if df_R.empty:
            st.markdown("**対右投手**")
            st.caption("対右投手のデータがありません")
        else:
            draw_spray_chart(df_R, key=f"bat_spray_R_{key_suffix}",
                             height=440, title="対右投手")
    with col_L:
        if df_L.empty:
            st.markdown("**対左投手**")
            st.caption("対左投手のデータがありません")
        else:
            draw_spray_chart(df_L, key=f"bat_spray_L_{key_suffix}",
                             height=440, title="対左投手")


def _apply_situation(df: pd.DataFrame, sit: str) -> pd.DataFrame:
    if sit == "全体":
        return df
    if sit == "ランナーなし":
        rc_col = _pick_col(df, ["走者状況", "ランナー状況", "Runner"])
        if rc_col:
            return df[df[rc_col].astype(str).str.contains(
                r"なし|000|empty", case=False, na=False, regex=True)]
    if sit == "得点圏":
        sc_col = _pick_col(df, ["走者得点圏"])
        if sc_col:
            return df[pd.to_numeric(df[sc_col], errors="coerce") >= 1]
        rc_col = _pick_col(df, ["走者状況", "ランナー状況", "Runner"])
        if rc_col:
            return df[df[rc_col].astype(str).str.contains(
                r"二塁|三塁|2|3", na=False, regex=True)]
    if sit == "vs左投手":
        hc = _pick_col(df, ["投手投げ手左右", "投手投球左右", "PitcherHand"])
        if hc:
            return df[df[hc].astype(str).str.contains("左|L", na=False)]
    if sit == "vs右投手":
        hc = _pick_col(df, ["投手投げ手左右", "投手投球左右", "PitcherHand"])
        if hc:
            return df[df[hc].astype(str).str.contains("右|R", na=False)]
    return df
