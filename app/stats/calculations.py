# -*- coding: utf-8 -*-
"""投手・打者スタッツ計算関数（ノートブックから移植）"""
import pandas as pd
import numpy as np
from .constants import WOBA_WEIGHTS


# ──────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────
def format_innings_from_outs(outs: int) -> str:
    try:
        outs = int(outs)
    except Exception:
        return "0"
    if outs <= 0:
        return "0"
    whole, rem = divmod(outs, 3)
    if rem == 0:
        return str(whole)
    if whole == 0:
        return f"{rem}/3"
    return f"{whole} {rem}/3"


def _pick_col(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_date_series(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip()
    ss = (ss.str.replace("年", "-", regex=False)
            .str.replace("月", "-", regex=False)
            .str.replace("日", "",  regex=False)
            .str.replace("/",  "-", regex=False)
            .str.replace(".",  "-", regex=False))
    return pd.to_datetime(ss, errors="coerce").dt.strftime("%Y-%m-%d")


def load_csvs(data_dir: str) -> pd.DataFrame:
    """data/ 内の全 CSV を結合して返す。
    ・プロセス内キャッシュ (ファイルの mtime/size をキー)
    ・ディスクキャッシュ (data/.cache.parquet) でアプリ再起動後も高速
    ・並列読込で大量CSV (650+ファイル) を高速化"""
    import os
    from concurrent.futures import ThreadPoolExecutor
    global _CSV_CACHE
    try:
        _CSV_CACHE
    except NameError:
        _CSV_CACHE = {"key": None, "df": None}

    if not os.path.isdir(data_dir):
        return pd.DataFrame()

    # キャッシュキー: スキーマバージョン + ファイル一覧と各ファイルの mtime/size
    # ★ スキーマを変更したらこのバージョンを上げる（旧パーケットを自動再ビルドする）
    # v2 = _norm_date / _year 列を事前計算してキャッシュ
    CACHE_SCHEMA_VERSION = "v2"
    sig = [("__schema__", CACHE_SCHEMA_VERSION)]
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".csv"):
            continue
        p = os.path.join(data_dir, f)
        try:
            stt = os.stat(p)
            sig.append((f, int(stt.st_mtime), stt.st_size))
        except Exception:
            sig.append((f, 0, 0))
    key = tuple(sig)

    # プロセス内キャッシュ
    if _CSV_CACHE["key"] == key and _CSV_CACHE["df"] is not None:
        return _CSV_CACHE["df"]

    # ディスクキャッシュ（再起動後もインスタント読込）
    cache_dir = os.path.join(data_dir, ".cache")
    cache_path = os.path.join(cache_dir, "merged.parquet")
    key_path   = os.path.join(cache_dir, "key.txt")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        if os.path.isfile(cache_path) and os.path.isfile(key_path):
            with open(key_path, "r", encoding="utf-8") as f:
                cached_key = f.read().strip()
            if cached_key == str(key):
                df = pd.read_parquet(cache_path)
                _CSV_CACHE["key"] = key
                _CSV_CACHE["df"] = df
                return df
    except Exception:
        # parquet 読込で失敗したらフォールスルー
        pass

    # 並列読込（I/O中心なので ThreadPool でGILを回避できる）
    def _read_one(fname):
        path = os.path.join(data_dir, fname)
        for enc in ("utf-8", "cp932", "utf-8-sig"):
            try:
                tmp = pd.read_csv(path, encoding=enc)
                tmp["_source_csv"] = fname
                return tmp
            except Exception:
                continue
        return None

    # files にはスキーマ要素 ("__schema__", "v2") も含まれるので、
    # CSV ファイル名（3要素タプルの先頭）だけ抽出する
    files = [entry[0] for entry in sig
             if len(entry) == 3 and isinstance(entry[0], str)
             and entry[0].endswith(".csv")]
    if not files:
        return pd.DataFrame()

    # ファイル数に応じてワーカー数を調整（最大 16）
    workers = min(16, max(4, len(files) // 32))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_read_one, files))
    dfs = [r for r in results if r is not None]
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    if "投手氏名" in df.columns:
        df.rename(columns={"投手氏名": "投手名"}, inplace=True)

    # ★ パフォーマンス改善：日付列を一度だけ正規化して保存
    # 各ページが filter_by_year で毎回 pd.to_datetime を再計算するのを避ける
    date_col = _pick_col(df, ["日付", "試合日", "ゲーム日", "実施日"])
    if date_col:
        norm = _normalize_date_series(df[date_col])
        df["_norm_date"] = norm
        dt = pd.to_datetime(norm, errors="coerce")
        # 年は Int16 で保存（NaN は -1 ではなく Int16 の NA）
        df["_year"] = dt.dt.year.astype("Int16")

    # ディスクキャッシュに保存（失敗しても続行）
    try:
        df.to_parquet(cache_path, index=False)
        with open(key_path, "w", encoding="utf-8") as f:
            f.write(str(key))
    except Exception:
        pass

    _CSV_CACHE["key"] = key
    _CSV_CACHE["df"] = df
    return df


def filter_by_year(df: pd.DataFrame, year_label: str) -> pd.DataFrame:
    """対象年フィルタ。year_label は "すべて" or "2026年" 形式。
    load_csvs が事前計算した _year 列があれば即時マスク（数十万行でも数ms）。"""
    if not year_label or year_label == "すべて" or df.empty:
        return df
    try:
        target = int(str(year_label).replace("年", "").strip())
    except Exception:
        return df
    # ★ 高速パス：load_csvs で _year を事前計算済み
    if "_year" in df.columns:
        return df[df["_year"] == target]
    # フォールバック（旧式）：列が無い場合のみ pd.to_datetime
    date_col = _pick_col(df, ["日付", "試合日", "ゲーム日", "実施日"])
    if not date_col:
        return df
    nd = pd.to_datetime(_normalize_date_series(df[date_col]), errors="coerce")
    mask = nd.dt.year == target
    return df[mask]


def get_year_default_start(year_label: str) -> str:
    """対象年フィルタから「開始日」のデフォルトを返す（YYYY-01-01）。
    "すべて" のときは空文字を返す（=指定なし）。"""
    if not year_label or year_label == "すべて":
        return ""
    try:
        target = int(str(year_label).replace("年", "").strip())
        return f"{target}-01-01"
    except Exception:
        return ""


def filter_by_team_date(df: pd.DataFrame, team: str, date: str) -> pd.DataFrame:
    date_col = _pick_col(df, ["日付", "試合日", "ゲーム日", "実施日", "Date", "game_date"])
    team_col = _pick_col(df, ["投手チーム", "PitcherTeam", "team_pitcher"])
    if not date_col or not team_col:
        return pd.DataFrame()
    # ★ load_csvs が事前計算した _norm_date 列を優先使用
    if "_norm_date" in df.columns:
        norm = df["_norm_date"]
    else:
        norm = _normalize_date_series(df[date_col])
    target = pd.to_datetime(date.replace("/", "-")).strftime("%Y-%m-%d")
    mask = (df[team_col].astype(str) == str(team)) & (norm == target)
    return df[mask].copy()


def filter_by_player_period(df: pd.DataFrame, player_col: str, player: str,
                             start: str = None, end: str = None) -> pd.DataFrame:
    df2 = df[df[player_col].astype(str) == str(player)]
    if not (start or end):
        return df2.copy()
    date_col = _pick_col(df2, ["日付", "試合日", "ゲーム日", "実施日", "Date", "game_date"])
    if not date_col:
        return df2.copy()
    # ★ load_csvs が事前計算した _norm_date 列を優先使用
    if "_norm_date" in df2.columns:
        nd = df2["_norm_date"]
    else:
        nd = _normalize_date_series(df2[date_col])
    mask = pd.Series(True, index=df2.index)
    if start:
        mask &= (nd >= start)
    if end:
        mask &= (nd <= end)
    return df2[mask].copy()


# ──────────────────────────────────────────
# 投手：打席最終行の抽出
# ──────────────────────────────────────────
def _extract_final_pitches_per_pa(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    if "打席結果" not in d.columns:
        return d.iloc[0:0]
    is_final = d["打席結果"].notna() & (~d["打席結果"].isin(["投球", "牽制"]))
    d["pa_id"] = is_final.shift(1, fill_value=True).cumsum()
    return d.loc[d.groupby("pa_id").tail(1).index].copy()


# ──────────────────────────────────────────
# 投手：基本成績
# ──────────────────────────────────────────
def calculate_pitcher_stats_combined(data: pd.DataFrame) -> dict:
    d = data.copy()
    outs = 0
    if "アウトカウント" in d.columns and "投球後アウトカウント" in d.columns:
        diff = (pd.to_numeric(d["投球後アウトカウント"], errors="coerce") -
                pd.to_numeric(d["アウトカウント"],       errors="coerce")).fillna(0)
        outs = int(diff.clip(lower=0).sum())

    ip_float = outs / 3.0
    innings_str = format_innings_from_outs(outs)

    final = _extract_final_pitches_per_pa(d)
    if final.empty or "打席結果" not in final.columns:
        h = hr = so = bb = hbp = ab = pa = 0
        ba = 0.0
    else:
        res = final["打席結果"].astype(str)
        h   = int(res.str.contains(r"安|２|３|本", na=False, regex=True).sum())
        hr  = int(res.str.contains("本", na=False).sum())
        so  = int(res.str.contains("三振", na=False).sum())
        bb  = int(res.str.contains("四球", na=False).sum())
        hbp = int(res.str.contains("死球", na=False).sum())
        ab  = int(final[~res.str.contains(r"四球|死球|犠|失策", na=False, regex=True)].shape[0])
        bb_hbp = int(res.str.contains("球", na=False).sum())
        sac_b  = int(res.str.contains(r"犠打|犠バント|犠牲打", na=False, regex=True).sum())
        sac_f  = int(res.str.contains(r"犠飛|犠牲飛", na=False, regex=True).sum())
        pa  = ab + bb_hbp + sac_b + sac_f
        ba  = h / ab if ab > 0 else 0.0

    er   = int(pd.to_numeric(d["自責点"], errors="coerce").fillna(0).sum()) if "自責点" in d.columns else 0
    era  = er * 9 / ip_float if ip_float > 0 else 0.0
    whip = (h + bb) / ip_float if ip_float > 0 else 0.0
    k_pct   = so / pa * 100 if pa > 0 else 0.0
    bb_pct  = bb / pa * 100 if pa > 0 else 0.0
    total_p = int(d["球種"].dropna().count()) if "球種" in d.columns else len(d)

    return {
        "投球回": innings_str, "球数": total_p, "打席": pa, "打数": ab,
        "自責点": er, "被安打": h, "被本塁打": hr, "奪三振": so, "四球": bb, "死球": hbp,
        "被打率": round(ba, 3), "防御率": round(era, 2), "WHIP": round(whip, 2),
        "K%": round(k_pct, 1), "BB%": round(bb_pct, 1),
        "K-BB%": round(k_pct - bb_pct, 1),
        "K/BB": round(so / bb, 2) if bb > 0 else 0.0,
    }


# ──────────────────────────────────────────
# 投手：ゲームメトリクス（Strike%, 3球追い込み% 等）
# ──────────────────────────────────────────
def calculate_game_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    pr  = df["投球結果"].astype(str) if "投球結果" in df.columns else pd.Series(dtype=str)
    pa  = df["打席結果"].astype(str) if "打席結果" in df.columns else pd.Series(dtype=str)
    bq  = df["打球性質"].astype(str) if "打球性質" in df.columns else pd.Series(dtype=str)
    total = len(df)

    ball_cnt = pr.eq("ボール").sum()
    hbp_cnt  = pa.str.contains("死球", na=False).sum()
    strike_pct = round((total - ball_cnt - hbp_cnt) / total * 100, 1) if total else 0.0

    # 1st-Strike%
    if "打席内球数" in df.columns:
        fp = df[pd.to_numeric(df["打席内球数"], errors="coerce") == 1]
        if len(fp):
            fp_pr  = fp["投球結果"].astype(str) if "投球結果" in fp.columns else pd.Series(dtype=str)
            fp_pa  = fp["打席結果"].astype(str)  if "打席結果"  in fp.columns else pd.Series(dtype=str)
            fp_b   = fp_pr.eq("ボール").sum()
            fp_hbp = fp_pa.str.contains("死球", na=False).sum()
            first_strike_pct = round((len(fp) - fp_b - fp_hbp) / len(fp) * 100, 1)
        else:
            first_strike_pct = 0.0
    else:
        first_strike_pct = 0.0

    # Whiff%
    whiff_n  = int(pr.eq("空振り").sum())
    swing_n  = int(pr.isin(["空振り", "ファウル"]).sum()) + int(bq.isin(["ゴロ", "フライ", "ライナー"]).sum())
    whiff_pct = round(whiff_n / swing_n * 100, 1) if swing_n > 0 else 0.0

    # 3球追い込み%（仕様変更：新ノートブック準拠）
    # ★ 1球目・2球目で打席が終わったものは対象外
    # ★ 3球目で打席が終わった場合：3球三振なら状況発生+成功、それ以外（インプレイ）は分母から除外
    # ★ 4球目以降がある打席：状況発生として数え、4球目を投げる前のストライクカウントが2以上なら成功
    #
    # 【パフォーマンス】Python for ループでは35万行で数十秒かかるため、
    # groupby.agg + ベクトル演算で計算する。
    three_count = 0
    three_success = 0
    if "打席内球数" in df.columns:
        d_tmp = df.copy()
        d_tmp["_pn"] = pd.to_numeric(d_tmp["打席内球数"], errors="coerce")
        # _pn==1 ごとに pa_id を切る
        d_tmp["_pa_id"] = d_tmp["_pn"].eq(1).cumsum()

        # 各打席（pa_id）から必要な4つの値を一括で取り出す：
        #   max_pn          ：打席内最大球数
        #   pa_result       ：最終行の打席結果
        #   s_cnt_pn4       ：4球目の「ストライクカウント」（無ければ NaN）
        #   post_s_cnt_pn3  ：3球目の「投球後ストライクカウント」（fallback 用）
        d_tmp["_pa_res_str"] = (d_tmp.get("打席結果",
                                pd.Series([""] * len(d_tmp))).astype(str))
        d_tmp["_s_cnt"]      = pd.to_numeric(
            d_tmp.get("ストライクカウント", pd.Series([np.nan] * len(d_tmp))),
            errors="coerce")
        d_tmp["_post_s_cnt"] = pd.to_numeric(
            d_tmp.get("投球後ストライクカウント", pd.Series([np.nan] * len(d_tmp))),
            errors="coerce")

        grp = d_tmp.groupby("_pa_id", sort=False)
        max_pn_s = grp["_pn"].max()

        # 各打席の最終行の打席結果（_pn==max の行を取る）。
        # groupby.tail(1) は各グループ1行なので set_index 後の index は一意。
        last_rows = grp.tail(1)
        pa_result_s = last_rows.set_index("_pa_id")["_pa_res_str"]

        # 4球目（_pn==4）の S と、3球目（_pn==3）の post-S を pa_id ごとに引く。
        # ★ データ異常（_pn 欠損や同一打席内に複数 _pn==4）で重複が発生しうるため、
        #    groupby.first() で「各打席の最初の該当行」だけ採用する（一意保証）。
        df_pn4 = d_tmp[d_tmp["_pn"] == 4]
        if not df_pn4.empty:
            s_cnt_pn4_s = df_pn4.groupby("_pa_id", sort=False)["_s_cnt"].first()
        else:
            s_cnt_pn4_s = pd.Series(dtype=float)

        df_pn3 = d_tmp[d_tmp["_pn"] == 3]
        if not df_pn3.empty:
            post_s_pn3_s = df_pn3.groupby("_pa_id", sort=False)["_post_s_cnt"].first()
        else:
            post_s_pn3_s = pd.Series(dtype=float)

        # pa_id をキーに DataFrame を組み立てて条件評価
        per_pa = pd.DataFrame({"max_pn": max_pn_s})
        per_pa["pa_result"] = pa_result_s.reindex(per_pa.index).fillna("")
        per_pa["s_cnt_pn4"] = s_cnt_pn4_s.reindex(per_pa.index)
        per_pa["post_s_pn3"] = post_s_pn3_s.reindex(per_pa.index)

        # max_pn >= 3 のみ対象
        per_pa = per_pa[per_pa["max_pn"] >= 3]

        if not per_pa.empty:
            is_3 = per_pa["max_pn"] == 3
            is_4plus = per_pa["max_pn"] >= 4

            # max_pn==3 で「三振」を含む打席 → 状況発生+成功
            ko_at_3 = is_3 & per_pa["pa_result"].str.contains("三振", na=False)

            # max_pn>=4：4球目の S が 2 以上 → 成功（fallback: 3球目 post-S）
            s_val = per_pa["s_cnt_pn4"].copy()
            # 4球目の S が NaN なら 3球目 post-S で補う
            s_val = s_val.fillna(per_pa["post_s_pn3"])
            success_4plus = is_4plus & (s_val.fillna(-1) >= 2)

            # 集計
            three_count = int(ko_at_3.sum() + is_4plus.sum())
            three_success = int(ko_at_3.sum() + success_4plus.sum())

    three_pct = round(three_success / three_count * 100, 1) if three_count > 0 else 0.0

    return {
        "Strike%":       strike_pct,
        "1st-Strike%":   first_strike_pct,
        "Whiff数":       whiff_n,
        "Whiff%":        whiff_pct,
        "3球追い込み状況": three_count,
        "3球追い込み成功": three_success,
        "3球追い込み%":   three_pct,
    }


# ──────────────────────────────────────────
# 投手：球種別メトリクス
# ──────────────────────────────────────────
def calculate_pitchtype_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """投手の球種別メトリクス（参考ノートブック aggregate_metrics 準拠）。
    出力列: 投球数 / 投球割合 / 平均球速 / 球速比（％） / Zone% /
            Chase% / Whiff% / SwStr% / PutAway% / GB% / FB% / 被打率
    """
    if df.empty or "球種" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["球種"] = d["球種"].astype(str).replace({
        "チェンジアップ": "チェンジ", "縦スライダー": "縦スラ", "カットボール": "カット"
    })

    # 結果球 / pa_id 推定（最終球を pa の終端とみなす）
    is_final = (d.get("打席結果", pd.Series([None]*len(d))).notna()
                & ~d.get("打席結果", pd.Series(dtype=str)).astype(str).isin(["投球", "牽制"]))
    new_pa_marker = is_final.shift(1, fill_value=True)
    d["_pa_id"] = new_pa_marker.cumsum()
    final_pitches_df = d.loc[d.groupby("_pa_id").tail(1).index]

    # ストレートの平均球速（球速比の基準）
    ff_v = pd.to_numeric(d.loc[d["球種"] == "ストレート", "球速"], errors="coerce")
    ff_mean = ff_v[ff_v > 0].mean() if not ff_v[ff_v > 0].empty else 0.0

    total_p = len(d)

    rows = []
    for pt in d["球種"].dropna().unique():
        sub = d[d["球種"] == pt]
        n_pitches = len(sub)
        if n_pitches == 0:
            continue

        # 平均球速・最高球速
        velo = pd.to_numeric(sub.get("球速", pd.Series(dtype=float)), errors="coerce")
        v_pos = velo[velo > 0]
        mean_v = float(v_pos.mean()) if not v_pos.empty else float("nan")
        max_v  = float(v_pos.max())  if not v_pos.empty else float("nan")

        # Zone%
        in_zone = pd.to_numeric(sub.get("ストライクゾーン", pd.Series(dtype=float)),
                                errors="coerce").fillna(0).eq(1)
        zone_pct = float(in_zone.mean() * 100) if n_pitches else 0.0

        # Chase%
        valid_zone = {str(i) for i in range(1, 10)}
        pa_res = sub.get("打席結果", pd.Series(dtype=str)).astype(str)
        pr     = sub.get("投球結果", pd.Series(dtype=str)).astype(str)
        zone_str = sub.get("投球コース", pd.Series(dtype=str)).astype(str)
        cond_a = pa_res.isin(["空三振","安打","二塁打","三塁打","本塁打","エンタイトル"])
        cond_b = pr.isin(["空振り","ファウル"]) & (~zone_str.isin(valid_zone))
        chase_pct = float(((cond_a | cond_b).sum() / n_pitches) * 100)

        # Whiff%
        bq = sub.get("打球性質", pd.Series(dtype=str)).astype(str)
        whiffs = pr.eq("空振り")
        fouls  = pr.eq("ファウル")
        bip    = bq.isin(["ゴロ", "フライ", "ライナー"])
        swings = whiffs | fouls | bip
        n_swings = int(swings.sum())
        n_whiffs = int(whiffs.sum())
        whiff_pct = (n_whiffs / n_swings * 100) if n_swings > 0 else 0.0

        # SwStr%
        swstr_pct = (n_whiffs / n_pitches * 100) if n_pitches else 0.0

        # PutAway%
        sc = pd.to_numeric(sub.get("ストライクカウント", pd.Series(dtype=float)),
                           errors="coerce")
        two_strike = sub[sc == 2]
        n_2s = len(two_strike)
        n_ko = int(two_strike.get("打席結果", pd.Series(dtype=str))
                   .astype(str).isin(["空三振", "見三振"]).sum()) if n_2s else 0
        putaway_pct = (n_ko / n_2s * 100) if n_2s else 0.0

        # GB% / FB%
        denom = sub[bq.isin(["ゴロ","ライナー","フライ"]) & (pr != "ファウル")]
        if not denom.empty:
            denom_bq = denom["打球性質"].astype(str)
            gb_pct = float((denom_bq == "ゴロ").sum() / len(denom) * 100)
            fb_pct = float((denom_bq == "フライ").sum() / len(denom) * 100)
        else:
            gb_pct = 0.0
            fb_pct = 0.0

        # 被打率（最終球で集計）
        fp_pt = final_pitches_df[final_pitches_df["球種"] == pt]
        if fp_pt.empty:
            ba = 0.0
        else:
            fp_res = fp_pt["打席結果"].astype(str)
            hits = fp_res.str.contains("本|安|２|３", na=False).sum()
            ab   = fp_pt[fp_pt["打席結果"].notna()
                         & ~fp_res.str.contains("四球|死球|犠|失策", na=False)].shape[0]
            ba = (hits / ab) if ab else 0.0

        # 投球割合・球速比
        rate = round(n_pitches / total_p * 100, 1) if total_p else 0.0
        ratio = round(mean_v / ff_mean * 100, 1) if (np.isfinite(mean_v) and ff_mean) else float("nan")

        # 平均球速の表示（ストレートのみ最高球速併記）
        if not np.isfinite(mean_v):
            velo_text = "-"
        elif pt == "ストレート" and np.isfinite(max_v):
            velo_text = f"{mean_v:.1f} ({max_v:.1f})"
        else:
            velo_text = f"{mean_v:.1f}"

        rows.append({
            "球種":         pt,
            "投球数":       n_pitches,
            "投球割合":     rate,
            "平均球速":     velo_text,
            "球速比（％）": ratio,
            "Zone%":        round(zone_pct, 1),
            "Chase%":       round(chase_pct, 1),
            "Whiff%":       round(whiff_pct, 1),
            "SwStr%":       round(swstr_pct, 1),
            "PutAway%":     round(putaway_pct, 1),
            "GB%":          round(gb_pct, 1),
            "FB%":          round(fb_pct, 1),
            "被打率":       round(ba, 3),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # 並び順：投球割合の降順
    out = out.sort_values("投球割合", ascending=False).reset_index(drop=True)
    return out


def calculate_pitchtype_metrics_by_side(df: pd.DataFrame) -> dict:
    """対右打者・対左打者ぶんの球種別メトリクスを返す。
    return: {"R": DataFrame, "L": DataFrame}
    """
    out = {"R": pd.DataFrame(), "L": pd.DataFrame()}
    if df.empty or "打者打席左右" not in df.columns:
        return out
    side = df["打者打席左右"].astype(str)
    df_R = df[side.str.contains("右", na=False)]
    df_L = df[side.str.contains("左", na=False)]
    out["R"] = calculate_pitchtype_metrics(df_R)
    out["L"] = calculate_pitchtype_metrics(df_L)
    return out


def calculate_pitcher_overall_pitch_metrics(df: pd.DataFrame) -> dict:
    """投手の「全球種合計」レベルでの Zone% / Chase% / SwStr% / PutAway% /
    GB% / FB% を返す。指標順位検索（全球種モード）で使用。

    各定義は calculate_pitchtype_metrics の球種別ロジックを「全投球」に
    適用したもの。Whiff% は calculate_game_metrics 側で算出済みなのでここでは含めない。
    """
    if df.empty:
        return {
            "Zone%": 0.0, "Chase%": 0.0, "SwStr%": 0.0,
            "PutAway%": 0.0, "GB%": 0.0, "FB%": 0.0,
        }

    n_pitches = len(df)
    pr = df["投球結果"].astype(str) if "投球結果" in df.columns else pd.Series([""] * n_pitches)
    bq = df["打球性質"].astype(str) if "打球性質" in df.columns else pd.Series([""] * n_pitches)
    pa_res = df["打席結果"].astype(str) if "打席結果" in df.columns else pd.Series([""] * n_pitches)

    # Zone%
    in_zone = pd.to_numeric(df.get("ストライクゾーン", pd.Series(dtype=float)),
                            errors="coerce").fillna(0).eq(1)
    zone_pct = float(in_zone.mean() * 100) if n_pitches else 0.0

    # Chase%
    valid_zone = {str(i) for i in range(1, 10)}
    zone_str = df.get("投球コース", pd.Series(dtype=str)).astype(str)
    cond_a = pa_res.isin(["空三振", "安打", "二塁打", "三塁打", "本塁打", "エンタイトル"])
    cond_b = pr.isin(["空振り", "ファウル"]) & (~zone_str.isin(valid_zone))
    chase_pct = float(((cond_a | cond_b).sum() / n_pitches) * 100) if n_pitches else 0.0

    # SwStr%（空振り ÷ 全投球）
    n_whiffs = int(pr.eq("空振り").sum())
    swstr_pct = (n_whiffs / n_pitches * 100) if n_pitches else 0.0

    # PutAway%（2ストライクから三振で打席終了 ÷ 2ストライクからの投球）
    sc = pd.to_numeric(df.get("ストライクカウント", pd.Series(dtype=float)),
                       errors="coerce")
    two_strike = df[sc == 2]
    n_2s = len(two_strike)
    if n_2s:
        n_ko = int(two_strike.get("打席結果", pd.Series(dtype=str))
                   .astype(str).isin(["空三振", "見三振"]).sum())
    else:
        n_ko = 0
    putaway_pct = (n_ko / n_2s * 100) if n_2s else 0.0

    # GB% / FB%（ファウル以外のフェア打球が分母）
    denom = df[bq.isin(["ゴロ", "ライナー", "フライ"]) & (pr != "ファウル")]
    if not denom.empty:
        denom_bq = denom["打球性質"].astype(str)
        gb_pct = float((denom_bq == "ゴロ").sum() / len(denom) * 100)
        fb_pct = float((denom_bq == "フライ").sum() / len(denom) * 100)
    else:
        gb_pct = 0.0
        fb_pct = 0.0

    return {
        "Zone%":    round(zone_pct, 1),
        "Chase%":   round(chase_pct, 1),
        "SwStr%":   round(swstr_pct, 1),
        "PutAway%": round(putaway_pct, 1),
        "GB%":      round(gb_pct, 1),
        "FB%":      round(fb_pct, 1),
    }


# ──────────────────────────────────────────
# 打者：球種別セイバー（参考ノートブック準拠）
# ──────────────────────────────────────────
def calculate_batter_pitchtype_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """打者の球種別セイバー。
    列: 球種 / 球数 / 打数 / 安打 / 打率 / 長打率 /
        Whiff% / Swing% / Contact% / O-Swing% / Z-Contact%
    """
    if df.empty or "球種" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    # is_pitch 互換：投球結果が「投球」or 球種が記録されているものを 1 とみなす
    if "is_pitch" not in d.columns:
        d["is_pitch"] = d["球種"].notna()

    pa_res = d.get("打席結果", pd.Series(dtype=str)).astype(str)
    pr     = d.get("投球結果", pd.Series(dtype=str)).astype(str)
    bq     = d.get("打球性質", pd.Series(dtype=str)).astype(str)
    in_zone = pd.to_numeric(d.get("ストライクゾーン", pd.Series(dtype=float)),
                            errors="coerce").fillna(0).eq(1)

    rows = []
    pitch_types = (d.loc[d["is_pitch"] == True, "球種"]
                    .dropna().unique())

    for pt in pitch_types:
        m_pt = (d["is_pitch"] == True) & (d["球種"] == pt)
        n_p = int(m_pt.sum())
        if n_p == 0:
            continue
        sub = d[m_pt]
        sub_pr   = pr.loc[m_pt]
        sub_bq   = bq.loc[m_pt]
        sub_zone = in_zone.loc[m_pt]

        # 打席終了行から集計（打数・安打・打率・長打率）
        pa_end = sub[sub["打席結果"].notna()].copy()
        pa_res_e = pa_end.get("打席結果", pd.Series(dtype=str)).astype(str)
        ab = pa_end[~pa_res_e.str.contains("四球|死球|犠|失策", na=False)].shape[0]
        hits = int(pa_res_e.str.contains("本|安|２|３", na=False).sum())
        singles = int((pa_res_e.str.contains("安", na=False)
                       & ~pa_res_e.str.contains("２|３|本", na=False)).sum())
        doubles = int(pa_res_e.str.contains("２", na=False).sum())
        triples = int(pa_res_e.str.contains("３", na=False).sum())
        hrs     = int(pa_res_e.str.contains("本", na=False).sum())
        ba   = (hits / ab) if ab else 0.0
        tb = singles + doubles * 2 + triples * 3 + hrs * 4
        slg  = (tb / ab) if ab else 0.0

        # スイング系
        whiffs = sub_pr.eq("空振り")
        fouls  = sub_pr.eq("ファウル")
        bip    = sub_bq.isin(["ゴロ", "ライナー", "フライ"])
        swings = whiffs | fouls | bip
        n_sw   = int(swings.sum())
        n_wh   = int(whiffs.sum())
        n_in   = int(sub_zone.sum())
        n_out  = int((~sub_zone).sum())

        # O-Swing% : ボールゾーンへのスイング / ボール球
        o_sw_n = int((swings & ~sub_zone).sum())
        # Z-Contact% : ストライクゾーンでの空振り以外（コンタクト）/ ストライクゾーンスイング
        z_sw_n   = int((swings & sub_zone).sum())
        z_cont_n = int(((fouls | bip) & sub_zone).sum())

        swing_pct   = (n_sw / n_p * 100) if n_p else 0.0
        whiff_pct   = (n_wh / n_sw * 100) if n_sw else 0.0
        contact_pct = ((n_sw - n_wh) / n_sw * 100) if n_sw else 0.0
        o_swing_pct = (o_sw_n / n_out * 100) if n_out else 0.0
        z_contact_pct = (z_cont_n / z_sw_n * 100) if z_sw_n else 0.0

        rows.append({
            "球種":      str(pt),
            "球数":      n_p,
            "打数":      ab,
            "安打":      hits,
            "打率":      round(ba, 3),
            "長打率":    round(slg, 3),
            "Whiff%":    round(whiff_pct, 1),
            "Swing%":    round(swing_pct, 1),
            "Contact%":  round(contact_pct, 1),
            "O-Swing%":  round(o_swing_pct, 1),
            "Z-Contact%": round(z_contact_pct, 1),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # 並び順：ストレート優先、その後 球数降順
    out["_pri"] = (out["球種"] != "ストレート").astype(int)
    out = (out.sort_values(["_pri", "球数"], ascending=[True, False])
              .drop(columns=["_pri"]).reset_index(drop=True))
    return out


# ──────────────────────────────────────────
# 打者：初球（1stストライク時）Swing率
# ──────────────────────────────────────────
def calculate_first_strike_swing_rate(df: pd.DataFrame) -> dict:
    """カウント0-0時のスイング率を、全球種／ストレート／変化球／カーブ別に。"""
    if df.empty:
        return {"全打席": 0.0, "ストレート": 0.0, "変化球": 0.0, "カーブ": 0.0}

    d = df.copy()
    if "is_pitch" not in d.columns:
        d["is_pitch"] = d["球種"].notna() if "球種" in d.columns else True

    sc = pd.to_numeric(d.get("ストライクカウント", pd.Series(dtype=float)),
                       errors="coerce")
    first = d[(sc == 0) & (d["is_pitch"] == True)].copy()
    if first.empty:
        return {"全打席": 0.0, "ストレート": 0.0, "変化球": 0.0, "カーブ": 0.0}

    breaking = ["ツーシーム", "カットボール", "スライダー", "縦スライダー",
                "カーブ", "チェンジアップ", "スプリット", "シンカー", "特殊球",
                "カット", "縦スラ", "チェンジ"]

    def _swing_rate(data: pd.DataFrame) -> float:
        if data.empty:
            return 0.0
        total = int(data["is_pitch"].sum())
        if total == 0:
            return 0.0
        bip = data.get("打球性質", pd.Series(dtype=str)).astype(str).isin(
            ["ゴロ", "ライナー", "フライ"])
        whiff = data.get("投球結果", pd.Series(dtype=str)).astype(str) == "空振り"
        foul = data.get("投球結果", pd.Series(dtype=str)).astype(str) == "ファウル"
        n_sw = int((bip | whiff | foul).sum())
        return float(n_sw / total * 100)

    return {
        "全打席":    round(_swing_rate(first), 1),
        "ストレート": round(_swing_rate(
            first[first["球種"].astype(str) == "ストレート"]), 1),
        "変化球":    round(_swing_rate(
            first[first["球種"].astype(str).isin(breaking)]), 1),
        "カーブ":    round(_swing_rate(
            first[first["球種"].astype(str) == "カーブ"]), 1),
    }


# ──────────────────────────────────────────
# 打者：2ストライク到達後の平均投球数 / ファウル数 / O-Swing%
# ──────────────────────────────────────────
def calculate_two_strike_metrics(df: pd.DataFrame) -> dict:
    """2ストライク到達後の平均投球数、平均ファウル数、O-Swing%。
    ベクトル化版（旧 groupby ループ版より高速）。"""
    default = {"2s後の平均投球数": 0.0, "2s後の平均ファウル数": 0.0,
               "2s後のO-Swing%": 0.0}
    if df.empty:
        return default
    if "ストライクカウント" not in df.columns or "球数" not in df.columns \
       or "投球結果" not in df.columns:
        return default

    work = df.copy()
    if "is_pitch" not in work.columns:
        work["is_pitch"] = work["球種"].notna() if "球種" in work.columns else True

    # 結果球フラグの推定（ベクトル化）
    if "結果球" in work.columns:
        end_flag = pd.to_numeric(work["結果球"], errors="coerce").fillna(0).astype(int).eq(1)
    elif "打席結果" in work.columns:
        end_flag = work["打席結果"].notna()
    else:
        return default

    # game_id 列を作成
    if "game_id" not in work.columns:
        if "日付" in work.columns:
            work["game_id"] = work["日付"].astype(str).fillna("")
        else:
            work["game_id"] = ""

    # 「打席結果」がある行も結果球扱い
    if "打席結果" in work.columns:
        end_flag = end_flag | work["打席結果"].notna()

    # pa_id を作成（ゲームごとに累積カウント、結果球で 1 増える前にカウント）
    # → end_flag をシフト累積
    work["__end"] = end_flag.astype(int)
    # cumsum を game_id ごと、end_flag をシフトして「結果球の次の行」から新打席を開始
    pa_id = work.groupby("game_id")["__end"].cumsum() - work["__end"]
    work["_pa_id"] = pa_id.astype(int)

    # 投球行のみ抽出
    work = work[work["is_pitch"] == True].copy()
    if work.empty:
        return default

    work["球数"] = pd.to_numeric(work["球数"], errors="coerce")
    work["ストライクカウント"] = pd.to_numeric(work["ストライクカウント"], errors="coerce")

    bq = work.get("打球性質", pd.Series(dtype=str)).astype(str)
    work["__bip"]   = bq.isin(["ゴロ", "ライナー", "フライ"])
    work["__whiff"] = work["投球結果"] == "空振り"
    work["__foul"]  = work["投球結果"].isin(["ファウル", "ファール"])
    work["__swing"] = work["__bip"] | work["__whiff"] | work["__foul"]

    if "投球コース" in work.columns:
        work["投球コース"] = pd.to_numeric(work["投球コース"], errors="coerce")
        work["__o_zone"] = work["投球コース"].between(10, 25, inclusive="both")
    else:
        work["__o_zone"] = False

    # ★ ベクトル化集計：「打席ごとに最初の2ストライク球数」と「最終球数」を groupby agg で算出
    pa_keys = ["game_id", "_pa_id"]

    # 各打席の最終球数（結果球が立っている行の中の最後の球数 - ただし簡易に max を使用）
    g = work.groupby(pa_keys, sort=False)

    # 各打席の最終結果球の球数
    is_end_pitch = work["__end"] == 1
    end_balls = (work.loc[is_end_pitch].groupby(pa_keys, sort=False)["球数"]
                 .last())

    # 各打席で初めて 2 ストライクになった行の球数
    is_2s = work["ストライクカウント"] == 2
    two_s_balls = (work.loc[is_2s].groupby(pa_keys, sort=False)["球数"]
                   .first())

    # 共通の打席IDで join
    common_idx = end_balls.index.intersection(two_s_balls.index)
    if len(common_idx) == 0:
        return default
    e = end_balls.loc[common_idx]
    s = two_s_balls.loc[common_idx]

    valid = e.notna() & s.notna() & (s <= e)
    e = e[valid]
    s = s[valid]
    if e.empty:
        return default

    pitches_after = (e - s + 1).astype(float)
    avg_pitches = float(pitches_after.mean())

    # 2ストライク到達時の各打席のファウル数
    fouls_per_pa = (work.loc[(work["ストライクカウント"] == 2) & work["__foul"]]
                    .groupby(pa_keys, sort=False).size())
    fouls_per_pa = fouls_per_pa.reindex(common_idx, fill_value=0)[valid]
    avg_fouls = float(fouls_per_pa.mean()) if not fouls_per_pa.empty else 0.0

    # O-Swing% : 2S到達球以降のボールゾーンでの swing 集計
    # 各行に「自分の打席の2S球数」を join し、 球数 >= 2S 球数 でフィルタ
    s_full = two_s_balls.reindex(work.set_index(pa_keys).index, fill_value=np.nan)
    work["_start"] = s_full.values
    after = work[(work["_start"].notna()) & (work["球数"] >= work["_start"]) &
                 (work["球数"] <= work.set_index(pa_keys).join(end_balls.rename("__end_b"))["__end_b"].reindex(work.set_index(pa_keys).index).values)]
    # 上の式は複雑なので、簡素化する：単純に 2S 以降のレコードを取得
    after = work.copy()
    after["_start"] = s_full.values
    # 各レコードの所属打席の終了球数も同様に
    e_full = end_balls.reindex(after.set_index(pa_keys).index, fill_value=np.nan)
    after["_end"] = e_full.values
    after = after[after["_start"].notna() & after["_end"].notna() &
                  (after["球数"] >= after["_start"]) &
                  (after["球数"] <= after["_end"])]

    o_pitches = int(after["__o_zone"].sum())
    o_swings  = int((after["__o_zone"] & after["__swing"]).sum())
    o_swing_pct = (o_swings / o_pitches * 100) if o_pitches > 0 else 0.0

    return {
        "2s後の平均投球数": round(avg_pitches, 2),
        "2s後の平均ファウル数": round(avg_fouls, 2),
        "2s後のO-Swing%": round(o_swing_pct, 1),
    }


# ──────────────────────────────────────────
# 打者：基本成績
# ──────────────────────────────────────────
def calculate_baseball_stats(df: pd.DataFrame, label: str = "全体") -> dict:
    if df.empty:
        return {"状況": label}
    res = df["打席結果"].astype(str) if "打席結果" in df.columns else pd.Series(dtype=str)
    ab   = int(df[~res.str.contains(r"四球|死球|犠|失策", na=False, regex=True)].shape[0])
    h    = int(res.str.contains(r"安|２|３|本", na=False, regex=True).sum())
    s1   = int((res.str.contains("安", na=False) & ~res.str.contains(r"２|３|本", na=False, regex=True)).sum())
    d2   = int(res.str.contains("２", na=False).sum())
    t3   = int(res.str.contains("３", na=False).sum())
    hr   = int(res.str.contains("本", na=False).sum())
    bb_hbp = int(res.str.contains("球", na=False).sum())
    so   = int(res.str.contains("三振", na=False).sum())
    sac_b = int(res.str.contains(r"犠打|犠バント", na=False, regex=True).sum())
    sac_f = int(res.str.contains(r"犠飛|犠牲飛", na=False, regex=True).sum())
    rbi  = int(pd.to_numeric(df["打点数"], errors="coerce").fillna(0).sum()) if "打点数" in df.columns else 0
    pa   = ab + bb_hbp + sac_b + sac_f

    ba   = h / ab if ab > 0 else 0.0
    obp_den = ab + bb_hbp + sac_f
    obp  = (h + bb_hbp) / obp_den if obp_den > 0 else 0.0
    tb   = s1 + d2*2 + t3*3 + hr*4
    slg  = tb / ab if ab > 0 else 0.0
    ops  = obp + slg
    iso  = slg - ba

    walks = int(res.str.contains("四球", na=False).sum())
    hbps  = int(res.str.contains("死球", na=False).sum())
    ibbs  = int(res.str.contains(r"敬遠|故意四球", na=False, regex=True).sum())
    woba_den = ab + walks - ibbs + sac_f + hbps
    woba = 0.0
    if woba_den > 0:
        w = WOBA_WEIGHTS
        woba = (w["BB"]*(walks-ibbs) + w["HBP"]*hbps +
                w["1B"]*s1 + w["2B"]*d2 + w["3B"]*t3 + w["HR"]*hr) / woba_den

    return {
        "状況": label, "打席": pa, "打数": ab, "安打": h,
        "二塁打": d2, "三塁打": t3, "本塁打": hr, "四死球": bb_hbp,
        "三振": so, "打点": rbi,
        "打率": round(ba, 3), "出塁率": round(obp, 3), "長打率": round(slg, 3),
        "OPS": round(ops, 3), "ISO": round(iso, 3), "wOBA": round(woba, 3),
    }


# ──────────────────────────────────────────
# 打者：セイバーメトリクス
# ──────────────────────────────────────────
def calculate_sabermetrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    pa_df   = df[df["打席結果"].notna()] if "打席結果" in df.columns else df
    pitch_df = df[df["is_pitch"] == True] if "is_pitch" in df.columns else df
    n_pa    = max(len(pa_df), 1)
    n_pitch = max(len(pitch_df), 1)

    res = pa_df["打席結果"].astype(str) if "打席結果" in pa_df.columns else pd.Series(dtype=str)
    pr  = pitch_df["投球結果"].astype(str) if "投球結果" in pitch_df.columns else pd.Series(dtype=str)
    bq  = pitch_df["打球性質"].astype(str) if "打球性質" in pitch_df.columns else pd.Series(dtype=str)
    course = pd.to_numeric(pitch_df["投球コース"], errors="coerce") if "投球コース" in pitch_df.columns else pd.Series(dtype=float)

    so   = int(res.isin(["空三振", "見三振", "逃三振"]).sum())
    walks= int(res.str.contains("四球", na=False).sum())
    bip  = bq.isin(["ゴロ", "ライナー", "フライ"])
    whiff= pr.eq("空振り")
    swing= bip | whiff
    n_sw = int(swing.sum())
    n_wh = int(whiff.sum())

    in_zone  = course.between(1, 9)
    out_zone = course.between(10, 25)
    z_sw = int((swing & in_zone).sum())
    o_sw = int((swing & out_zone).sum())
    z_tot= int(in_zone.sum())
    o_tot= int(out_zone.sum())
    z_bip= int((bip & in_zone).sum())
    o_bip= int((bip & out_zone).sum())

    # Pull-air% — ベクトル化版
    # 分母：結果球==1 の打球（邪飛除く・x/y座標あり）= 全結果球数
    # 分子：上記のうち「フライ or ライナー」かつ引っ張り方向角度
    pull_air = 0.0
    if "打球位置x座標" in df.columns and "打球位置y座標" in df.columns and "結果球" in df.columns:
        vd = df[(df["結果球"] == 1) & df["打球位置x座標"].notna() & df["打球位置y座標"].notna()].copy()
        if "打席結果" in vd.columns:
            vd = vd[~vd["打席結果"].astype(str).str.contains("邪飛", na=False)]
        vd["_x"] = pd.to_numeric(vd["打球位置x座標"], errors="coerce")
        vd["_y"] = pd.to_numeric(vd["打球位置y座標"], errors="coerce")
        vd = vd.dropna(subset=["_x", "_y"])
        denom = len(vd)  # ★ 分母：全結果球数（邪飛除く）
        if denom > 0:
            bq2 = vd["打球性質"].astype(str) if "打球性質" in vd.columns else pd.Series(dtype=str)
            fl_li = bq2.isin(["ライナー", "フライ"])
            sub = vd[fl_li]
            if not sub.empty:
                ang = np.degrees(np.arctan2(sub["_y"].values, sub["_x"].values))
                side = sub.get("打者打席左右", pd.Series(dtype=str)).astype(str).values
                # 右打者: 60〜90度、左打者: 0〜30度
                pull_mask = (((side == "右") & (ang >= 60) & (ang <= 90)) |
                             ((side == "左") & (ang >= 0)  & (ang <= 30)))
                num = int(pull_mask.sum())
                pull_air = round(num / denom * 100, 1)

    return {
        "K%":        round(so    / n_pa * 100, 1),
        "BB%":       round(walks / n_pa * 100, 1),
        "Swing%":    round(n_sw  / n_pitch * 100, 1),
        "Whiff%":    round(n_wh  / n_sw * 100, 1) if n_sw > 0 else 0.0,
        "SwStr%":    round(n_wh  / n_pitch * 100, 1),
        "Contact%":  round(int(bip.sum()) / n_sw * 100, 1) if n_sw > 0 else 0.0,
        "Z-Swing%":  round(z_sw / z_tot * 100, 1) if z_tot > 0 else 0.0,
        "O-Swing%":  round(o_sw / o_tot * 100, 1) if o_tot > 0 else 0.0,
        "Z-Contact%":round(z_bip / z_sw * 100, 1) if z_sw > 0 else 0.0,
        "O-Contact%":round(o_bip / o_sw * 100, 1) if o_sw > 0 else 0.0,
        "Pull-air%": pull_air,
    }
