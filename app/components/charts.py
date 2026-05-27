# -*- coding: utf-8 -*-
"""共通可視化コンポーネント"""
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import streamlit as st
from ..stats.constants import PITCH_COLORS
from ..stats.calculations import _pick_col


# ──────────────────────────────────────────
# Plotly のグローバル既定スタイル（ライトモード強制）
# ★ plotly_white の参照を直接書き換えないよう、コピーを取って独自テンプレートを構築。
# ──────────────────────────────────────────
import copy as _copy
_light_template = _copy.deepcopy(pio.templates["plotly_white"])
_light_template.layout.paper_bgcolor = "#FFFFFF"
_light_template.layout.plot_bgcolor = "#FFFFFF"
_light_template.layout.font = dict(
    family=('-apple-system, BlinkMacSystemFont, "Segoe UI", '
            '"Helvetica Neue", "Hiragino Sans", sans-serif'),
    color="#1F2A37",
    size=12,
)
_light_template.layout.legend = dict(
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="#D6DEE5",
    font=dict(color="#1F2A37", size=11),
)
_light_template.layout.colorway = [
    "#2C5F7C", "#E55A4C", "#F5A623", "#4A9B8E",
    "#9B59B6", "#3498DB", "#E74C3C", "#1ABC9C",
]
# ★ xaxis/yaxis はテンプレートで上書きしない（チャート個別の range が壊れるため）
pio.templates["enemil_light"] = _light_template
pio.templates.default = "enemil_light"


# ──────────────────────────────────────────
# 共通ユーティリティ
# ──────────────────────────────────────────
def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _darker(hex_color: str, factor: float = 0.5):
    """色を暗くした hex 文字列を返す"""
    r, g, b = _hex_to_rgb(hex_color)
    r2 = max(0, int(r * factor))
    g2 = max(0, int(g * factor))
    b2 = max(0, int(b * factor))
    return f"#{r2:02x}{g2:02x}{b2:02x}"


# ──────────────────────────────────────────
# ホームベース五角形（捕手/投手視点の上下反転対応）
# ──────────────────────────────────────────
def _homeplate_path(view: str = "catcher",
                    x_center: float = 150.0,
                    plate_top_y: float = 25.0,
                    plate_bot_y: float = 5.0,
                    side_y: float = 15.0,
                    half_w: float = 90.0):
    """
    ホームベース五角形の SVG path を返す。
    プロットの下端（y=0〜30 付近）にホームベースを配置することを想定。
    view が "pitcher" の場合は上下反転（先端が上を向く）して描画。
    座標系: x=0..300, y=0..300（ストライクゾーンは x=60-240, y=60-240）。
    """
    if view == "catcher":
        # 捕手視点：頂点（先端）は下、平らな面は上
        pts = [
            (x_center - half_w, plate_top_y),  # 投手側・左
            (x_center + half_w, plate_top_y),  # 投手側・右
            (x_center + half_w, side_y),       # サイド・右
            (x_center,          plate_bot_y),  # 先端（下）
            (x_center - half_w, side_y),       # サイド・左
        ]
    else:
        # 投手視点：頂点を上に反転（先端が上、平らな面が下）
        flip = lambda y: plate_top_y + plate_bot_y - y  # 上下反転
        pts = [
            (x_center - half_w, flip(plate_top_y)),
            (x_center + half_w, flip(plate_top_y)),
            (x_center + half_w, flip(side_y)),
            (x_center,          flip(plate_bot_y)),
            (x_center - half_w, flip(side_y)),
        ]
    return "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"


# ──────────────────────────────────────────
# 投球コースヒートマップ（球種別 KDE）
# ──────────────────────────────────────────
# KDE 計算結果をプロセス内でキャッシュ（scipy gaussian_kde は ~50-150ms と重い）
# x, y は同一投手・同一球種・同一期間ならまったく同じ値が来るので、
# ndarray の bytes をキーにすれば確実にヒットする
from functools import lru_cache

@lru_cache(maxsize=512)
def _compute_kde_grid_cached(x_bytes: bytes, y_bytes: bytes, n: int,
                              grid_n: int, x_min: float, x_max: float,
                              y_min: float, y_max: float,
                              bw_method: float, thresh: float):
    """KDE のキャッシュ実装。tuple 化された引数で LRU cache が効く。"""
    from scipy.stats import gaussian_kde
    x = np.frombuffer(x_bytes, dtype=np.float64)
    y = np.frombuffer(y_bytes, dtype=np.float64)
    if len(x) < 3:
        return None
    try:
        values = np.vstack([x, y])
        kde = gaussian_kde(values, bw_method=bw_method)
    except Exception:
        return None
    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    z = kde(positions).reshape(xx.shape)
    z_max = z.max() if z.size else 0
    if z_max > 0:
        z = np.where(z >= thresh * z_max, z, np.nan)
    return xs, ys, z


def _compute_kde_grid(x: np.ndarray, y: np.ndarray,
                      grid_n: int = 80,
                      x_range=(0, 300), y_range=(0, 300),
                      bw_method: float = 0.4,
                      thresh: float = 0.02):
    """
    scipy gaussian_kde で 2D 密度を計算して (xs, ys, z) を返す。
    z は thresh 未満を NaN にマスクして薄い領域を非表示にする。
    LRU キャッシュ経由で、同じ x/y データなら計算をskip。
    """
    if len(x) < 3:
        return None
    # ndarray を contiguous float64 にして bytes 化
    xa = np.ascontiguousarray(x, dtype=np.float64)
    ya = np.ascontiguousarray(y, dtype=np.float64)
    return _compute_kde_grid_cached(
        xa.tobytes(), ya.tobytes(), len(xa),
        grid_n, float(x_range[0]), float(x_range[1]),
        float(y_range[0]), float(y_range[1]),
        float(bw_method), float(thresh),
    )


def _draw_pitch_kde_fig(p_df: pd.DataFrame, pitch_type: str,
                        view: str = "catcher",
                        title_suffix: str = "",
                        height: int = 320):
    """
    1 球種ぶんの KDE ヒートマップ figure を返す。投球位置x座標/y座標 を使用。
    color: white → pitch_color → dark(pitch_color)
    view: "catcher"（x軸反転＝捕手目線）/ "pitcher"（x軸通常＝投手目線）
    """
    x = pd.to_numeric(p_df["投球位置x座標"], errors="coerce")
    y = pd.to_numeric(p_df["投球位置y座標"], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()
    n = len(x)

    base_color = PITCH_COLORS.get(pitch_type, "#FF0000")
    dark = _darker(base_color, 0.45)
    colorscale = [
        [0.00, "#FFFFFF"],
        [0.45, base_color],
        [1.00, dark],
    ]

    fig = go.Figure()

    # KDE 計算（試行）
    grid = _compute_kde_grid(x, y) if n >= 3 else None
    if grid is not None:
        xs, ys, z = grid
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=z,
            colorscale=colorscale,
            showscale=False,
            zsmooth="best",
            hoverinfo="skip",
        ))
    elif n > 0:
        # データが少なすぎる場合は散布図でフォールバック
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(color=base_color, size=10, opacity=0.6),
            hoverinfo="skip", showlegend=False,
        ))

    # ストライクゾーン枠（60-240 の正方形）
    fig.add_shape(
        type="rect", x0=60, x1=240, y0=60, y1=240,
        line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    # ホームベース
    fig.add_shape(
        type="path", path=_homeplate_path(view=view),
        line=dict(color="black", width=2),
        fillcolor="rgba(255,255,255,0.0)",
        layer="above",
    )

    # 軸設定：捕手視点はxを反転（右打者の内角が画面右）、投手視点はそのまま
    x_axis_kwargs = dict(
        range=[300, 0] if view == "catcher" else [0, 300],
        showgrid=False, zeroline=False, showticklabels=False,
        scaleanchor="y", scaleratio=1,
    )
    y_axis_kwargs = dict(
        range=[0, 300],
        showgrid=False, zeroline=False, showticklabels=False,
    )

    title = f"{pitch_type} (N={n})"
    if title_suffix:
        title = f"{pitch_type}  {title_suffix} (N={n})"

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=x_axis_kwargs, yaxis=y_axis_kwargs,
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
    )
    return fig


def draw_zone_heatmap(df: pd.DataFrame, title: str = "投球コース",
                      view: str = "catcher", top_n: int = 4,
                      key_prefix: str = "hm"):
    """
    球種別の投球コースヒートマップを上位 top_n 球種ぶん横並びで描画。
    投球位置x座標/y座標 を用いた KDE 表示。
    ★ 単一の Plotly subplots として描画することで、スマホでも自動的に
       縮小されつつ 4 列レイアウトが維持される。
    """
    from plotly.subplots import make_subplots
    if "球種" not in df.columns:
        st.info("球種データがありません")
        return
    if "投球位置x座標" not in df.columns or "投球位置y座標" not in df.columns:
        st.info("投球位置座標データがありません")
        return

    d = df.dropna(subset=["投球位置x座標", "投球位置y座標", "球種"]).copy()
    if d.empty:
        st.info(f"{title}: 投球データがありません")
        return

    counts = d["球種"].value_counts()
    ordered = []
    if "ストレート" in counts.index:
        ordered.append("ストレート")
    for pt in counts.index:
        if pt not in ordered:
            ordered.append(pt)
    show_types = ordered[:top_n]
    n_sub = len(show_types)
    if n_sub == 0:
        st.info(f"{title}: 球種が見つかりません")
        return

    st.markdown(f"**{title}（{'捕手目線' if view=='catcher' else '投手目線'}）**")

    # subplot タイトルは球種名 + (N=...) を表示
    subplot_titles = []
    for pt in show_types:
        n = int((d["球種"] == pt).sum())
        subplot_titles.append(f"{pt} (N={n})")

    fig = make_subplots(
        rows=1, cols=n_sub,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        shared_yaxes=False,
    )

    for i, pt in enumerate(show_types):
        sub = d[d["球種"] == pt]
        x = pd.to_numeric(sub["投球位置x座標"], errors="coerce")
        y = pd.to_numeric(sub["投球位置y座標"], errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask].to_numpy()
        y = y[mask].to_numpy()
        n = len(x)

        base_color = PITCH_COLORS.get(pt, "#FF0000")
        dark = _darker(base_color, 0.45)
        colorscale = [
            [0.00, "#FFFFFF"],
            [0.45, base_color],
            [1.00, dark],
        ]

        col_idx = i + 1

        # KDE またはスキャッタ
        grid = _compute_kde_grid(x, y) if n >= 3 else None
        if grid is not None:
            xs, ys, z = grid
            fig.add_trace(
                go.Heatmap(
                    x=xs, y=ys, z=z,
                    colorscale=colorscale,
                    showscale=False,
                    zsmooth="best",
                    hoverinfo="skip",
                ),
                row=1, col=col_idx,
            )
        elif n > 0:
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="markers",
                    marker=dict(color=base_color, size=10, opacity=0.6),
                    hoverinfo="skip", showlegend=False,
                ),
                row=1, col=col_idx,
            )

        # ストライクゾーン枠
        fig.add_shape(
            type="rect", x0=60, x1=240, y0=60, y1=240,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
            row=1, col=col_idx,
        )
        # ホームベース
        fig.add_shape(
            type="path", path=_homeplate_path(view=view),
            line=dict(color="black", width=2),
            fillcolor="rgba(255,255,255,0.0)",
            layer="above",
            row=1, col=col_idx,
        )

        # 各 subplot の軸設定
        fig.update_xaxes(
            range=[300, 0] if view == "catcher" else [0, 300],
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor=f"y{col_idx}" if col_idx > 1 else "y",
            scaleratio=1,
            row=1, col=col_idx,
        )
        fig.update_yaxes(
            range=[0, 300],
            showgrid=False, zeroline=False, showticklabels=False,
            row=1, col=col_idx,
        )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
    )
    # subplot タイトルのスタイル統一
    for ann in fig.layout.annotations:
        if ann.text and "(N=" in ann.text:
            ann.font = dict(size=12, color="#1F2A37")

    st.plotly_chart(fig, use_container_width=True,
                    key=f"{key_prefix}_{view}")


def draw_zone_heatmap_by_side(df: pd.DataFrame, title: str = "投球コース",
                              view: str = "catcher", top_n: int = 4,
                              key_prefix: str = "hm"):
    """
    対右打者・対左打者ぶんを横並びの2セクションで描画。
    key_prefix を変えれば同一ページ内で複数回呼んでも衝突しない。
    """
    side_col = "打者打席左右" if "打者打席左右" in df.columns else None
    if not side_col:
        draw_zone_heatmap(df, title=title, view=view, top_n=top_n,
                          key_prefix=f"{key_prefix}_all")
        return

    df_R = df[df[side_col].astype(str).str.contains("右", na=False)]
    df_L = df[df[side_col].astype(str).str.contains("左", na=False)]

    st.markdown(f"### {title}（{'捕手目線' if view=='catcher' else '投手目線'}）")
    st.markdown("#### 対右打者")
    if df_R.empty:
        st.caption("対右打者のデータがありません")
    else:
        draw_zone_heatmap(df_R, title="", view=view, top_n=top_n,
                          key_prefix=f"{key_prefix}_R")
    st.markdown("#### 対左打者")
    if df_L.empty:
        st.caption("対左打者のデータがありません")
    else:
        draw_zone_heatmap(df_L, title="", view=view, top_n=top_n,
                          key_prefix=f"{key_prefix}_L")


# ──────────────────────────────────────────
# 球速ヒストグラム
# ──────────────────────────────────────────
def draw_velocity_histogram(df: pd.DataFrame, pitcher_name: str = "",
                            key: str = "velo_hist"):
    if "球種" not in df.columns or "球速" not in df.columns:
        st.info("球速データがありません")
        return
    d = df.copy()
    d["球種"] = d["球種"].astype(str).replace({
        "チェンジアップ": "チェンジ", "縦スライダー": "縦スラ", "カットボール": "カット"
    })
    d["球速_n"] = pd.to_numeric(d["球速"], errors="coerce")
    d = d.dropna(subset=["球速_n"])
    d = d[d["球速_n"].between(90, 165)]

    fig = go.Figure()
    for pt in d["球種"].unique():
        sub = d[d["球種"] == pt]["球速_n"]
        color = PITCH_COLORS.get(pt, "#888888")
        fig.add_trace(go.Histogram(
            x=sub, name=pt,
            xbins=dict(start=100, end=160, size=1),
            marker_color=color, opacity=0.8,
        ))
        if len(sub) > 0:
            avg = sub.mean()
            fig.add_vline(x=avg, line_dash="dash", line_color=color,
                          annotation_text=f"{pt} {avg:.1f}",
                          annotation_position="top")

    fig.update_layout(
        barmode="stack", title=f"{pitcher_name} 球速分布",
        xaxis_title="球速 (km/h)", yaxis_title="投球数",
        xaxis=dict(range=[100, 160]),
        height=320, margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ──────────────────────────────────────────
# 球種ドーナツチャート（単一・分割）
# ──────────────────────────────────────────
def draw_pitch_donut(df: pd.DataFrame, title: str = "球種割合",
                     key: str = "donut"):
    if "球種" not in df.columns or df.empty:
        st.info(f"{title}: 球種データがありません")
        return
    d = df.copy()
    d["球種"] = d["球種"].astype(str).replace({
        "チェンジアップ": "チェンジ", "縦スライダー": "縦スラ", "カットボール": "カット"
    })
    counts = d["球種"].value_counts()
    if counts.empty:
        st.info(f"{title}: 球種データがありません")
        return
    colors = [PITCH_COLORS.get(k, "#888888") for k in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.45, marker_colors=colors,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=title, height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def draw_pitch_donut_split(df: pd.DataFrame, key_prefix: str = "donut"):
    """対右・対左で2つのドーナツを横並びに表示。
    key_prefix を変えれば同一ページ内で複数回呼んでも衝突しない。"""
    side_col = "打者打席左右" if "打者打席左右" in df.columns else None
    col1, col2 = st.columns(2)
    if side_col:
        df_R = df[df[side_col].astype(str).str.contains("右", na=False)]
        df_L = df[df[side_col].astype(str).str.contains("左", na=False)]
    else:
        df_R = df
        df_L = df.iloc[0:0]
    with col1:
        draw_pitch_donut(df_R, title="球種割合（対右打者）",
                         key=f"{key_prefix}_R")
    with col2:
        draw_pitch_donut(df_L, title="球種割合（対左打者）",
                         key=f"{key_prefix}_L")


# ──────────────────────────────────────────
# カウント別投球割合（3行×4列、12個のパイチャート）
# ──────────────────────────────────────────
def draw_count_based_pitch_ratio(df: pd.DataFrame, title: str = "",
                                 key_prefix: str = "cnt_pitch",
                                 height: int = 540):
    """ボールカウント・ストライクカウントの組合せ12種類（0-0〜3-2）について、
    それぞれ球種別の投球割合を pie で描画する。
    参考ノートブック準拠：3行×4列の subplot レイアウト。"""
    from plotly.subplots import make_subplots
    if "ボールカウント" not in df.columns or "ストライクカウント" not in df.columns:
        st.info("カウントデータがありません")
        return
    if "球種" not in df.columns:
        st.info("球種データがありません")
        return

    d = df.copy()
    d["ボールカウント"] = pd.to_numeric(d["ボールカウント"], errors="coerce")
    d["ストライクカウント"] = pd.to_numeric(d["ストライクカウント"], errors="coerce")
    d = d.dropna(subset=["ボールカウント", "ストライクカウント", "球種"])
    if d.empty:
        st.info("カウントデータがありません")
        return

    d["__count"] = (d["ボールカウント"].astype(int).astype(str) + "-"
                    + d["ストライクカウント"].astype(int).astype(str))
    d["球種"] = d["球種"].astype(str).replace({
        "チェンジアップ": "チェンジ", "縦スライダー": "縦スラ",
        "カットボール": "カット"
    })

    # 12 カウント順序（参考コード準拠）
    counts_list = ["0-0", "1-0", "2-0", "3-0",
                   "0-1", "1-1", "2-1", "3-1",
                   "0-2", "1-2", "2-2", "3-2"]

    # 全球種で凡例の色を統一するため、登場球種を確定
    all_types = list(d["球種"].value_counts().index)

    # 3 行 × 4 列の subplot
    # subplot_titles はカウント文字列を渡す
    fig = make_subplots(
        rows=3, cols=4,
        specs=[[{"type": "domain"}]*4]*3,
        subplot_titles=counts_list,
        horizontal_spacing=0.03, vertical_spacing=0.12,
    )

    for i, cnt in enumerate(counts_list):
        r = i // 4 + 1
        c = i % 4 + 1
        pc_data = d[d["__count"] == cnt]
        if pc_data.empty:
            # データなし → 空の灰色ドーナツ
            fig.add_trace(
                go.Pie(
                    values=[1], labels=["（データなし）"],
                    marker_colors=["#E8ECEF"],
                    marker=dict(line=dict(color="#D6DEE5", width=1)),
                    hole=0.55,
                    textinfo="none",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=r, col=c,
            )
            continue
        vc = pc_data["球種"].value_counts()
        total = int(vc.sum())
        labels = [str(x) for x in vc.index]
        values = vc.values
        colors = [PITCH_COLORS.get(lab, "#888888") for lab in labels]
        fig.add_trace(
            go.Pie(
                values=values, labels=labels,
                marker_colors=colors,
                marker=dict(line=dict(color="white", width=1)),
                hole=0.55,
                sort=False,
                direction="clockwise",
                textinfo="percent",
                texttemplate="%{percent:.0%}",
                textposition="outside",
                textfont=dict(size=10, color="#1F2A37"),
                hovertemplate="%{label}: %{value} 球 (%{percent})<extra></extra>",
                showlegend=False,
            ),
            row=r, col=c,
        )
        # 中央（ドーナツの穴）に総球数を表示
        # subplot 中心座標を計算 (3行4列の domain)
        x_c = (c - 0.5) / 4
        y_c = 1.0 - (r - 0.5) / 3
        fig.add_annotation(
            x=x_c, y=y_c, xref="paper", yref="paper",
            text=f"<b>{total}</b>",
            showarrow=False,
            font=dict(size=13, color="#2C5F7C"),
        )

    # 凡例代わりに、上端に登場した球種を色チップで列挙
    legend_html_parts = []
    for pt in all_types:
        color = PITCH_COLORS.get(pt, "#888888")
        legend_html_parts.append(
            f"<span style='display:inline-block; margin:0 8px 0 0;'>"
            f"<span style='display:inline-block; width:11px; height:11px; "
            f"background:{color}; border-radius:50%; "
            f"margin-right:4px; vertical-align:middle;'></span>"
            f"<span style='font-size:12px; color:#1F2A37; "
            f"vertical-align:middle;'>{pt}</span></span>"
        )
    legend_html = ("<div style='text-align:center; margin:4px 0 8px 0;'>"
                   + "".join(legend_html_parts) + "</div>")

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 25, b=10),
        paper_bgcolor="#FFFFFF",
        showlegend=False,
    )
    # subplot タイトル（カウント表記）のスタイル統一
    # ★ ここで「カウント以外の余分な annotation」（plotly 内部で
    #   稀に発生する undefined テキスト）を非表示にする
    for ann in fig.layout.annotations:
        if ann.text in counts_list:
            ann.font = dict(size=13, color="#2C5F7C")
        elif ann.text and ann.text.startswith("<b>"):
            # 中央の総球数 — そのまま
            pass
        else:
            # それ以外（undefined や空文字）は隠す
            if not ann.text or ann.text == "undefined":
                ann.text = ""
                ann.showarrow = False

    if legend_html_parts:
        st.markdown(legend_html, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=key_prefix)


def draw_count_based_pitch_ratio_by_side(df: pd.DataFrame,
                                         key_prefix: str = "cnt_pitch"):
    """対右打者・対左打者ぶんでカウント別パイチャートを左右に並べて描画。
    視覚的な区切りを強化（ヘッダー帯＋細い縦の仕切り）。"""
    side_col = "打者打席左右" if "打者打席左右" in df.columns else None
    st.markdown("### カウント別投球割合")
    if not side_col:
        draw_count_based_pitch_ratio(df, key_prefix=f"{key_prefix}_all")
        return

    df_R = df[df[side_col].astype(str).str.contains("右", na=False)]
    df_L = df[df[side_col].astype(str).str.contains("左", na=False)]

    n_R = len(df_R)
    n_L = len(df_L)

    # 2 カラム + 中央の細い仕切り
    col_R, col_div, col_L = st.columns([10, 1, 10])

    with col_R:
        st.markdown(
            f"""
<div style="background: linear-gradient(135deg, #2C5F7C 0%, #3A6B85 100%);
            color: white; padding: 6px 12px; border-radius: 4px;
            text-align: center; font-weight: 600; font-size: 14px;
            margin-bottom: 8px;">
    対 右打者　<span style="font-size:12px; opacity:0.85;">（{n_R} 球）</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if df_R.empty:
            st.caption("対右打者のデータがありません")
        else:
            draw_count_based_pitch_ratio(df_R, key_prefix=f"{key_prefix}_R",
                                         height=460)

    with col_div:
        # 縦の仕切り線（細い灰色の縦バー）— モバイルでは CSS で非表示
        st.markdown(
            """
<div class="enemil-vert-divider"
     style="border-left: 1px solid #D6DEE5; height: 520px;
            margin: 8px auto 0 auto; width: 1px;">
</div>
""",
            unsafe_allow_html=True,
        )

    with col_L:
        st.markdown(
            f"""
<div style="background: linear-gradient(135deg, #6FA8C9 0%, #5A93B5 100%);
            color: white; padding: 6px 12px; border-radius: 4px;
            text-align: center; font-weight: 600; font-size: 14px;
            margin-bottom: 8px;">
    対 左打者　<span style="font-size:12px; opacity:0.85;">（{n_L} 球）</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if df_L.empty:
            st.caption("対左打者のデータがありません")
        else:
            draw_count_based_pitch_ratio(df_L, key_prefix=f"{key_prefix}_L",
                                         height=460)


# ──────────────────────────────────────────
# 投球コース（投球到達位置）プロット
# 安打 / 凡打 / 空振り / ファウル それぞれを 1 枚に表示。
# 球種ごとに色とマーカー形状を変える（参考ノートブック draw_scatter_plot 準拠）
# ──────────────────────────────────────────
def _pitch_marker_symbol(pitch_type: str, throw_hand: str,
                         view: str = "catcher") -> str:
    """球種と投手の利き腕に応じた Plotly マーカーシンボル名を返す。
    matplotlib の `<` `>` `p` `v` `*` を Plotly の symbol 名にマッピング。
    捕手目線では左右を反転させる（マーカーの向きの感じを揃えるため）。"""
    pt = "" if pitch_type is None else str(pitch_type).strip()
    hand = "" if throw_hand is None else str(throw_hand).strip()
    if hand == "右":
        hand = "R"
    elif hand == "左":
        hand = "L"
    # 捕手目線：左右反転
    if view == "catcher":
        if hand == "R":
            hand = "L"
        elif hand == "L":
            hand = "R"

    if pt == "ストレート":
        return "circle"
    elif pt in ("スライダー", "カットボール", "カット", "縦スライダー", "縦スラ"):
        # `<` (R), `>` (L)
        return "triangle-left" if hand == "R" else "triangle-right"
    elif pt == "カーブ":
        return "pentagon"  # matplotlib `p`
    elif pt in ("スプリット", "チェンジアップ", "チェンジ", "フォーク"):
        return "triangle-down"  # `v`
    elif pt == "ツーシーム":
        # `>` (R), `<` (L)
        return "triangle-right" if hand == "R" else "triangle-left"
    elif pt == "特殊球":
        return "star"
    else:
        return "circle"


def _homeplate_path_for_outcome(view: str = "catcher") -> str:
    """投球コースプロット用ホームベース：参考ノートブック draw_scatter_plot 準拠。
    catcher: 下（手前）が尖る五角形 [(240,40),(60,40),(60,20),(150,0),(240,20)]
    pitcher: 上（奥）が尖る五角形   [(240,0),(60,0),(60,20),(150,40),(240,20)]"""
    if view == "catcher":
        return "M 240 40 L 60 40 L 60 20 L 150 0 L 240 20 Z"
    else:
        return "M 240 0 L 60 0 L 60 20 L 150 40 L 240 20 Z"


def _draw_one_pitch_outcome(sub_df: pd.DataFrame, title: str,
                            view: str = "catcher",
                            height: int = 320, key: str = "pco"):
    """1パネル分の散布図を描く。sub_df は対象球（hit / out / whiff / foul）のみ。"""
    fig = go.Figure()
    # ストライクゾーン
    fig.add_shape(
        type="rect", x0=60, x1=240, y0=60, y1=240,
        line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)",
        layer="above",
    )
    # ホームベース
    fig.add_shape(
        type="path", path=_homeplate_path_for_outcome(view=view),
        line=dict(color="black", width=2),
        fillcolor="rgba(255,255,255,0)",
        layer="above",
    )

    if not sub_df.empty:
        # 球種ごとにグループ化してプロット
        for pt in sub_df["球種"].dropna().astype(str).unique():
            mask = sub_df["球種"].astype(str) == pt
            grp = sub_df[mask]
            xs = pd.to_numeric(grp["投球位置x座標"], errors="coerce").tolist()
            ys = pd.to_numeric(grp["投球位置y座標"], errors="coerce").tolist()
            color = PITCH_COLORS.get(pt, "#757575")
            # マーカーシンボルは行ごとに（投手左右が違う場合）
            hands = grp.get("投手投げ手左右", pd.Series([""] * len(grp))).tolist()
            symbols = [_pitch_marker_symbol(pt, h, view=view) for h in hands]
            # hover info: 球種＋投手＋日付
            pitchers = grp.get("投手名", pd.Series([""] * len(grp))).astype(str).tolist()
            dates = grp.get("日付", pd.Series([""] * len(grp))).astype(str).tolist()
            custom = [[pt, pchr, dt] for pchr, dt in zip(pitchers, dates)]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(
                    symbol=symbols,
                    color=color, size=11,
                    line=dict(color="black", width=1),
                ),
                customdata=custom,
                hovertemplate=(
                    "球種: %{customdata[0]}<br>"
                    "投手: %{customdata[1]}<br>"
                    "日付: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
                name=pt,
            ))

    x_axis_kwargs = dict(
        range=[300, 0] if view == "catcher" else [0, 300],
        showgrid=False, zeroline=False, showticklabels=False,
        scaleanchor="y", scaleratio=1,
    )
    y_axis_kwargs = dict(
        range=[0, 300],
        showgrid=False, zeroline=False, showticklabels=False,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#1F2A37"),
                   x=0.5, xanchor="center"),
        xaxis=x_axis_kwargs, yaxis=y_axis_kwargs,
        height=height,
        margin=dict(l=5, r=5, t=30, b=5),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def draw_pitch_outcome_scatter_by_side(df: pd.DataFrame,
                                       view: str = "catcher",
                                       key_prefix: str = "pco"):
    """投球到達位置プロットを「対右投手」「対左投手」それぞれで、
    安打 / 凡打 / 空振り / ファウル の4枚 × 2 列で描画。
    球種ごとに色と形状を変える。"""
    if "投球位置x座標" not in df.columns or "投球位置y座標" not in df.columns:
        st.info("投球位置データがありません")
        return
    if "球種" not in df.columns:
        st.info("球種データがありません")
        return

    d = df.copy()
    d = d.dropna(subset=["投球位置x座標", "投球位置y座標"])
    if d.empty:
        st.info("プロットできるデータがありません")
        return

    hand_col = _pick_col(d, ["投手投げ手左右", "投手投球左右", "PitcherHand"])
    if hand_col:
        d_R = d[d[hand_col].astype(str).str.contains("右|R", na=False)]
        d_L = d[d[hand_col].astype(str).str.contains("左|L", na=False)]
    else:
        d_R = d
        d_L = d.iloc[0:0]

    import re as _re
    # 4分類を抽出する関数
    def _classify(data: pd.DataFrame) -> dict:
        if data.empty:
            return {"hit": data, "out": data, "whiff": data, "foul": data}
        res = data["打席結果"].astype(str) if "打席結果" in data.columns else pd.Series([""] * len(data))
        pr  = data["投球結果"].astype(str) if "投球結果" in data.columns else pd.Series([""] * len(data))

        # 打席結果ベースの hit / out（結果球＝最終球の行）
        # 結果球==1 を「打席を締めくくる行」として扱う
        is_final_row = (pd.to_numeric(data.get("結果球", 0), errors="coerce").fillna(0) == 1)
        if not is_final_row.any():
            # フォールバック：打席結果が記入されている行を最終行とみなす
            is_final_row = res.notna() & (res != "") & (res != "nan")

        # is_ab: 四球・死球・犠打・失策を除外
        is_ab = ~res.str.contains(r"四球|死球|犠|失策", na=False, regex=True)
        # is_hit: 安打系
        is_hit = res.str.contains(r"本|安|２|３", na=False, regex=True)

        hit_mask = is_final_row & is_ab & is_hit
        out_mask = is_final_row & is_ab & ~is_hit
        whiff_mask = pr.eq("空振り")
        foul_mask = pr.isin(["ファウル", "ファール"])

        return {
            "hit":   data[hit_mask],
            "out":   data[out_mask],
            "whiff": data[whiff_mask],
            "foul":  data[foul_mask],
        }

    panels_R = _classify(d_R)
    panels_L = _classify(d_L)

    titles = [("hit", "安打"), ("out", "凡打"),
              ("whiff", "空振り"), ("foul", "ファウル")]

    # 全体球数のサマリ（小さく1行で）
    st.caption(
        f"対 右投手：全 {len(d_R)} 球 ／ 対 左投手：全 {len(d_L)} 球"
    )

    # 各分類について、対右と対左を 2 列で並べる。
    # 各列の中で「対 右投手 — カテゴリ（n 球）」帯 → 図 の順に配置。
    # こうすると、モバイルで横並びが縦並びに折り返されてもタイトルが
    # 必ずその直下の図の説明になり、対応関係が崩れない。
    for plot_key, label in titles:
        col_R, col_L = st.columns(2)
        with col_R:
            n_r = len(panels_R[plot_key])
            st.markdown(
                "<div style='background:linear-gradient(135deg,#2C5F7C 0%,#3A6B85 100%);"
                "color:white;padding:6px 12px;border-radius:4px;text-align:center;"
                f"font-weight:600;margin-bottom:4px;'>対 右投手 — {label}（{n_r} 球）</div>",
                unsafe_allow_html=True,
            )
            _draw_one_pitch_outcome(
                panels_R[plot_key], title="", view=view, height=300,
                key=f"{key_prefix}_R_{plot_key}",
            )
        with col_L:
            n_l = len(panels_L[plot_key])
            st.markdown(
                "<div style='background:linear-gradient(135deg,#6FA8C9 0%,#5A93B5 100%);"
                "color:white;padding:6px 12px;border-radius:4px;text-align:center;"
                f"font-weight:600;margin-bottom:4px;'>対 左投手 — {label}（{n_l} 球）</div>",
                unsafe_allow_html=True,
            )
            _draw_one_pitch_outcome(
                panels_L[plot_key], title="", view=view, height=300,
                key=f"{key_prefix}_L_{plot_key}",
            )

    # 凡例（球種色＋形状）
    legend_html_parts = []
    seen_pts = sorted(set(
        list(d["球種"].dropna().astype(str).unique()) if "球種" in d.columns else []
    ))
    for pt in seen_pts:
        color = PITCH_COLORS.get(pt, "#757575")
        legend_html_parts.append(
            f"<span style='display:inline-block; margin:0 10px 4px 0;'>"
            f"<span style='display:inline-block; width:11px; height:11px;"
            f" background:{color}; border-radius:50%;"
            f" margin-right:5px; vertical-align:middle;'></span>"
            f"<span style='font-size:12px; color:#1F2A37;'>{pt}</span></span>"
        )
    if legend_html_parts:
        st.markdown(
            "<div style='text-align:center; margin:8px 0 4px 0;'>"
            + "".join(legend_html_parts) + "</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "マーカー形状：● ストレート　◀/▶ スライダー・カット系（投手の利き腕で向きが変化）"
        "　⬟ カーブ　▼ チェンジ・スプリット　★ 特殊球"
    )


# ──────────────────────────────────────────
# スプレーチャート（参考ノートブック準拠）
# ──────────────────────────────────────────
def draw_spray_chart(df: pd.DataFrame, key: str = "spray",
                     height: int = 560, title: str = "スプレーチャート"):
    """
    打球位置x座標 / y座標 をそのまま使用（回転なし）。
    内野ダイヤモンドは (0,0)-(105,105) の正方形、本塁=(0,0)。
    マーカー：フライ=○、ライナー=△、ゴロ=＋（十字）
    色：安打=赤、凡打=黒
    プロット範囲：x ∈ [-20, 300], y ∈ [-80, 300]（正方形）
    """
    needed = {"打球位置x座標", "打球位置y座標"}
    if not needed.issubset(df.columns):
        st.info("打球座標データがありません")
        return
    d = df.dropna(subset=["打球位置x座標", "打球位置y座標"]).copy()
    if d.empty:
        st.info("有効な打球データがありません")
        return
    d["_x"] = pd.to_numeric(d["打球位置x座標"], errors="coerce")
    d["_y"] = pd.to_numeric(d["打球位置y座標"], errors="coerce")
    d = d.dropna(subset=["_x", "_y"])
    if d.empty:
        st.info("有効な打球データがありません")
        return

    # 打球分類：hit/out × フライ/ライナー/ゴロ
    import re as _re
    plot_data = {}  # (h_type, b_key) -> {"x":[], "y":[], "hover":[]}
    for _, row in d.iterrows():
        res = str(row.get("打席結果", ""))
        b_type = str(row.get("打球性質", ""))
        if pd.isna(row.get("打席結果")) or res == "nan" or res == "":
            continue
        # 打数になる打席（四球・死球・犠打・失策は除く）
        is_ab = not bool(_re.search(r"四球|死球|犠|失策", res))
        is_hit = bool(_re.search(r"本|安|２|３", res))
        if not is_ab:
            continue

        if "フライ" in b_type:
            b_key = "フライ"
        elif "ライナー" in b_type:
            b_key = "ライナー"
        elif "ゴロ" in b_type:
            b_key = "ゴロ"
        else:
            continue

        # hover に表示する補助情報を集める
        pt_name  = row.get("球種", "")
        pitcher  = row.get("投手名", "")
        game_dt  = row.get("日付", "")
        hover = {
            "打席結果": res,
            "球種": str(pt_name) if pt_name and not pd.isna(pt_name) else "—",
            "投手": str(pitcher) if pitcher and not pd.isna(pitcher) else "—",
            "日付": str(game_dt) if game_dt and not pd.isna(game_dt) else "—",
        }

        h_type = "hit" if is_hit else "out"
        k = (h_type, b_key)
        plot_data.setdefault(k, {"x": [], "y": [], "hover": []})
        plot_data[k]["x"].append(row["_x"])
        plot_data[k]["y"].append(row["_y"])
        plot_data[k]["hover"].append(hover)

    # 軸範囲
    x_range = [-20, 300]
    y_range = [-80, 300]

    fig = go.Figure()

    # ── ① 内野ダイヤモンド（参考コードでは (0,0)-(105,105) の正方形を描画） ──
    fig.add_trace(go.Scatter(
        x=[0, 105, 105, 0, 0], y=[0, 0, 105, 105, 0],
        mode="lines", line=dict(color="black", width=1.8),
        showlegend=False, hoverinfo="skip",
    ))

    # ── ② 内野アーチ（2次補間相当を多項式フィット）──
    try:
        from scipy import interpolate as _interp
        x_in_pts = np.array([142.0, 113.0, 0.0])
        y_in_pts = np.array([0.0, 113.0, 142.0])
        theta_in = np.arctan2(y_in_pts, x_in_pts)
        r_in = np.hypot(x_in_pts, y_in_pts)
        idx_in = np.argsort(theta_in)
        theta_in, r_in = theta_in[idx_in], r_in[idx_in]
        f_in = _interp.interp1d(theta_in, r_in, kind="quadratic")
        t_fine_in = np.linspace(theta_in.min(), theta_in.max(), 100)
        r_fine_in = f_in(t_fine_in)
        in_x = r_fine_in * np.cos(t_fine_in)
        in_y = r_fine_in * np.sin(t_fine_in)
    except Exception:
        in_x = np.array([142, 113, 0])
        in_y = np.array([0, 113, 142])
    fig.add_trace(go.Scatter(
        x=in_x, y=in_y, mode="lines",
        line=dict(color="black", width=1.6),
        showlegend=False, hoverinfo="skip",
    ))

    # ── ③ 外野フェンスアーチ ──
    try:
        from scipy import interpolate as _interp
        x_out_pts = np.array([256.0, 273.0, 282.0, 260.0, 200.0,
                              119.0, 37.0, 0.0])
        y_out_pts = np.array([0.0, 37.0, 119.0, 200.0, 260.0,
                              282.0, 273.0, 256.0])
        theta_out = np.arctan2(y_out_pts, x_out_pts)
        r_out = np.hypot(x_out_pts, y_out_pts)
        idx_out = np.argsort(theta_out)
        theta_out, r_out = theta_out[idx_out], r_out[idx_out]
        f_out = _interp.interp1d(theta_out, r_out, kind="cubic")
        t_fine_out = np.linspace(theta_out.min(), theta_out.max(), 120)
        r_fine_out = f_out(t_fine_out)
        out_x = r_fine_out * np.cos(t_fine_out)
        out_y = r_fine_out * np.sin(t_fine_out)
    except Exception:
        out_x = np.array([256, 273, 282, 260, 200, 119, 37, 0])
        out_y = np.array([0, 37, 119, 200, 260, 282, 273, 256])
    fig.add_trace(go.Scatter(
        x=out_x, y=out_y, mode="lines",
        line=dict(color="black", width=1.8),
        showlegend=False, hoverinfo="skip",
    ))

    # ── ④ ファウルライン（本塁から1塁線・3塁線方向） ──
    fig.add_trace(go.Scatter(
        x=[0, 256], y=[0, 0], mode="lines",
        line=dict(color="black", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, 256], mode="lines",
        line=dict(color="black", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))

    # ── ⑤ プロットデータ（カテゴリ別マーカー・色） ──
    marker_map = {"フライ": "circle", "ライナー": "triangle-up",
                  "ゴロ": "cross"}
    color_map = {"hit": "#D32F2F", "out": "#222222"}
    label_map = {"hit": "安打", "out": "凡打"}
    # 凡例の並び順を固定
    order = [("hit", "フライ"), ("hit", "ライナー"), ("hit", "ゴロ"),
             ("out", "フライ"), ("out", "ライナー"), ("out", "ゴロ")]
    for k in order:
        if k not in plot_data:
            continue
        h_type, b_key = k
        data = plot_data[k]
        if not data["x"]:
            continue
        label = f"{label_map[h_type]}（{b_key}）"
        # customdata は 4列 (打席結果, 球種, 投手, 日付) の 2D 配列
        custom = [[h["打席結果"], h["球種"], h["投手"], h["日付"]]
                  for h in data["hover"]]
        fig.add_trace(go.Scatter(
            x=data["x"], y=data["y"], mode="markers",
            name=label,
            marker=dict(
                symbol=marker_map[b_key],
                color=color_map[h_type],
                size=10 if b_key != "ゴロ" else 11,
                line=dict(color="white", width=0.8),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "球種: %{customdata[1]}<br>"
                "投手: %{customdata[2]}<br>"
                "日付: %{customdata[3]}"
                "<extra></extra>"
            ),
        ))

    # ── レイアウト ──
    # アスペクト比 1:1 を厳密に維持するため、scaleanchor を使う
    fig.update_layout(
        title=title,
        xaxis=dict(range=x_range, showgrid=False, zeroline=False,
                   showticklabels=False,
                   constrain="domain"),
        yaxis=dict(range=y_range, showgrid=False, zeroline=False,
                   showticklabels=False,
                   scaleanchor="x", scaleratio=1,
                   constrain="domain"),
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h", y=-0.02, x=0.5, xanchor="center",
            font=dict(size=11), bgcolor="rgba(255,255,255,0.6)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ──────────────────────────────────────────
# バレットチャート（参考ノートブック準拠：パーセンタイル表示）
# ──────────────────────────────────────────
def _percentile_of_value(v: float, arr) -> float:
    """値 v のパーセンタイル位置（0〜1）を返す。
    参考ノートブックの percentile_of_value と同じロジック。"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(v):
        return float("nan")
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmin == vmax:
        return 0.5
    if v <= vmin:
        return 0.0
    if v >= vmax:
        return 1.0
    strict = int(np.sum(arr < v))
    weak = int(np.sum(arr <= v))
    return (strict + weak) / 2.0 / arr.size


def _percentile_to_rgb(p: float):
    """パーセンタイル(0〜1) → matplotlib 'bwr' カラーマップに 0.25 ホワイト混合
    した RGB タプル (0〜1)。参考ノートブック percentile_to_rgb と同じ。"""
    if not np.isfinite(p):
        return (1.0, 1.0, 1.0)
    p = float(max(0.0, min(1.0, p)))
    # bwr: blue(0,0,1) -> white(1,1,1) -> red(1,0,0)
    if p <= 0.5:
        # blue -> white
        t = p * 2.0
        r = 0.0 + (1.0 - 0.0) * t
        g = 0.0 + (1.0 - 0.0) * t
        b = 1.0 + (1.0 - 1.0) * t
    else:
        # white -> red
        t = (p - 0.5) * 2.0
        r = 1.0
        g = 1.0 + (0.0 - 1.0) * t
        b = 1.0 + (0.0 - 1.0) * t
    # 25% ホワイト混合
    mix = 0.25
    r = r * (1 - mix) + 1.0 * mix
    g = g * (1 - mix) + 1.0 * mix
    b = b * (1 - mix) + 1.0 * mix
    return (r, g, b)


def _rgb_to_css(rgb):
    r, g, b = rgb
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def draw_bullet_chart_percentile(items: list, title: str = "Batting",
                                 key: str = "bullet_pct"):
    """
    参考ノートブック準拠のバレットチャート。
    items: [{"label", "value", "percentile" (0..1), "value_text"}]
    各バーは:
    - 背景: 灰色 (#DADADA) の細いバー
    - 塗り: パーセンタイルに応じた bwr 色のバー（同じ太さ）
    - 円: 値の位置に白丸、中にパーセンタイル数値（1〜100）
    - 値テキスト: 右側
    """
    if not items:
        st.info("バレットチャート用データがありません")
        return

    n = len(items)
    fig = go.Figure()

    # 座標系はすべて paper（0..1）に正規化して使う
    # ・add_shape / add_annotation / scatter のいずれも xref="paper"/yref="paper" を明示
    # ★ value_x の最大値が 1.10 なので、xaxis の range は [0, 1.15]
    label_right_x = 0.18
    bar_x0 = 0.22
    bar_x1 = 0.92
    value_x = 1.10
    top_y = 0.93
    bottom_y = 0.05
    row_h = (top_y - bottom_y) / max(1, n)

    for i, it in enumerate(items):
        y0 = top_y - (i + 1) * row_h
        y_center = y0 + row_h * 0.52
        bar_h = row_h * 0.46

        p = it.get("percentile", float("nan"))
        p_clip = float(max(0.0, min(1.0, p))) if np.isfinite(p) else float("nan")

        # ─ ① 背景バー（灰色） ─
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=bar_x0, x1=bar_x1,
            y0=y_center - bar_h / 2, y1=y_center + bar_h / 2,
            fillcolor="#DADADA", line=dict(color="#B5B5B5", width=0.8),
            layer="below",
        )

        # ─ ② 塗りバー（パーセンタイル色） ─
        if np.isfinite(p_clip):
            bar_rgb = _percentile_to_rgb(p_clip)
            fill = _rgb_to_css(bar_rgb)
        else:
            fill = "#FFFFFF"
        # 塗りバーは bar_x0 〜 (bar_x0 + (bar_x1-bar_x0)*p_clip) の幅で
        bar_w = (bar_x1 - bar_x0) * (p_clip if np.isfinite(p_clip) else 0)
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=bar_x0, x1=bar_x0 + bar_w,
            y0=y_center - bar_h / 2, y1=y_center + bar_h / 2,
            fillcolor=fill, line=dict(width=0),
            layer="below",
        )

        # ─ ③ 値の位置に円 ─
        circle_x = (bar_x0 + (bar_x1 - bar_x0) * p_clip
                    if np.isfinite(p_clip) else bar_x0)
        pct_int = int(round(p_clip * 100)) if np.isfinite(p_clip) else 0
        if pct_int < 1 and np.isfinite(p_clip):
            pct_int = 1
        elif pct_int > 100:
            pct_int = 100

        if np.isfinite(p_clip):
            edge_color = _rgb_to_css(_percentile_to_rgb(p_clip))
            if 0.45 <= p_clip <= 0.55:
                edge_color = "#333333"
        else:
            edge_color = "#333333"

        fig.add_trace(go.Scatter(
            x=[circle_x], y=[y_center],
            xaxis="x", yaxis="y",
            mode="markers+text",
            marker=dict(
                size=22, color="#FFFFFF",
                line=dict(color=edge_color, width=1.5),
            ),
            text=[f"<b>{pct_int}</b>" if np.isfinite(p_clip) else ""],
            textfont=dict(size=11, color="#000000"),
            textposition="middle center",
            hoverinfo="skip", showlegend=False,
        ))

        # ─ ④ 指標ラベル（左） ─
        fig.add_annotation(
            x=label_right_x, y=y_center,
            xref="paper", yref="paper",
            text=f"<b>{it['label']}</b>",
            showarrow=False, xanchor="right", yanchor="middle",
            font=dict(size=14, color="#000000"),
        )

        # ─ ⑤ 値テキスト（右） ─
        v_text = it.get("value_text", "")
        fig.add_annotation(
            x=value_x, y=y_center,
            xref="paper", yref="paper",
            text=f"<b>{v_text}</b>",
            showarrow=False, xanchor="right", yanchor="middle",
            font=dict(size=13, color="#000000"),
        )

    # Scatter の値（円）は data 座標で配置している（x=circle_x, y=y_center）。
    # circle_x も y_center も 0..1 の範囲なので、xaxis/yaxis を 0..1 に固定。
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)) if title else None,
        xaxis=dict(range=[0, 1.0], visible=False,
                   showgrid=False, zeroline=False,
                   fixedrange=True),
        yaxis=dict(range=[0, 1.0], visible=False,
                   showgrid=False, zeroline=False,
                   fixedrange=True),
        height=max(200, 55 * n + 40),
        margin=dict(l=0, r=10, t=40 if title else 10, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# 旧バレットチャート関数（後方互換のため保持）— normalized 版を percentile 版に転送
def draw_bullet_chart_normalized(items: list, title: str = "Batting",
                                 key: str = "bullet_n"):
    """後方互換ラッパー：新しい percentile 版を呼ぶ。
    items に "percentile" が含まれていなければ、value を min/max から
    パーセンタイル換算する。"""
    if not items:
        st.info("バレットチャート用データがありません")
        return

    converted = []
    for it in items:
        v = it.get("value")
        if "percentile" in it and it["percentile"] is not None:
            p = it["percentile"]
        elif v is None or pd.isna(v):
            p = float("nan")
        else:
            vmin = float(it.get("min", 0.0))
            vmax = float(it.get("max", 1.0))
            if vmax <= vmin:
                vmax = vmin + 1.0
            p = (float(v) - vmin) / (vmax - vmin)
            if it.get("better", "high") == "low":
                p = 1.0 - p
        # value_text 整形
        if "value_text" in it:
            vtxt = it["value_text"]
        elif v is None or pd.isna(v):
            vtxt = "—"
        elif isinstance(v, float):
            label = it.get("label", "")
            if label in ("wOBA", "ISO", "OPS", "打率", "出塁率", "長打率"):
                vtxt = f"{v:.3f}"
            else:
                vtxt = f"{v:.1f}"
        else:
            vtxt = str(v)
        converted.append({
            "label": it["label"],
            "value": v,
            "percentile": p,
            "value_text": vtxt,
        })
    draw_bullet_chart_percentile(converted, title=title, key=key)


# 古い API も別名で残す
def draw_bullet_chart(items: list, title: str = "指標比較",
                      key: str = "bullet"):
    draw_bullet_chart_normalized(items, title=title, key=key)


# ──────────────────────────────────────────
# KPI 推移グラフ
# ──────────────────────────────────────────
def draw_kpi_trend(kpi_data: list, kpi_key: str, kpi_label: str,
                   unit: str, better: str, window: int = 1,
                   key: str = "kpi"):
    if not kpi_data:
        st.info("データがありません")
        return

    df = pd.DataFrame(kpi_data).dropna(subset=["value"])
    df = df.sort_values("date").reset_index(drop=True)

    if window > 1:
        df["group"] = df.index // window
        grouped = df.groupby("group").agg(
            date=("date", "first"),
            date_end=("date", "last"),
            value=("value", "mean"),
            n=("value", "count"),
        ).reset_index(drop=True)
        grouped["label"] = grouped.apply(
            lambda r: r["date"] if r["date"] == r["date_end"]
                      else f"{r['date']}〜{r['date_end']}", axis=1
        )
        x_vals = grouped["label"].tolist()
        y_vals = grouped["value"].tolist()
        hover  = [f"n={r['n']}<br>{r['value']:.2f} {unit}"
                  for _, r in grouped.iterrows()]
    else:
        x_vals = df["date"].tolist()
        y_vals = df["value"].tolist()
        hover  = [f"{v:.2f} {unit}" for v in y_vals]

    avg = float(np.mean(y_vals))
    direction = "↑ 高いほど良い" if better == "high" else "↓ 低いほど良い"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines+markers", name=kpi_label,
        hovertext=hover, hoverinfo="x+text",
        line=dict(width=2), marker=dict(size=7),
    ))
    fig.add_hline(y=avg, line_dash="dash", line_color="gray",
                  annotation_text=f"平均 {avg:.2f}",
                  annotation_position="bottom right")

    fig.update_layout(
        title=f"{kpi_label}  {direction}",
        xaxis_title="試合日",
        yaxis_title=f"{kpi_label} ({unit})" if unit else kpi_label,
        height=280, margin=dict(l=40, r=40, t=50, b=60),
        xaxis=dict(type="category", tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ──────────────────────────────────────────
# スタッツテーブル：青-白-赤グラデーション着色（HTML直接生成）
# ──────────────────────────────────────────
def _format_cell_value(v, col_name: str, is_float_col: bool,
                       small_decimal: bool, is_int_col: bool = False,
                       force_2dp: bool = False) -> str:
    """セル値を整形して文字列にする。
    force_2dp が True なら小数第2位まで（打率系の3桁デフォルトを上書き）。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if is_int_col:
        try:
            fv = float(v)
            if pd.isna(fv):
                return "—"
            return str(int(round(fv)))
        except Exception:
            return str(v)
    if force_2dp:
        try:
            fv = float(v)
            if pd.isna(fv):
                return "—"
            return f"{fv:.2f}"
        except Exception:
            return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) or is_float_col:
        try:
            fv = float(v)
            if pd.isna(fv):
                return "—"
            # 打率系（絶対値2未満）は3桁、その他は2桁
            if small_decimal:
                return f"{fv:.3f}"
            return f"{fv:.2f}"
        except Exception:
            return str(v)
    return str(v)


def _cell_bg_color(v, vmin: float, vmean: float, vmax: float,
                   better: str) -> str:
    """値に応じた背景色 (CSS rgb()) を返す。値が無効なら空文字列。"""
    cmap_high = ["#5A8AC6", "#FFFFFF", "#F8696B"]  # 青→白→赤
    cmap_low  = ["#F8696B", "#FFFFFF", "#5A8AC6"]  # 赤→白→青
    try:
        f = float(v)
    except Exception:
        return ""
    if pd.isna(f) or vmax == vmin:
        return ""
    cmap = cmap_low if better == "low" else cmap_high
    if f <= vmean:
        norm = 0.5 * (f - vmin) / (vmean - vmin) if vmean != vmin else 0.5
    else:
        norm = 0.5 + 0.5 * (f - vmean) / (vmax - vmean) if vmax != vmean else 0.5
    norm = max(0.0, min(1.0, norm))
    if norm <= 0.5:
        t = norm * 2
        c0, c1 = cmap[0], cmap[1]
    else:
        t = (norm - 0.5) * 2
        c0, c1 = cmap[1], cmap[2]
    r0, g0, b0 = _hex_to_rgb(c0)
    r1, g1, b1 = _hex_to_rgb(c1)
    r = int(r0 + (r1 - r0) * t)
    g = int(g0 + (g1 - g0) * t)
    b = int(b0 + (b1 - b0) * t)
    return f"rgb({r},{g},{b})"


def render_colored_stats_table(df: pd.DataFrame,
                               metric_directions: dict = None,
                               key: str = "tbl",
                               min_count: int = 0,
                               count_col: str = None,
                               distributions: dict = None,
                               int_cols: list = None,
                               decimals_2_cols: list = None,
                               no_color_cols: list = None,
                               reverse_color_cols: list = None):
    """
    青-白-赤グラデーションで着色したテーブルを HTML で直接描画。
    pandas Styler / jinja2 に依存しないため pandas のバージョンに左右されない。
    metric_directions: {"指標名": "high"|"low"}
    min_count / count_col: 指定すると count_col の値がしきい値未満の行は着色しない。
    distributions: {"指標名": np.array([...])} を渡すと、各セルをこの分布の
                    min/mean/max を基準に着色する（リーグ全体ベースの絶対着色）。
                    1行表でも有効。
    int_cols: 整数として表示する列名のリスト。
    decimals_2_cols: 小数第2位で表示する列名のリスト。
                     ※打率系の3桁デフォルトより優先される。
    no_color_cols: 着色をスキップする列名のリスト。
    reverse_color_cols: 着色のカラースケールを反転する列名のリスト
                       （例: # 順位列で 1 を赤、最後を青にする）。"""
    if df.empty:
        st.info("表示するデータがありません")
        return

    metric_directions = metric_directions or {}
    distributions = distributions or {}
    int_cols_set = set(int_cols or [])
    decimals_2_set = set(decimals_2_cols or [])
    no_color_set = set(no_color_cols or [])
    reverse_color_set = set(reverse_color_cols or [])
    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])]

    # 各列の min/mean/max を事前計算
    col_stats = {}
    for c in numeric_cols:
        # 外部分布があればそれを使う、なければ df 内の値で計算
        if c in distributions:
            arr = np.asarray(distributions[c], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                col_stats[c] = None
            else:
                col_stats[c] = (float(arr.min()), float(arr.mean()),
                                float(arr.max()))
        else:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.empty:
                col_stats[c] = None
            else:
                col_stats[c] = (float(s.min()), float(s.mean()), float(s.max()))

    # 小数列の判定（最大絶対値が2未満なら打率系として3桁表示）
    small_decimal_cols = set()
    float_cols = set()
    for c in numeric_cols:
        try:
            if df[c].dtype.kind == "f":
                float_cols.add(c)
                if df[c].abs().max() < 2:
                    small_decimal_cols.add(c)
        except Exception:
            pass

    # min_count マスク
    if count_col and count_col in df.columns:
        mask = pd.to_numeric(df[count_col], errors="coerce").fillna(0) >= min_count
    else:
        mask = pd.Series(True, index=df.index)

    # HTML 生成
    rows_html = []
    # ヘッダー
    th_cells = "".join(
        f'<th style="padding:7px 10px; background:#E7ECF0; '
        f'border-bottom:2px solid #2C5F7C; text-align:center; '
        f'font-weight:600; font-size:12.5px; color:#2C5F7C; '
        f'white-space:nowrap;">{c}</th>'
        for c in df.columns
    )
    rows_html.append(f"<tr>{th_cells}</tr>")

    # データ行
    for i, (_, row) in enumerate(df.iterrows()):
        cell_html = []
        apply_color = bool(mask.iloc[i])
        for c in df.columns:
            v = row[c]
            text = _format_cell_value(
                v, c,
                is_float_col=(c in float_cols),
                small_decimal=(c in small_decimal_cols),
                is_int_col=(c in int_cols_set),
                force_2dp=(c in decimals_2_set),
            )
            # 着色対象か判定
            colorable = (c in numeric_cols and apply_color
                         and col_stats.get(c) is not None
                         and c not in no_color_set)
            if colorable:
                vmin, vmean, vmax = col_stats[c]
                better = metric_directions.get(c, "high")
                # reverse_color_cols が指定されていれば direction を反転
                if c in reverse_color_set:
                    better = "low" if better == "high" else "high"
                bg = _cell_bg_color(v, vmin, vmean, vmax, better)
                style = (
                    f"padding:5px 10px; text-align:center; "
                    f"font-size:12.5px; color:#111; "
                    f"border-bottom:1px solid #E0E6EC; "
                    f"white-space:nowrap;"
                )
                if bg:
                    style += f"background-color:{bg};"
                cell_html.append(f'<td style="{style}">{text}</td>')
            else:
                style = (
                    "padding:5px 10px; text-align:center; "
                    "font-size:12.5px; color:#222; "
                    "border-bottom:1px solid #E0E6EC; "
                    "white-space:nowrap;"
                )
                cell_html.append(f'<td style="{style}">{text}</td>')
        rows_html.append(f"<tr>{''.join(cell_html)}</tr>")

    # 全体を table タグで囲む（横スクロール対応）
    table_html = (
        '<div style="overflow-x:auto; margin:6px 0 14px 0;">'
        '<table style="border-collapse:collapse; '
        'width:100%; font-family:-apple-system,BlinkMacSystemFont,'
        '\'Segoe UI\',sans-serif;">'
        + "".join(rows_html) +
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
