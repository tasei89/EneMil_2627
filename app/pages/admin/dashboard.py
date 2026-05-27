# -*- coding: utf-8 -*-
"""管理者ダッシュボード"""
import os, streamlit as st, pandas as pd
from ...auth import (get_all_users, add_user, reset_password, update_user,
                      delete_user, get_event_log, validate_password, log_event)

DATA_DIR = os.environ.get("DATA_DIR", "data")


def render():
    from ...components.header import render_page_header
    render_page_header("管理者ダッシュボード", icon="⚙️")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 利用概要",
        "👥 ユーザー別",
        "🔥 操作ヒートマップ",
        "👤 ユーザー管理",
        "📁 データ管理",
    ])

    with tab1:
        _render_overview()
    with tab2:
        _render_user_activity()
    with tab3:
        _render_heatmap()
    with tab4:
        _render_user_management()
    with tab5:
        _render_data_management()


# ──────────────────────────────────────────
# タブ①：利用概要
# ──────────────────────────────────────────
def _render_overview():
    logs = get_event_log(500)
    df = pd.DataFrame(logs)
    if df.empty:
        st.info("ログデータがありません"); return

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    today = pd.Timestamp.now().normalize()

    login_df = df[df["event_type"] == "login"]
    dau = login_df[login_df["ts"].dt.normalize() == today]["user_id"].nunique()
    wau = login_df[login_df["ts"] >= today - pd.Timedelta(days=7)]["user_id"].nunique()
    mau = login_df[login_df["ts"] >= today - pd.Timedelta(days=30)]["user_id"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("本日のアクティブユーザー", dau)
    c2.metric("今週のアクティブユーザー", wau)
    c3.metric("今月のアクティブユーザー", mau)

    st.divider()
    st.markdown("#### 直近ログイン")
    recent = login_df.sort_values("ts", ascending=False).head(10)[["user_id", "ts"]].copy()
    recent["ts"] = recent["ts"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(recent.rename(columns={"user_id": "ユーザーID", "ts": "日時"}),
                 use_container_width=True, hide_index=True)


# ──────────────────────────────────────────
# タブ②：ユーザー別利用状況
# ──────────────────────────────────────────
def _render_user_activity():
    logs = get_event_log(2000)
    df = pd.DataFrame(logs)
    if df.empty:
        st.info("ログデータがありません"); return

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    users = df["user_id"].dropna().unique().tolist()

    summary = []
    for uid in users:
        u_df = df[df["user_id"] == uid]
        n_sessions = int(u_df[u_df["event_type"] == "login"].shape[0])
        last_login = u_df[u_df["event_type"] == "login"]["ts"].max()
        top_feat = (u_df[u_df["event_type"] == "feature_use"]["detail"]
                    .value_counts().index[0]
                    if not u_df[u_df["event_type"] == "feature_use"].empty else "-")
        summary.append({
            "ユーザーID": uid,
            "セッション数": n_sessions,
            "最終ログイン": last_login.strftime("%Y-%m-%d %H:%M") if pd.notna(last_login) else "-",
            "主な利用機能": top_feat,
        })

    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────
# タブ③：操作ヒートマップ
# ──────────────────────────────────────────
def _render_heatmap():
    import plotly.express as px
    logs = get_event_log(2000)
    df = pd.DataFrame(logs)
    if df.empty:
        st.info("ログデータがありません"); return

    pv = df[df["event_type"] == "page_view"]["detail"].value_counts().reset_index()
    pv.columns = ["ページ", "回数"]
    page_label = {
        "pitcher": "投手分析", "batter": "打者分析",
        "leaderboard": "指標順位検索", "game_summary": "投手試合データ",
        "kpi_trend": "KPI推移", "admin": "管理者",
    }
    pv["ページ"] = pv["ページ"].map(page_label).fillna(pv["ページ"])

    fig = px.bar(pv, x="回数", y="ページ", orientation="h",
                 title="ページ別アクセス数", color="回数",
                 color_continuous_scale="Blues")
    fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────
# タブ④：ユーザー管理
# ──────────────────────────────────────────
def _render_user_management():
    st.markdown("#### ユーザー一覧")

    users = get_all_users()
    df_u = pd.DataFrame(users)
    if not df_u.empty:
        df_u["状態"] = df_u["is_active"].apply(lambda x: "✅ 有効" if x else "🚫 無効")
        st.dataframe(
            df_u[["id", "display_name", "role", "position", "状態", "last_login"]]
              .rename(columns={"id":"ID","display_name":"表示名","role":"ロール",
                               "position":"ポジション","last_login":"最終ログイン"}),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # ──── 新規登録フォーム ────
    with st.expander("➕ 新規ユーザー登録"):
        with st.form("add_user_form"):
            new_id   = st.text_input("ログインID（半角英数字）")
            new_name = st.text_input("表示名")
            new_role = st.selectbox("ロール", ["user", "admin"])
            new_pos  = st.selectbox("ポジション区分", ["投手", "野手", "-"])
            new_pw   = st.text_input("初期パスワード（空欄で自動生成）", type="password")
            submitted = st.form_submit_button("登録する", type="primary")
            if submitted:
                if not new_id or not new_name:
                    st.error("IDと表示名は必須です")
                elif new_pw and not validate_password(new_pw):
                    st.error("パスワードは8文字以上で英大文字・英小文字・数字を含む必要があります")
                else:
                    ok, result = add_user(new_id, new_name, new_role, new_pos, new_pw or None)
                    if ok:
                        st.success(f"✅ 登録完了！　ID: {new_id}")
                        if not new_pw:
                            st.info(f"🔑 初期パスワード: `{result}`　← ここでしか確認できません。ユーザーに直接お伝えください。")
                        st.rerun()
                    else:
                        st.error(result)

    # ──── 編集・無効化・削除 ────
    with st.expander("✏️ ユーザー編集 / パスワードリセット / 削除"):
        if not users:
            st.info("ユーザーがいません")
        else:
            uid_sel = st.selectbox("対象ユーザー", [u["id"] for u in users], key="edit_uid")
            target = next((u for u in users if u["id"] == uid_sel), None)
            if target:
                with st.form("edit_user_form"):
                    e_name   = st.text_input("表示名", value=target["display_name"])
                    e_role   = st.selectbox("ロール", ["user","admin"],
                                            index=0 if target["role"]=="user" else 1)
                    e_pos    = st.text_input("ポジション", value=target.get("position",""))
                    e_active = st.checkbox("有効", value=bool(target["is_active"]))
                    if st.form_submit_button("更新"):
                        update_user(uid_sel, e_name, e_role, e_pos, e_active)
                        st.success("更新しました"); st.rerun()

                if st.button(f"🔑 {uid_sel} のパスワードをリセット", key="reset_pw"):
                    new_pw = reset_password(uid_sel)
                    st.success(f"新しい仮パスワード: `{new_pw}`　← ユーザーに直接お伝えください。")

                st.divider()
                st.markdown(f"**⚠️ {uid_sel} を削除**（取り消し不可）")
                confirm_id = st.text_input("削除確認のため対象IDを入力", key="del_confirm")
                if st.button("削除する", type="primary", key="del_btn"):
                    if confirm_id == uid_sel:
                        delete_user(uid_sel)
                        st.success(f"{uid_sel} を削除しました"); st.rerun()
                    else:
                        st.error("IDが一致しません")


# ──────────────────────────────────────────
# タブ⑤：データ管理（CSV）
# ──────────────────────────────────────────
def _render_data_management():
    st.markdown("#### 試合データ管理（CSV）")
    os.makedirs(DATA_DIR, exist_ok=True)

    # アップロード
    uploaded = st.file_uploader(
        "CSVをアップロード（複数選択可）",
        type=["csv"], accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            save_path = os.path.join(DATA_DIR, f.name)
            with open(save_path, "wb") as fp:
                fp.write(f.read())
            log_event("csv_upload", f.name)
        st.success(f"{len(uploaded)} ファイルをアップロードしました")
        st.rerun()

    st.divider()
    st.markdown("#### アップロード済みファイル")

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    if not csv_files:
        st.info("CSVファイルがありません"); return

    for fname in csv_files:
        fpath = os.path.join(DATA_DIR, fname)
        fsize = os.path.getsize(fpath)
        fmtime = pd.Timestamp(os.path.getmtime(fpath), unit="s").strftime("%Y-%m-%d %H:%M")
        size_str = f"{fsize/1024:.1f} KB" if fsize < 1024*1024 else f"{fsize/1024/1024:.1f} MB"

        c1, c2, c3, c4 = st.columns([4, 1.5, 1.5, 1])
        c1.write(fname)
        c2.caption(size_str)
        c3.caption(fmtime)
        with c4:
            with open(fpath, "rb") as fp:
                st.download_button("⬇", fp, file_name=fname, key=f"dl_{fname}", use_container_width=True)

        # 削除ボタン（確認フラグで2段階）
        del_key = f"del_{fname}"
        if st.session_state.get(del_key):
            st.warning(f"「{fname}」を削除しますか？")
            col_y, col_n, _ = st.columns([1, 1, 4])
            if col_y.button("削除", key=f"confirm_{fname}", type="primary"):
                os.remove(fpath)
                log_event("csv_delete", fname)
                st.session_state.pop(del_key, None)
                st.success(f"{fname} を削除しました"); st.rerun()
            if col_n.button("キャンセル", key=f"cancel_{fname}"):
                st.session_state.pop(del_key, None); st.rerun()
        else:
            if st.button("🗑️ 削除", key=f"delreq_{fname}"):
                st.session_state[del_key] = True; st.rerun()
        st.divider()
