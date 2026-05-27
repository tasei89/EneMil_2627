# 野球スタッツ可視化アプリ

社会人野球のスカウティング・試合準備を支援する分析Webアプリケーション。

## 機能

- 🏟️ **投手分析**: K%, BB%, Strike%, 3球追い込み%, 球種別メトリクス, ヒートマップ
- 🏏 **打者分析**: wOBA, ISO, O-Swing%, Whiff%, スプレーチャート
- 📊 **リーダーズボード**: 全選手の指標を一覧・ソート・ハイライト
- 📋 **試合サマリー**: 日付・チームを指定して投手スタッツを確認
- 📈 **KPI推移**: 試合単位・集計単位選択で動的グラフ表示
- ⚙️ **管理者ダッシュボード**: ユーザー管理・CSV管理・利用状況分析

## セットアップ

### 1. リポジトリをクローン
```bash
git clone git@github.com:<org>/<repo>.git
cd baseball-stats-app
```

### 2. パッケージをインストール
```bash
pip install -r requirements.txt
```

### 3. 環境変数を設定
```bash
cp .env.example .env
# .env を編集して SITE_PASSWORD と SECRET_KEY を設定
```

### 4. data/ フォルダを作成
```bash
mkdir -p data db
# 試合CSVを data/ に配置するか、起動後に管理者画面からアップロード
```

### 5. アプリを起動
```bash
streamlit run main.py
```

ブラウザで http://localhost:8501 にアクセスします。

## 初回ログイン

起動時に初期管理者アカウントが自動作成されます。  
コンソールに表示される **初期パスワード** でログインし、すぐに変更してください。

- ID: `admin`
- PW: 起動時にコンソールに表示されます

## デプロイ（Railway）

1. Railway にサインアップ → 「New Project」→「Deploy from GitHub repo」
2. プライベートリポジトリを選択
3. Variables タブで環境変数を設定（`.env.example` 参照）
4. Volumes タブで `/app/data` と `/app/db` をマウント（永続化）
5. デプロイ完了後、自動生成されたURLをチームに共有

## ディレクトリ構成

```
baseball-stats-app/
├── main.py                  # エントリポイント
├── requirements.txt
├── .gitignore
├── .env.example
├── data/                    # ★ gitignore対象（試合CSV）
├── db/                      # ★ gitignore対象（SQLite DB）
└── app/
    ├── auth.py              # 認証・ユーザー管理
    ├── stats/
    │   ├── constants.py     # 定数（球種色, KPI定義等）
    │   └── calculations.py  # スタッツ計算関数
    ├── components/
    │   └── charts.py        # 可視化コンポーネント
    └── pages/
        ├── pitcher_analysis.py
        ├── batter_analysis.py
        ├── leaderboard.py
        ├── game_summary.py
        ├── kpi_trend.py
        └── admin/
            └── dashboard.py
```

## CSVフォーマット

以下の列名に対応しています（大文字小文字を区別しません）。

| 列名 | 内容 |
|------|------|
| 投手名 / 投手氏名 | 投手の氏名 |
| 打者名 | 打者の氏名 |
| 投手チーム | 投手所属チーム |
| 日付 / 試合日 | 試合日（YYYY/MM/DD, YYYY-MM-DD, YYYY年M月D日 いずれも可） |
| 球種 | ストレート, スライダー 等 |
| 球速 | 球速（km/h） |
| 投球コース | 1〜9（ゾーン内）, 10〜25（ゾーン外） |
| 打席結果 | 安打, 三振, 四球 等 |
| 投球結果 | 空振り, ファウル, ボール 等 |
| 打球性質 | ゴロ, フライ, ライナー |
| 打席内球数 | その打席の何球目か |
| 投球後ストライクカウント | 投球後のSカウント |
| アウトカウント / 投球後アウトカウント | アウト数（投球回計算用） |
| 自責点 | 自責点 |
