# Transform - 動的パターン生成 & 表示システム

このプロジェクトは、自然言語のプロンプトから動的なパターンを生成し、Raspberry Piを使用して物理的に表示するシステムです。また、Plotlyを使用してパターンを視覚化する機能も備えています。

## 特徴

- 自然言語のプロンプトから動的なパターンを生成
- Raspberry Piを使用した物理的グリッド表示（サーボモーター制御）
- Plotlyを使用したインタラクティブな視覚化
- Google Gemini APIを活用した高度なパターン生成

## セットアップ

### 必要なもの

- Python 3.8以上
- Raspberry Pi（物理グリッド表示用）
- サーボモーターと制御ボード（物理グリッド表示用）
- Google Gemini APIキー

### インストール

1. リポジトリをクローンします:
   ```bash
   git clone [リポジトリURL]
   cd Transform
   ```

2. 仮想環境を作成して有効化:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # または
   # .\venv\Scripts\activate  # Windows
   ```

3. 必要なパッケージをインストール:
   ```bash
   pip install -r requirements.txt
   ```

4. 環境変数の設定:
   `.env` ファイルを作成し、以下のようにAPIキーを設定します:
   ```
   GEMINI_API_KEY=あなたのAPIキー
   ```

## 使い方

### 1. Plotlyで視覚化する場合

```bash
python plotly.py
```

プロンプトが表示されたら、表示したいパターンを説明するテキストを入力します。

### 2. Raspberry Piで物理グリッドを制御する場合

```bash
python raspberry_pi.py
```

プロンプトに従って操作してください。

### 3. コンソールでグリッドを表示する場合

```bash
python grid.py
```

## ファイル構成

- `raspberry_pi.py`: Raspberry Pi用のメインプログラム（物理グリッド制御）
- `plotly.py`: ブラウザでインタラクティブな視覚化を行うプログラム
- `grid.py`: コンソール上でグリッドを表示するプログラム
- `.env`: APIキーなどの環境変数を設定するファイル
- `requirements.txt`: 依存パッケージの一覧

## カスタマイズ

- グリッドサイズを変更するには、各スクリプト内の `GRID_SIZE` 定数を変更します。
- アニメーションの速度を変更するには、`time.sleep()` の値を調整します。

## ライセンス

このプロジェクトはオープンソースです。詳細はLICENSEファイルを参照してください。

## 貢献

バグレポートや機能要望は、Issueトラッカーからお願いします。プルリクエストも歓迎します。

## 作者

[あなたの名前]

---

*このプロジェクトは実験的なものであり、継続的に開発が進められています。*
