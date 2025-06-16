import os
import re
import numpy as np
# import plotly.graph_objects as go # 変更点: Plotlyは不要なので削除
import google.generativeai as genai
from dotenv import load_dotenv
import time

# --- 0. APIキーの設定 (.envファイルから) ---
# このセクションは変更ありません
def setup_api_key():
    """
    .envファイルからAPIキーを読み込んで設定する
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("🛑 エラー: APIキーが見つかりません。")
        print("プロジェクトフォルダに .env ファイルを作成し、'GEMINI_API_KEY=あなたのキー' と記述してください。")
        return False

    try:
        genai.configure(api_key=api_key)
        print("✅ APIキーの読み込みに成功しました。")
        return True
    except Exception as e:
        print(f"🛑 APIキーの設定中にエラーが発生しました: {e}")
        return False

# --- 1. LLM (Gemini) を使って動的パターンを生成する関数 ---
# このセクションは変更ありません
def get_dynamic_pattern_from_gemini(prompt: str, grid_size: int) -> str:
    """
    ユーザーのプロンプトから、Geminiを使って形状を生成するPython関数コードを返却させる。
    """
    system_prompt = f"""
    あなたはMITの石井裕先生の「TRANSFORM」を制御するエキスパートです。
    ユーザーからの抽象的な指示に基づき、それを実現するためのPython関数を生成してください。

    # 関数の仕様
    - 関数名は `generate_height_map` としてください。
    - 引数は `x`, `y`, `t` の3つです。
      - `x`, `y`: NumPyのmeshgridから生成された2D配列。テーブルの座標を表す。
      - `t`: 時間を表す浮動小数点数。0.0から1.0の間で変化する。
    - 返り値は、`x`, `y` と同じ形状のNumPy配列 `z` としてください。`z` は各座標のピンの高さを表します。
    - NumPyライブラリ（`np`としてインポート済み）を自由に使用して、複雑で美しいパターンを生成してください。
    - コードブロック（```python ... ```）で、関数定義の部分だけを返してください。

    # 生成例
    ## ユーザー指示: 「中心から広がる穏やかな波紋」
    ```python
    def generate_height_map(x, y, t):
        d = np.sqrt(x**2 + y**2)
        frequency = 5
        amplitude = 0.3
        speed = 10
        z = amplitude * np.sin(d * frequency - t * speed) * (1 - t)
        return z
    ```
    """
    # この関数の内部ロジックは変更ありません
    try:
        generation_config = {'temperature': 0.7, 'top_p': 0.9, 'max_output_tokens': 2048}
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
        full_prompt = f"{system_prompt}\n\n# ユーザー指示: 「{prompt}」"
        print("Geminiに形状生成をリクエスト中...")
        response = model.generate_content(full_prompt)
        if not response or not response.text:
            print("🛑 エラー: Geminiから空のレスポンスが返されました。")
            return get_fallback_pattern(prompt)
        code_match = re.search(r'```python\n(.*?)```', response.text, re.S)
        if code_match:
            print("✅ 関数の生成に成功しました。")
            return code_match.group(1)
        else:
            print("🛑 エラー: Geminiが有効なPythonコードを返しませんでした。")
            print("フォールバック関数を使用します。")
            return get_fallback_pattern(prompt)
    except Exception as e:
        print(f"🛑 Gemini APIエラー: {e}")
        print("フォールバック関数を使用します。")
        return get_fallback_pattern(prompt)

def get_fallback_pattern(prompt: str) -> str:
    # この関数は変更ありません
    prompt_lower = prompt.lower()
    if "波" in prompt or "wave" in prompt_lower:
        return """def generate_height_map(x, y, t):
    d = np.sqrt(x**2 + y**2); frequency = 3; amplitude = 0.4; speed = 8
    z = amplitude * np.sin(d * frequency - t * speed) * np.exp(-d * 0.5); return z"""
    elif "渦" in prompt or "spiral" in prompt_lower:
        return """def generate_height_map(x, y, t):
    angle = np.arctan2(y, x); r = np.sqrt(x**2 + y**2)
    spiral = angle + r * 3 - t * 10; z = 0.3 * np.sin(spiral) * np.exp(-r * 0.8); return z"""
    else:
        return """def generate_height_map(x, y, t):
    z = 0.3 * np.sin(x * 4 + t * 6) * np.cos(y * 3 + t * 8); return z"""

# --- 2. 新しい表示用関数 ---
def display_grid(grid, t):
    """
    2次元の0/1グリッドをコンソールに表示する
    """
    # 画面をクリア (Windowsでは'cls', Mac/Linuxでは'clear')
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"TRANSFORM シミュレーション (t = {t:.2f})")
    print("--------------------")
    for row_index, row in enumerate(grid):
        # アルファベットの行ラベルを生成 (A, B, C, ...)
        row_label = chr(ord('a') + row_index)
        
        # スペースで区切られた数値を表示
        # 記号を変えると見やすくなります (例: '■ ' if cell == 1 else '□ ')
        row_str = ' '.join(str(cell) for cell in row)
        print(f"{row_label} | {row_str}")
        
    print("--------------------")

# --- 3. メインのシミュレーション処理 (改訂版) ---
def main():
    print("🚀 TRANSFORM 組み込みシミュレーター開始")
    
    if not setup_api_key():
        return

    # 変更点: グリッドサイズを小さくして見やすくする
    GRID_SIZE = 10 
    ANIMATION_STEPS = 100 # アニメーションの細かさ

    user_prompt = input("どのような形を生成しますか？ (例: 中心から広がる波紋): ")
    if not user_prompt.strip():
        user_prompt = "美しい波紋"

    generated_code = get_dynamic_pattern_from_gemini(user_prompt, GRID_SIZE)
    if not generated_code:
        print("🛑 パターン生成に失敗しました。")
        return

    try:
        namespace = {}
        exec(generated_code, {"np": np}, namespace)
        generate_height_map = namespace['generate_height_map']
        print("✅ コードの実行に成功しました。")
    except Exception as e:
        print(f"🛑 エラー: 生成されたコードの実行中に問題が発生しました: {e}")
        return

    # 座標の準備
    x = np.linspace(-1, 1, GRID_SIZE)
    y = np.linspace(-1, 1, GRID_SIZE)
    x, y = np.meshgrid(x, y)

    # アニメーションループ
    for i in range(ANIMATION_STEPS):
        t = i / (ANIMATION_STEPS - 1)
        
        # 1. 高さマップを計算
        z = generate_height_map(x, y, t)
        
        # 2. 高さマップを0と1に変換 (しきい値は 0.0)
        #    z > 0.0 の場所が 1 (ON) に、それ以外が 0 (OFF) になります。
        binary_grid = np.where(z > 0.0, 1, 0)
        
        # 3. グリッドをコンソールに表示
        display_grid(binary_grid, t)

        # 4. アニメーションのための待機
        time.sleep(0.05)
    
    print("✅ シミュレーション完了。")


# --- 4. プログラムの実行 ---
if __name__ == '__main__':
    main()