# raspberry_pi_main.py (4x4グリッド 最終版)

import os
import re
import time
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# --- ラズパイ用ライブラリ ---
import board
import busio
from adafruit_servokit import ServoKit

# --- 0. APIキーの設定 (.envファイルから) ---
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
def get_dynamic_pattern_from_gemini(prompt: str, grid_size: int) -> str:
    """
    ユーザーのプロンプトから、Geminiを使って形状を生成するPython関数コードを返却させる。
    """
    system_prompt = f"""
    # 指示
    Python関数`generate_height_map(x, y, t)`を生成せよ。
    引数`x`, `y`はnumpy配列、`t`は0.0から1.0のfloat。
    返り値はnumpy配列`z`とせよ。
    ユーザーが指示するパターンを`z`として計算するコードを生成すること。
    numpyは`np`としてインポート済み。
    解説は不要。Pythonのコードブロックのみを返答せよ。

    # 例：ユーザー指示「中心から広がる波紋」
    ```python
    def generate_height_map(x, y, t):
        d = np.sqrt(x**2 + y**2)
        z = 0.3 * np.sin(d * 5 - t * 10) * (1 - t)
        return z
    ```
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        full_prompt = f"{system_prompt}\n\n# ユーザー指示: 「{prompt}」"
        print("Geminiに形状生成をリクエスト中...")
        response = model.generate_content(full_prompt, request_options={"timeout": 60})
        
        if not response.parts:
            print("🛑 エラー: Geminiから応答がありませんでした。フォールバックします。")
            return get_fallback_pattern(prompt)

        code_match = re.search(r'```python\n(.*?)```', response.text, re.S)
        if code_match:
            print("✅ 関数の生成に成功しました。")
            return code_match.group(1)
        else:
            print("🛑 エラー: Geminiが有効なPythonコードを返しませんでした。フォールバックします。")
            return get_fallback_pattern(prompt)
    except Exception as e:
        print(f"🛑 Gemini APIエラー: {e}。フォールバックします。")
        return get_fallback_pattern(prompt)

def get_fallback_pattern(prompt: str) -> str:
    """
    APIエラー時に、プロンプトに応じて基本的なパターンを返す
    """
    prompt_lower = prompt.lower()
    if "渦" in prompt or "spiral" in prompt_lower:
        return """def generate_height_map(x, y, t):
    angle = np.arctan2(y, x); r = np.sqrt(x**2 + y**2)
    spiral = angle * 2 + r * 4 - t * 10; z = 0.4 * np.sin(spiral) * np.exp(-r * 0.5); return z"""
    else: # デフォルトは波紋
        return """def generate_height_map(x, y, t):
    d = np.sqrt(x**2 + y**2); frequency = 4; amplitude = 0.5; speed = 8
    z = amplitude * np.sin(d * frequency - t * speed) * np.exp(-d * 0.5); return z"""


# --- 2. メイン処理 ---
def main():
    if not setup_api_key():
        return

    # --- ハードウェア初期設定 ---
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        kit = ServoKit(channels=16, i2c=i2c)
        print("✅ サーボドライバーに接続しました。")
    except Exception as e:
        print(f"🛑 エラー: サーボドライバーに接続できません。I2Cが有効か、配線を確認してください。")
        print(f"   詳細: {e}")
        return

    # --- 4x4正方形マッピングと各種設定 ---
    GRID_SIZE = 4
    servo_to_grid_map = {
        i: (i // GRID_SIZE, i % GRID_SIZE) for i in range(GRID_SIZE * GRID_SIZE)
    }
    
    ANGLE_UP = 120
    ANGLE_DOWN = 30
    ANIMATION_STEPS = 100

    user_prompt = input("どのような形を生成しますか？ (例: 中心から広がる波紋): ")
    if not user_prompt.strip(): user_prompt = "美しい波紋"

    generated_code = get_dynamic_pattern_from_gemini(user_prompt, GRID_SIZE)
    if not generated_code: return

    try:
        namespace = {}; exec(generated_code, {"np": np}, namespace)
        generate_height_map = namespace['generate_height_map']
    except Exception as e:
        print(f"🛑 エラー: 生成されたコードの実行中に問題が発生しました: {e}"); return

    x = np.linspace(-1, 1, GRID_SIZE); y = np.linspace(-1, 1, GRID_SIZE)
    x, y = np.meshgrid(x, y)

    # --- 制御ループ ---
    try:
        print("▶️ 制御ループ開始 (Ctrl+Cで停止)")
        for i in range(ANIMATION_STEPS):
            t = i / (ANIMATION_STEPS - 1)
            z = generate_height_map(x, y, t)
            binary_grid = np.where(z > 0.0, 1, 0)

            for channel, (row, col) in servo_to_grid_map.items():
                state = binary_grid[row, col]
                target_angle = ANGLE_UP if state == 1 else ANGLE_DOWN
                kit.servo[channel].angle = target_angle
            
            print(f"  t = {t:.2f}", end='\r')
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n⏹️ プログラムを停止します。")
    finally:
        print("サーボを初期位置に戻しています...")
        for channel in servo_to_grid_map.keys():
            kit.servo[channel].angle = ANGLE_DOWN
        print("✅ 完了")

# --- 3. プログラムの実行 ---
if __name__ == '__main__':
    main()