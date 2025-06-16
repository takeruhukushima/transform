import os
import re
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from dotenv import load_dotenv
import time

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

    try:
        # モデルの設定を改善
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_output_tokens': 2048,
        }
        
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=generation_config
        )
        
        full_prompt = f"{system_prompt}\n\n# ユーザー指示: 「{prompt}」"
        
        print("Geminiに形状生成をリクエスト中...")
        print("(最大30秒待機します...)")
        
        # タイムアウト付きでリクエスト
        start_time = time.time()
        response = model.generate_content(full_prompt)
        elapsed_time = time.time() - start_time
        
        print(f"✅ レスポンス取得完了 ({elapsed_time:.1f}秒)")
        
        # レスポンスの内容を確認
        if not response or not response.text:
            print("🛑 エラー: Geminiから空のレスポンスが返されました。")
            return None
            
        print("--- Geminiのレスポンス ---")
        print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        print("--------------------------")

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
    """
    Gemini APIが失敗した場合のフォールバック関数
    """
    print(f"プロンプト「{prompt}」に基づいて基本パターンを生成します。")
    
    # プロンプトに基づいて簡単なパターンを選択
    prompt_lower = prompt.lower()
    
    if "波" in prompt or "wave" in prompt_lower:
        return """def generate_height_map(x, y, t):
    d = np.sqrt(x**2 + y**2)
    frequency = 3
    amplitude = 0.4
    speed = 8
    z = amplitude * np.sin(d * frequency - t * speed) * np.exp(-d * 0.5)
    return z"""
    
    elif "渦" in prompt or "spiral" in prompt_lower:
        return """def generate_height_map(x, y, t):
    angle = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    spiral = angle + r * 3 - t * 10
    z = 0.3 * np.sin(spiral) * np.exp(-r * 0.8)
    return z"""
    
    else:
        # デフォルトのランダムなパターン
        return """def generate_height_map(x, y, t):
    freq1, freq2 = 4, 3
    phase1 = t * 6
    phase2 = t * 8
    z1 = 0.3 * np.sin(x * freq1 + phase1) * np.cos(y * freq2 + phase2)
    z2 = 0.2 * np.sin((x + y) * 2 + t * 5)
    z = z1 + z2
    return z"""

# --- 2. メインのシミュレーション処理 ---
def main():
    print("🚀 TRANSFORM シミュレーター開始")
    
    # APIキーのセットアップは変更なし
    if not setup_api_key():
        return

    # ユーザーからのプロンプト入力も変更なし
    GRID_SIZE = 60
    ANIMATION_STEPS = 50
    user_prompt = input("どのような形を生成しますか？ (例: 中心から広がる波紋, 対角線上を移動する波, etc.): ")
    if not user_prompt.strip():
        print("プロンプトが空です。デフォルトパターンを使用します。")
        user_prompt = "美しい波紋"

    # Geminiからコードを生成する部分も変更なし
    generated_code = get_dynamic_pattern_from_gemini(user_prompt, GRID_SIZE)
    if not generated_code:
        print("🛑 パターン生成に失敗しました。")
        return

    # 生成されたコードの実行部分も変更なし
    try:
        print("生成されたコードを実行中...")
        namespace = {}
        exec(generated_code, {"np": np}, namespace)
        generate_height_map = namespace['generate_height_map']
        print("✅ コードの実行に成功しました。")
    except Exception as e:
        print(f"🛑 エラー: 生成されたコードの実行中に問題が発生しました: {e}")
        print("--- 生成されたコード ---")
        print(generated_code)
        print("------------------------")
        return

    # 3Dアニメーションの準備
    print("3Dアニメーションを準備中...")
    x = np.linspace(-1, 1, GRID_SIZE)
    y = np.linspace(-1, 1, GRID_SIZE)
    x, y = np.meshgrid(x, y)

    frames = []
    for i in range(ANIMATION_STEPS):
        t = i / (ANIMATION_STEPS - 1)
        try:
            z = generate_height_map(x, y, t)
            frames.append(go.Frame(data=[go.Surface(x=x, y=y, z=z)], name=str(i)))
        except Exception as e:
            print(f"🛑 フレーム {i} の生成中にエラー: {e}")
            return

    # 【重要】ここからがプロットオブジェクト `fig` を作成する部分です
    # おそらくこの部分が欠けていました
    initial_z = generate_height_map(x, y, 0)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=initial_z, colorscale='viridis', cmin=-1, cmax=1)])

    fig.frames = frames
    fig.update_layout(
        title=f'LLMによるTRANSFORMシミュレーション: "{user_prompt}"',
        scene=dict(
            zaxis=dict(range=[-1.2, 1.2]),
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
            ]
        }],
        sliders=[{
            'steps': [
                {'method': 'animate', 'label': str(i), 'args': [[str(i)], {'mode': 'immediate'}]}
                for i in range(ANIMATION_STEPS)
            ]
        }]
    )

    print("✅ シミュレーション準備完了。プロットを表示します。")
    fig.show()


# --- 3. プログラムの実行 ---
# この部分も変更なし
if __name__ == '__main__':
    main()