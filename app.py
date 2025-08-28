# -*- coding: utf-8 -*-
"""
Streamlit × LangChain サンプル
- ラジオボタンで専門家の種類（睡眠/育児）を選択
- 選択に応じたシステムメッセージをLLMに渡して回答
- VS Code で実行可:  ターミナルで `streamlit run app.py`
必要パッケージ:
    pip install streamlit langchain langchain-openai tiktoken python-dotenv
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# .env 読み込み
# ----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY が見つかりません。.env を確認してください。")
    st.stop()

# ----------------------------
# UI 設定
# ----------------------------
st.set_page_config(page_title="専門家モード Q&A (睡眠/育児)", page_icon="💬", layout="centered")

st.title("💬 専門家モード Q&A")
st.caption("LangChain + Streamlit | ラジオで『睡眠』『育児』を選ぶと、その専門家として回答します。")

# ----------------------------
# モデル設定
# ----------------------------
model_name = st.sidebar.selectbox("モデル", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
temperature = st.sidebar.slider("温度（創造性）", 0.0, 1.0, 0.2, 0.1)

# ----------------------------
# 専門家プロファイル（システムメッセージ）
# ----------------------------
EXPERT_PROFILES = {
    "睡眠": (
        "あなたは睡眠医学と行動睡眠学に精通した専門家です。"
        "エビデンスに基づき、わかりやすく、実践可能なアドバイスを提供してください。"
        "睡眠衛生、概日リズム、入眠困難/中途覚醒、仮眠、光曝露、カフェイン、運動、"
        "メンタル・ストレスなどを考慮し、必要に応じて医療機関受診の目安も提示します。"
    ),
    "育児": (
        "あなたは発達心理学と小児保健に詳しい育児の専門家です。"
        "年齢（月齢/年齢）に応じた発達段階や個性を尊重しつつ、"
        "安全で現実的な提案を日本の生活環境を踏まえて行ってください。"
        "食事、睡眠、しつけ、情緒の安定、保育・学校連携、親のセルフケア等も配慮します。"
    ),
}

# ----------------------------
# LangChain チェーン作成
# ----------------------------
def build_chain(system_prompt: str) -> "Runnable":
    """ 選択された専門家のシステムメッセージを注入した Chain を返す """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{sys_msg}"),
            ("human", "{question}")
        ]
    )
    llm = ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature)
    chain = prompt | llm | StrOutputParser()
    return chain

# ----------------------------
# 入力UI
# ----------------------------
expert = st.radio("専門家を選択してください：", ["睡眠", "育児"], horizontal=True)
user_text = st.text_area(
    "質問を入力してください（症状や状況、年齢など具体的に書くと精度が上がります）",
    height=140,
    placeholder="例）夜更かしが続いて朝起きられません。スマホを寝る直前まで見てしまいます。改善策は？",
)

# セッションにチャット履歴を保持
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去ログ表示
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# 実行ボタン & 履歴クリア
col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("実行 ▶︎", type="primary")
with col2:
    clear = st.button("履歴クリア 🧹")

if clear:
    st.session_state.messages = []
    st.experimental_rerun()

if run:
    if not user_text.strip():
        st.warning("質問を入力してください。")
        st.stop()

    sys_msg = EXPERT_PROFILES[expert]
    chain = build_chain(sys_msg)

    # ユーザーの発話を表示&保存
    st.session_state.messages.append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    # 推論 & 表示
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            try:
                answer = chain.invoke({"sys_msg": sys_msg, "question": user_text})
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.stop()
            st.markdown(answer)
    # 履歴に追加
    st.session_state.messages.append(("assistant", answer))

# フッター
st.markdown("---")
st.caption("© 専門家モードQ&A | LangChain + Streamlit デモ")

