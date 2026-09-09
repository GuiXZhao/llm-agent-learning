"""Demo 1: 多角色 AI 对话助手

通用场景：用户选择 AI 角色，进行多轮对话。
运行：streamlit run demos/demo1-ai-chat-assistant/app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="多角色 AI 对话助手", page_icon="💬", layout="wide")

ROLES = {
    "通用助手": "你是一个 helpful 的 AI 助手，回答简洁清晰。",
    "Python 学习教练": "你是 Python 入门教练，用简单例子解释概念，适合零基础学员。",
    "面试练习官": "你是 AI 应用开发方向的面试官，回答专业简洁，适合 oral 面试准备。",
}

st.title("多角色 AI 对话助手")
st.caption("Demo 1 · 大模型应用实践 · 支持多轮对话与角色切换")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
    st.error("请在 .env 中配置 OPENAI_API_KEY")
    st.stop()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)
model = os.getenv("OPENAI_MODEL", "qwen-plus")

with st.sidebar:
    st.header("设置")
    role = st.selectbox("选择 AI 角色", list(ROLES.keys()))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    st.divider()
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("**技术栈**")
    st.markdown("Python · Streamlit · 百炼 API · Prompt Engineering")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    api_messages = [{"role": "system", "content": ROLES[role]}]
    api_messages.extend(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            resp = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
            )
            reply = resp.choices[0].message.content
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
