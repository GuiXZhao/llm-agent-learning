"""Demo 1: 多角色 AI 对话助手

整体流程：
  用户打开网页 → 选角色 → 输入问题 → 程序发给大模型 → 显示回复

运行命令：
  streamlit run demos/demo1-ai-chat-assistant/app.py
"""

# ========== 第 1 部分：导入工具 ==========
import os  # 读取 .env 里的 API Key

import streamlit as st  # 做网页界面的库（把 Python 变成网站）
from dotenv import load_dotenv  # 加载 .env 配置文件
from openai import OpenAI  # 调用大模型 API 的库

load_dotenv()  # 读取项目根目录的 .env 文件

# ========== 第 2 部分：网页基础设置 ==========
st.set_page_config(
    page_title="多角色 AI 对话助手",  # 浏览器标签页标题
    page_icon="💬",                  # 标签页图标
    layout="wide",                   # 宽屏布局
)

# ========== 第 3 部分：定义 3 种 AI 角色 ==========
# 字典：左边是界面显示的名字，右边是发给模型的 system prompt（角色设定）
ROLES = {
    "通用助手": "你是一个 helpful 的 AI 助手，回答简洁清晰。",
    "Python 学习教练": "你是 Python 入门教练，用简单例子解释概念，适合零基础学员。",
    "面试练习官": "你是 AI 应用开发方向的面试官，回答专业简洁，适合 oral 面试准备。",
}

st.title("多角色 AI 对话助手")  # 页面大标题
st.caption("Demo 1 · 大模型应用实践 · 支持多轮对话与角色切换")  # 副标题

# ========== 第 4 部分：检查 API Key ==========
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
    st.error("请在 .env 中配置 OPENAI_API_KEY")  # 网页上显示红色错误
    st.stop()  # 停止运行，不再往下执行

# ========== 第 5 部分：连接大模型服务器 ==========
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),  # 百炼 API 地址；优先读 .env，没有则用默认值
)
model = os.getenv("OPENAI_MODEL", "qwen-plus")  # 使用的模型名

# ========== 第 6 部分：左侧边栏（设置区）==========
with st.sidebar:  # with 表示「在这个区域里放控件」
    st.header("设置")
    role = st.selectbox("选择 AI 角色", list(ROLES.keys()))  # 下拉框选角色
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)  # 滑块调随机性
    st.divider()  # 分隔线
    if st.button("清空对话"):  # 按钮：点击后执行下面代码
        st.session_state.messages = []  # 清空对话记录
        st.rerun()  # 刷新页面
    st.markdown("**技术栈**")
    st.markdown("Python · Streamlit · 百炼 API · Prompt Engineering")

# ========== 第 7 部分：对话记忆（多轮对话的关键）==========
# session_state = Streamlit 的「记忆区」，刷新页面前数据不会丢
if "messages" not in st.session_state:
    st.session_state.messages = []  # 第一次打开时，初始化为空列表

# 把历史对话显示在页面上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # role 是 "user" 或 "assistant"
        st.write(msg["content"])  # 显示消息内容

# ========== 第 8 部分：用户输入 → 调用 API → 显示回复 ==========
if prompt := st.chat_input("输入你的问题..."):  # 底部输入框，有输入才进入 if
    # --- 8.1 保存并显示用户的问题 ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- 8.2 组装发给大模型的 messages ---
    # 第一条是 system（角色设定），后面是所有历史对话
    api_messages = [{"role": "system", "content": ROLES[role]}]
    api_messages.extend(st.session_state.messages)  # 追加 user/assistant 历史

    # --- 8.3 调用 API，等待回复 ---
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):  # 加载动画
            resp = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
            )
            reply = resp.choices[0].message.content  # 取出 AI 回复文字
            st.write(reply)

    # --- 8.4 把 AI 回复也存进记忆，下次追问时模型能看到 ---
    st.session_state.messages.append({"role": "assistant", "content": reply})
