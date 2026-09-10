"""Demo 2: 文档智能问答（RAG）

整体流程：
  上传 PDF/TXT → 切分文档 → 向量化 → 存入 Chroma → 用户提问 → 检索相关片段 → 大模型生成答案

和 Demo1 的区别：
  Demo1 = 直接问大模型（模型靠自身知识回答）
  Demo2 = 先查你上传的文档，再让大模型根据文档回答（RAG）

运行命令：
  streamlit run demos/demo2-doc-rag/app.py
"""

# ========== 第 1 部分：导入工具 ==========
import io          # 把二进制数据（PDF 字节流）转成可读对象
import os          # 读取 .env 里的 API Key
import tempfile    # 创建临时文件夹（Chroma 向量库需要存磁盘）

import streamlit as st  # 做网页界面的库（把 Python 变成网站）
from dotenv import load_dotenv  # 加载 .env 配置文件

# --- 下面是 RAG 专用的库 ---
from langchain_community.vectorstores import Chroma  # 向量数据库：存文档向量，支持相似度搜索
from langchain_core.documents import Document  # LangChain 的「文档」对象（一段文字 + 元数据）
from langchain_core.messages import HumanMessage, SystemMessage  # 对话消息格式
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 聊天模型 + 向量化模型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 把长文档切成小段
from pypdf import PdfReader  # 读取 PDF 文件内容

load_dotenv()  # 读取项目根目录的 .env 文件

# ========== 第 2 部分：网页基础设置 ==========
st.set_page_config(
    page_title="文档智能问答",  # 浏览器标签页标题
    page_icon="📄",              # 标签页图标
    layout="wide",               # 宽屏布局
)

st.title("文档智能问答")  # 页面大标题
st.caption("Demo 2 · RAG 检索增强生成 · 上传文档后基于内容问答")  # 副标题

# ========== 第 3 部分：检查 API Key ==========
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
    st.error("请在 .env 中配置 OPENAI_API_KEY")  # 网页上显示红色错误
    st.stop()  # 停止运行，不再往下执行

# 从 .env 读取配置；没写就用默认值
base_url = os.getenv(
    "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)  # 百炼 API 地址
chat_model = os.getenv("OPENAI_MODEL", "qwen-plus")  # 聊天用的模型（生成答案）
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-v3")  # 向量化模型（把文字变向量）

# ========== 第 4 部分：Session 状态初始化 ==========
# session_state = Streamlit 的「记忆区」，用户操作之间数据不会丢
# Demo2 比 Demo1 多记几样东西：向量库、文档名、片段数、已处理文件 ID

if "messages" not in st.session_state:
    st.session_state.messages = []  # 对话历史（和 Demo1 一样）
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  # 向量库对象（Chroma），存文档向量
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None  # 当前上传的文档名
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0  # 文档被切成了多少段
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None  # 防止同一个文件重复处理

# ========== 第 5 部分：初始化模型（缓存，避免每次刷新都重新创建）==========
# @st.cache_resource = 只创建一次，后面复用（省时间、省 API 连接开销）

@st.cache_resource
def get_embeddings():
    """Embedding 模型：把一段文字变成数字向量（例如 1024 维数组）"""
    return OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,  # 百炼专用：必须 False，否则 API 报错
    )


@st.cache_resource
def get_llm():
    """Chat 模型：根据检索到的文档片段，生成最终答案"""
    return ChatOpenAI(
        model=chat_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,  # 默认随机性偏低，回答更稳定
    )


embeddings = get_embeddings()  # 向量化模型实例
llm = get_llm()                # 聊天模型实例

# ========== 第 6 部分：工具函数（RAG 核心逻辑）==========

# 发给大模型的「系统提示词」：告诉模型只能根据参考文档回答，不能瞎编
# {context} 是占位符，后面会被检索到的文档片段替换
RAG_SYSTEM_PROMPT = """你是一个文档问答助手。请严格根据下面「参考文档」回答用户问题。

规则：
1. 只使用参考文档中的信息，不要编造
2. 如果文档里没有相关内容，明确说「文档中未找到相关信息」
3. 回答简洁清晰，必要时用条目列出要点

参考文档：
{context}
"""


def load_uploaded_file(uploaded_file) -> str:
    """第 6.1 步：读取用户上传的文件，返回纯文本字符串"""
    name = uploaded_file.name.lower()  # 文件名转小写，方便判断类型
    raw = uploaded_file.getvalue()     # 读取文件的二进制内容

    if name.endswith(".txt"):
        # TXT 可能是 utf-8 或 gbk 编码，逐个尝试
        for encoding in ("utf-8", "gbk", "utf-16"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")  # 都失败就忽略乱码

    if name.endswith(".pdf"):
        # PDF 用 pypdf 逐页提取文字
        reader = PdfReader(io.BytesIO(raw))  # 二进制 → PDF 阅读器
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()  # 多页合并成一段文字

    raise ValueError("仅支持 PDF 和 TXT 文件")


def build_vectorstore(text: str, chunk_size: int, chunk_overlap: int) -> tuple[Chroma, int]:
    """第 6.2 步：切分文档 → 向量化 → 存入 Chroma 向量库

    参数：
      text: 文档全文
      chunk_size: 每段最多多少字（侧边栏可调）
      chunk_overlap: 相邻两段重叠多少字（避免切断语义）
    返回：
      (向量库对象, 切分后的段数)
    """
    # --- 6.2.1 切分：长文档 → 多个小 chunk ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 优先按段落/句号切
    )
    chunks = splitter.split_text(text)  # 得到字符串列表，如 ["第一段...", "第二段..."]

    # --- 6.2.2 包装成 LangChain Document 对象 ---
    docs = [
        Document(page_content=c, metadata={"chunk_id": i})
        for i, c in enumerate(chunks)
    ]

    # --- 6.2.3 向量化 + 存入 Chroma ---
    persist_dir = tempfile.mkdtemp(prefix="demo2_rag_")  # 临时目录存向量
    vs = Chroma.from_documents(
        documents=docs,       # 要存的文档片段
        embedding=embeddings,   # 用哪个模型做向量化
        persist_directory=persist_dir,
    )
    return vs, len(chunks)


def retrieve_context(vectorstore: Chroma, question: str, top_k: int) -> tuple[str, list]:
    """第 6.3 步：检索 — 根据用户问题，找出最相关的 top_k 个文档片段

    原理：把问题也变成向量，在向量库里找「距离最近」的片段（相似度搜索）
    """
    docs = vectorstore.similarity_search(question, k=top_k)  # 返回 Document 列表
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)  # 拼成一段文字
    return context, docs


def answer_question(context: str, question: str, temperature: float) -> str:
    """第 6.4 步：生成 — 把检索到的文档 + 用户问题发给大模型，得到答案"""
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),  # 系统：你是文档助手 + 参考文档
        HumanMessage(content=question),  # 用户：实际问题
    ]
    resp = llm.bind(temperature=temperature).invoke(messages)  # 调用大模型
    return resp.content  # 取出回复文字


def render_sources(sources: list[str]) -> None:
    """在页面上展示 AI 引用了哪些文档片段（方便验证答案来源）"""
    st.caption("引用片段：")
    for i, src in enumerate(sources, 1):
        preview = src[:400] + ("..." if len(src) > 400 else "")  # 太长就截断
        st.markdown(f"**片段 {i}**")
        st.text(preview)


# ========== 第 7 部分：左侧边栏（上传 + 参数设置）==========
with st.sidebar:  # with 表示「在这个区域里放控件」
    st.header("文档上传")
    uploaded = st.file_uploader("上传 PDF 或 TXT", type=["pdf", "txt"])  # 文件上传组件

    st.divider()
    st.header("RAG 参数")
    # 下面 4 个滑块控制 RAG 行为，面试时可以讲这些参数的含义
    chunk_size = st.slider("Chunk Size（切分长度）", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap（重叠长度）", 0, 200, 50, 10)
    top_k = st.slider("Top-K（检索片段数）", 1, 5, 3)  # 每次检索取几段
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)  # 回答随机性

    st.divider()
    if st.button("清空对话"):  # 只清聊天记录，文档还在
        st.session_state.messages = []

    if st.button("清除文档"):  # 文档 + 对话全部清除
        st.session_state.vectorstore = None
        st.session_state.doc_name = None
        st.session_state.chunk_count = 0
        st.session_state.processed_file_id = None
        st.session_state.messages = []

    st.markdown("**技术栈**")
    st.markdown("LangChain · Chroma · Embedding · Streamlit · 百炼 API")

# ========== 第 8 部分：处理上传的文档（索引阶段）==========
# 用户一上传文件，Streamlit 会自动刷新页面，这里检测到新文件就开始处理
if uploaded is not None:
    file_id = f"{uploaded.name}_{uploaded.size}"  # 用「文件名+大小」当唯一标识
    if st.session_state.processed_file_id != file_id:  # 同一个文件不重复处理
        with st.spinner("正在处理文档（切分 + 向量化）..."):  # 加载动画
            try:
                text = load_uploaded_file(uploaded)  # 6.1 读文件
                if not text.strip():
                    st.error("文档内容为空，请换一份文件")
                    st.stop()

                vs, chunk_count = build_vectorstore(text, chunk_size, chunk_overlap)  # 6.2 建向量库
                # 处理结果存进 session_state，后面问答要用
                st.session_state.vectorstore = vs
                st.session_state.doc_name = uploaded.name
                st.session_state.chunk_count = chunk_count
                st.session_state.processed_file_id = file_id
                st.session_state.messages = []  # 新文档 → 清空旧对话
            except Exception as e:
                st.error(f"文档处理失败：{e}")
                st.stop()

# ========== 第 9 部分：显示当前文档状态 ==========
if st.session_state.vectorstore:
    st.info(
        f"当前文档：**{st.session_state.doc_name}** · "
        f"{st.session_state.chunk_count} 个片段 · "
        f"可以开始提问"
    )
else:
    st.warning("请先在左侧上传 PDF 或 TXT 文档。也可以用 `sample_docs/sample.txt` 测试。")

# ========== 第 10 部分：显示对话历史 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # role 是 "user" 或 "assistant"
        st.write(msg["content"])
        if msg.get("sources"):  # AI 回复可能附带引用片段
            render_sources(msg["sources"])

# ========== 第 11 部分：用户提问 → RAG 流程（检索 + 生成）==========
if prompt := st.chat_input("基于文档提问..."):  # 底部输入框，有输入才进入 if
    if not st.session_state.vectorstore:
        st.warning("请先上传文档")
        st.stop()

    # --- 11.1 保存并显示用户的问题 ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- 11.2 检索 + 生成 + 显示 AI 回复 ---
    with st.chat_message("assistant"):
        with st.spinner("检索文档 + 生成回答..."):
            context, docs = retrieve_context(  # 6.3 检索：问题 → 相关片段
                st.session_state.vectorstore, prompt, top_k
            )
            reply = answer_question(context, prompt, temperature)  # 6.4 生成：片段 + 问题 → 答案
            sources = [doc.page_content for doc in docs]  # 提取片段文字，用于展示引用
            st.write(reply)
            render_sources(sources)  # 展示 AI 引用了哪些段落

    # --- 11.3 把 AI 回复存进记忆，刷新页面后还能看到 ---
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
