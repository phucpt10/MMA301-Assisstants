import os
import streamlit as st
from rag import RAGIndex
from models import LLMProvider
from prompts import SYSTEM_PROMPT
from web_ingest import load_vendor_urls, fetch_vendor_docs

st.set_page_config(page_title="React Native Course Assistant", page_icon="📱", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ Cấu hình")
    provider = os.getenv("PROVIDER", "github").lower()
    provider = st.selectbox("Provider", ["github", "google"], index=0 if provider=="github" else 1)
    use_rag = st.checkbox("Dùng RAG (trích tài liệu)", value=True)
    use_local_docs = st.checkbox("Dùng tài liệu nội bộ (data/)", value=True)
    use_vendor_docs = st.checkbox("Dùng nguồn vendor (sources.yaml)", value=True)
    top_k = st.slider("Số đoạn trích dẫn (k)", 1, 8, 4)
    temperature = st.slider("Nhiệt độ (creativity)", 0.0, 1.0, 0.3)
    st.markdown("---")
    st.caption("Quản lý API keys trong Streamlit Secrets. Không commit secrets lên GitHub.")

# Cache resources
@st.cache_resource(show_spinner=True)
def load_index():
    idx = RAGIndex(data_dir="data")
    idx.build()  # chỉ nội bộ trước
    return idx

@st.cache_resource(show_spinner=True)
def load_llm(provider_choice: str):
    os.environ["PROVIDER"] = provider_choice
    return LLMProvider.from_env()

@st.cache_data(show_spinner=True, ttl=60*60*12)  # cache 12h
def get_vendor_docs():
    urls = load_vendor_urls("sources.yaml")
    return fetch_vendor_docs(urls)

st.title("📱 Trợ lý môn React Native (Giảng viên & Học viên)")
st.caption("Hỏi về khái niệm, best practices, lab/bài tập (gợi ý), quy định môn học...")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    with st.spinner("Đang nạp tài liệu nội bộ và xây dựng chỉ mục..."):
        st.session_state.index = load_index()
if "llm" not in st.session_state:
    with st.spinner("Đang khởi tạo mô hình..."):
        st.session_state.llm = load_llm(provider)

# Optional: upload tài liệu bổ sung ngay trong app
uploaded_files = st.file_uploader(
    "Tải thêm tài liệu (.md/.txt/.pdf) để tăng chất lượng trả lời",
    type=["md", "txt", "pdf"], accept_multiple_files=True
)
if uploaded_files:
    added = st.session_state.index.add_uploaded_files(uploaded_files)
    if added:
        st.success(f"Đã thêm {added} tài liệu tải lên vào chỉ mục.")

# Vendor sync
vendor_docs = []
if use_vendor_docs:
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("🔄 Sync nguồn vendor"):
            st.cache_data.clear()  # làm mới cache vendor docs
            st.experimental_rerun()
    with col2:
        st.caption("Sửa URLs trong sources.yaml nếu muốn bổ sung/giảm bớt nguồn.")
    with st.spinner("Đang tải nguồn vendor..."):
        vendor_docs = get_vendor_docs()
    if vendor_docs:
        st.info(f"Đã nạp {len(vendor_docs)} trang vendor. Sẽ dùng để trích dẫn khi RAG bật.")

# Kết hợp nguồn theo lựa chọn
st.session_state.index.reset_external_docs()
if use_vendor_docs and vendor_docs:
    st.session_state.index.add_external_docs(vendor_docs)
if not use_local_docs:
    st.session_state.index.disable_local_docs()

# Hiển thị lịch sử chat + trích dẫn
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("Nguồn trích dẫn"):
                for c in msg["citations"]:
                    st.write(f"- {c['source']} (score: {c['score']:.3f})")

question = st.chat_input("Đặt câu hỏi về React Native, bài tập, lab, yêu cầu môn học...")

def format_context(chunks):
    parts = []
    for i, ch in enumerate(chunks, 1):
        parts.append(f"[{i}] {ch['text']}\nSource: {ch['source']}")
    return "\n\n".join(parts)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            retrieved = []
            context = ""
            if use_rag:
                retrieved = st.session_state.index.search(question, top_k=top_k)
                context = format_context(retrieved) if retrieved else ""
            answer = st.session_state.llm.generate_answer(
                question=question,
                context=context,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature
            )
            st.markdown(answer)
            citations = [{"source": r["source"], "score": r["score"]} for r in retrieved] if retrieved else []
            if citations:
                with st.expander("Nguồn trích dẫn"):
                    for c in citations:
                        st.write(f"- {c['source']} (score: {c['score']:.3f})")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations if use_rag else []
    })
