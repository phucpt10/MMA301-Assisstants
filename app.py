import os
import requests  # để bắt lỗi timeout rõ ràng
import streamlit as st
from rag import RAGIndex
from models import LLMProvider
from prompts import SYSTEM_PROMPT
from web_ingest import load_vendor_urls, fetch_vendor_docs

st.set_page_config(page_title="React Native Course Assistant", page_icon="📱", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ Cấu hình")

    # Mật khẩu lớp (nếu APP_PASSWORD có trong Secrets)
    APP_PASSWORD = os.getenv("APP_PASSWORD")
    pw = None
    if APP_PASSWORD:
        pw = st.text_input("Password", type="password", help="Nhập mật khẩu lớp để truy cập.")

    # Provider
    provider_env = os.getenv("PROVIDER", "github").lower()
    provider = st.selectbox("Provider", ["github", "google"], index=0 if provider_env == "github" else 1)

    # Chọn model ngay trên UI (tùy provider)
    gh_default_model = os.getenv("GITHUB_MODELS_MODEL", "gpt-4o-mini")
    gg_default_model = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
    if provider == "github":
        model_name = st.text_input("Model (GitHub Models)", gh_default_model, help="Ví dụ: gpt-4o-mini, gpt-4o")
    else:
        model_name = st.text_input("Model (Gemini)", gg_default_model, help="Ví dụ: gemini-1.5-pro, gemini-1.5-flash")

    # Tùy chọn RAG và tham số
    use_rag = st.checkbox("Dùng RAG (trích tài liệu)", value=True)
    use_local_docs = st.checkbox("Dùng tài liệu nội bộ (data/)", value=True)
    use_vendor_docs = st.checkbox("Dùng nguồn vendor (sources.yaml)", value=True)
    top_k = st.slider("Số đoạn trích dẫn (k)", 1, 8, 4)
    temperature = st.slider("Nhiệt độ (creativity)", 0.0, 1.0, 0.3)

    # Nút test + placeholder hiển thị kết quả
    test_clicked = st.button("🧪 Test kết nối LLM")
    ping_placeholder = st.empty()

    st.markdown("---")
    st.caption("Quản lý API keys trong Streamlit Secrets. Không commit secrets lên GitHub.")

# Chặn truy cập nếu có mật khẩu nhưng chưa nhập đúng
if APP_PASSWORD and (pw or "") != APP_PASSWORD:
    st.info("Ứng dụng yêu cầu mật khẩu lớp. Vui lòng nhập ở sidebar.")
    st.stop()

# Cache resources
@st.cache_resource(show_spinner=True)
def load_index():
    idx = RAGIndex(data_dir="data")
    idx.build()  # chỉ nội bộ trước
    return idx

@st.cache_resource(show_spinner=True)
def load_llm(provider_choice: str, ui_model: str):
    # Áp dụng lựa chọn từ UI vào env trước khi khởi tạo LLM
    os.environ["PROVIDER"] = provider_choice
    if provider_choice == "github":
        os.environ["GITHUB_MODELS_MODEL"] = ui_model
    else:
        os.environ["GOOGLE_MODEL"] = ui_model
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

# Khởi tạo hoặc reload LLM khi đổi provider/model
need_reload_llm = (
    ("llm" not in st.session_state) or
    (st.session_state.get("llm_provider") != provider) or
    (st.session_state.get("llm_model") != model_name)
)
if need_reload_llm:
    with st.spinner("Đang khởi tạo mô hình..."):
        st.session_state.llm = load_llm(provider, model_name)
    st.session_state.llm_provider = provider
    st.session_state.llm_model = model_name

# Test ping
if test_clicked:
    try:
        if hasattr(st.session_state.llm, "ping"):
            with st.spinner("Đang kiểm tra kết nối LLM..."):
                result = st.session_state.llm.ping()
        else:
            result = "❌ ping() chưa được cài trong models.py — vui lòng cập nhật models.py."
    except Exception as e:
        ping_placeholder.error(f"❌ Ping exception: {e}")
    else:
        msg = str(result)
        (ping_placeholder.success if msg.startswith("✅") else ping_placeholder.error)(msg)

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
            try:
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
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
                st.error("LLM timeout. Thử giảm 'Số đoạn trích dẫn (k)', hoặc hỏi ngắn hơn. Bạn cũng có thể tăng timeout bằng cách đặt GITHUB_MODELS_TIMEOUT_READ trong Secrets (ví dụ 180).")
                st.caption(f"Chi tiết: {e}")
                answer, citations = "Xin lỗi, yêu cầu mất quá lâu để xử lý. Vui lòng thử lại.", []
            except Exception as e:
                st.error("Có lỗi khi gọi mô hình. Xem Logs để biết chi tiết.")
                st.caption(f"Chi tiết: {e}")
                answer, citations = "Xin lỗi, tôi không thể trả lời lúc này.", []
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations if use_rag else []
    })
