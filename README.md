# React Native Course Assistant (Streamlit + RAG)

Trợ lý AI cho môn React Native, triển khai trên Streamlit Community Cloud. Hỗ trợ:
- Tài liệu nội bộ (.md/.txt/.pdf) đặt trong `data/`
- Nguồn vendor chính thống từ các URL cấu hình trong `sources.yaml`
- LLM Provider: GitHub Models (Copilot) hoặc Google Generative AI

## 1. Chuẩn bị
- Tạo repo GitHub, copy các file trong dự án.
- Đặt syllabus, mô tả bài tập vào thư mục `data/` (ưu tiên `.md`/`.txt`; `.pdf` vẫn được trích text).
- Sửa `sources.yaml` để phù hợp giáo trình.

## 2. Cấu hình provider & secrets
Vào Streamlit Community Cloud → App → Settings → Secrets rồi dán:
```
PROVIDER = "github"            # hoặc "google"
# GitHub Models
GITHUB_MODELS_TOKEN = "ghp_..." 
GITHUB_MODELS_MODEL = "gpt-4o-mini"
# Google
# GOOGLE_API_KEY = "AIza..."
# GOOGLE_MODEL = "gemini-1.5-pro"
```
Local dev có thể tạo `.streamlit/secrets.toml` tương tự (đừng commit).

## 3. Cài đặt & chạy local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy lên Streamlit Community Cloud
1. Push repo lên GitHub.
2. https://share.streamlit.io → New app → chọn repo + `app.py`.
3. Thêm Secrets như mục (2), rồi Deploy.

## 5. Sử dụng
- Sidebar:
  - Provider: chọn GitHub hoặc Google.
  - Dùng RAG: bật để dùng trích dẫn.
  - Nguồn nội bộ (data/), nguồn vendor (sources.yaml): bật/tắt tuỳ nhu cầu.
  - Nút “Sync nguồn vendor”: tải lại nội dung trang vendor (cache 12h).
- Tải thêm tài liệu: có thể upload .md/.txt/.pdf ngay trong app (không lưu lâu dài sau phiên chạy).
- Chat: hỏi về kiến thức, lab; câu trả lời cố gắng kèm nguồn trích dẫn (file/URL).

## 6. Gợi ý nội dung trong `data/`
- `syllabus.md`: mục tiêu, lịch học, chấm điểm, chính sách.
- `labs/` và `assignments/`: yêu cầu, tiêu chí chấm.
- `faq.md`: câu hỏi thường gặp.
- `notes/`: các ghi chú chuyên đề (Navigation, State, Performance, Testing…).

## 7. Mở rộng
- Thay TF‑IDF bằng vector DB (FAISS/Chroma + embeddings).
- Thêm chức năng tìm kiếm theo từ khoá, lọc theo nguồn.
- Hạn chế “tiết lộ đáp án”: tùy biến `SYSTEM_PROMPT` chặt chẽ hơn hoặc phát hiện câu hỏi “xin đáp án”.

## 8. Bảo mật & đạo đức
- Không commit secrets.
- Tuân thủ robots.txt/vấn đề bản quyền khi trích nội dung vendor; khuyến nghị chỉ lấy trang công khai và dùng ở mức trích dẫn tham khảo học thuật.
