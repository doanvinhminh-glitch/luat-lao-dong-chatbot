# ⚖️ Trợ Lý Ảo Tư Vấn Luật Lao Động 2019 (Vietnam Labor Law RAG Chatbot)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-LLM-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-blueviolet.svg)

## 📌 Giới thiệu dự án
Đây là hệ thống Chatbot Trợ lý ảo ứng dụng công nghệ **RAG (Retrieval-Augmented Generation)** để tự động hóa công tác tư vấn và tra cứu Bộ luật Lao động 2019 cho doanh nghiệp. Dự án được phát triển nhằm giải quyết bài toán tra cứu thủ công tại bộ phận Hành chính Nhân sự (HR), giúp triệt tiêu hiện tượng "ảo giác AI" (Hallucination) và đảm bảo tính minh bạch tuyệt đối về mặt pháp lý.

🔗 [Trải nghiệm Ứng dụng trực tuyến (Live Demo) tại đây](https://luat-lao-dong-chatbot-vinh-minh.streamlit.app/)

## 🚀 Tính năng nổi bật
* **Hậu lọc thông minh bằng Regex (Strict Post-Filtering):** Tích hợp bộ lọc 16 Chương của Luật Lao động. Thuật toán tự động bóc tách số Điều luật và chặn đứng các ngữ cảnh nhiễu.
* **Truy xuất nguồn minh bạch (Source Traceability):** AI không hoạt động như một "hộp đen". Mỗi câu trả lời đều đính kèm thẻ trích dẫn nguyên văn Điều luật gốc để người dùng đối chiếu.
* **Hệ thống FAQ nghiệp vụ HR:** Tích hợp 8 kịch bản pháp lý thường gặp (Thưởng Tết, Thai sản, Lương làm thêm giờ...) sử dụng kỹ thuật *Prompt Engineering* ẩn để định tuyến truy vấn vector.
* **Cơ chế chịu lỗi & Dự phòng (Fallback Mechanism):** Tự động chuyển đổi mượt mà giữa mô hình `gemini-3.5-flash` và máy chủ dự phòng `gemini-3.1-flash-lite` khi gặp sự cố mạng hoặc quá tải API.
* **Quản trị Dữ liệu Phản hồi (Human-in-the-loop Logging):** Lưu vết toàn bộ đánh giá (Thumbs Up/Down) của người dùng, tự động xuất ra file `user_feedbacks.csv` với chuẩn mã hóa `utf-8-sig` phục vụ báo cáo Excel và Fine-tuning.

## 🛠️ Kiến trúc Công nghệ (Tech Stack)
* **Ngôn ngữ lõi:** Python
* **Giao diện Web:** Streamlit
* **Khung điều phối LLM:** LangChain
* **Vector Database:** Meta FAISS (Local Storage)
* **Mô hình Nhúng (Embedding):** `intfloat/multilingual-e5-base` (HuggingFace)
* **Mô hình Ngôn ngữ lớn (LLM):** Google Gemini Pro / Flash API



```bash
git clone [gihub/luat-lao-dong-chatbot](https://github.com/doanvinhminh-glitch/luat-lao-dong-chatbot)

cd luat-lao-dong-chatbot

pip install -r requirements.txt

.streamlit/secrets.toml

GOOGLE_API_KEY = "Điền-Key-Google-Gemini-Của-Bạn-Vào-Đây"

streamlit run Web-chatbot-streamlit.py



