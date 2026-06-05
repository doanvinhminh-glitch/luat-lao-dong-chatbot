import os
import re
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="Trợ lý Luật Lao Động AI", page_icon="⚖️", layout="wide")

# --- TÍNH NĂNG 3: BỘ LỌC CẤU HÌNH TOÀN BỘ 16 CHƯƠNG BỘ LUẬT LAO ĐỘNG 2019 ---
# Gom toàn bộ dữ liệu cấu hình phạm vi điều của các chương vào Dictionary
CHAPTER_MAPPING = {
    "Chương I: Quy định chung": (1, 10),
    "Chương II: Hợp đồng lao động": (11, 51),
    "Chương III: Giáo dục nghề nghiệp và phát triển kỹ năng nghề": (52, 60),
    "Chương IV: Đối thoại tại nơi làm việc, thương lượng tập thể, thỏa ước lao động tập thể": (61, 89),
    "Chương V: Tiền lương": (90, 104),
    "Chương VI: Thời giờ làm việc, thời giờ nghỉ ngơi": (105, 116),
    "Chương VII: An toàn, vệ sinh lao động": (117, 122),
    "Chương VIII: Kỷ luật lao động, trách nhiệm vật chất": (123, 134),
    "Chương IX: Quy định riêng đối với lao động nữ và bảo đảm bình đẳng giới": (135, 142),
    "Chương X: Quy định riêng đối với lao động chưa thành niên và một số lao động khác": (143, 167),
    "Chương XI: Bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp": (168, 171),
    "Chương XII: Tổ chức đại diện người lao động tại cơ sở": (172, 178),
    "Chương XIII: Giải quyết tranh chấp lao động": (179, 211),
    "Chương XIV: Quản lý nhà nước về lao động": (212, 217),
    "Chương XV: Thanh tra lao động, xử phạt vi phạm pháp luật về lao động": (218, 219),
    "Chương XVI: Điều khoản thi hành": (220, 221)
}

st.sidebar.header("⚖️ BỘ LỌC NÂNG CAO")
st.sidebar.markdown("Giới hạn phạm vi tìm kiếm dữ liệu theo chương.")

# Tự động nạp toàn bộ danh sách Chương từ Dictionary vào Selectbox
selected_filter = st.sidebar.selectbox(
    "Chọn Chương cần tra cứu:",
    ["Tất cả"] + list(CHAPTER_MAPPING.keys())
)
# --- TÍNH NĂNG NÂNG CẤP: DANH SÁCH 8 CÂU HỎI FAQ CHUẨN DOANH NGHIỆP ---
st.sidebar.markdown("---")
st.sidebar.subheader("💡 CÂU HỎI THƯỜNG GẶP (FAQ)")
st.sidebar.markdown("Bấm nhanh các tình huống thực tế để trợ lý AI giải đáp:")

# Mảng chứa 8 câu hỏi kinh điển nhất tại nơi làm việc
faq_questions = [
    {
        "icon": "⏱️", 
        "label": "Thời gian thử việc tối đa?", 
        "text": "Thời gian thử việc tối đa đối với trình độ đại học và các vị trí công việc khác được quy định là bao nhiêu ngày?"
    },
    {
        "icon": "📜", 
        "label": "Các loại hợp đồng lao động?", 
        "text": "Theo Luật Lao động 2019 hiện hành thì có mấy loại hợp đồng lao động và quy định cụ thể của từng loại là gì?"
    },
    {
        "icon": "🎁", 
        "label": "Quy định về tiền thưởng Tết?", 
        "text": "Doanh nghiệp có bắt buộc phải thưởng Tết cho người lao động không? Tiền thưởng Tết được quyết định dựa trên căn cứ nào?"
    },
    {
        "icon": "💵", 
        "label": "Lương tháng 13 tính thế nào?", 
        "text": "Lương tháng 13 có phải là quy định bắt buộc trong luật không và cách tính khoản tiền này như thế nào?"
    },
    {
        "icon": "💰", 
        "label": "Lương làm thêm giờ ngày nghỉ?", 
        "text": "Người lao động làm thêm giờ vào ngày nghỉ hằng tuần hoặc ngày lễ, tết thì được tính lương như thế nào?"
    },
    {
        "icon": "🏖️", 
        "label": "Số ngày nghỉ phép hằng năm?", 
        "text": "Người lao động làm việc đủ năm thì được nghỉ bao nhiêu ngày phép năm hưởng nguyên lương?"
    },
    {
        "icon": "🤰", 
        "label": "Chế độ thai sản của lao động nữ?", 
        "text": "Lao động nữ mang thai và sinh con được hưởng những quyền lợi, chế độ nghỉ ngơi và bảo vệ thai sản như thế nào?"
    },
    {
        "icon": "❌", 
        "label": "Hình thức kỷ luật sa thải?", 
        "text": "Doanh nghiệp được phép áp dụng hình thức kỷ luật sa thải người lao động trong những trường hợp cụ thể nào?"
    }
]

# Vòng lặp tự động tạo các nút bấm xếp dọc đẹp mắt trên Sidebar
for faq in faq_questions:
    if st.sidebar.button(f"{faq['icon']} {faq['label']}", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": faq["text"]})
        st.rerun()

st.title("⚖️ Trợ Lý Ảo Tư Vấn Luật Lao Động 2019")
st.caption("Đồ án tốt nghiệp Khoa học Máy tính - Hệ thống RAG nâng cao toàn diện 16 Chương")

# 1. Quản lý API Key (Tự động thích ứng linh hoạt giữa Local và Cloud)
if "GOOGLE_API_KEY" in st.secrets:
    MY_GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    MY_GOOGLE_API_KEY = "GOOGLE_API_KEY" # Điền key thật của bạn tại đây khi test Local trên VS Code

# 2. Dùng @st.cache_resource để NẠP MODEL ĐÚNG 1 LẦN 
@st.cache_resource
def init_rag_system():
    DB_DIR = "./faiss_db" 
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    vector_db = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    return vector_db

vector_db = init_rag_system()

# --- TÍNH NĂNG 2: THIẾT KẾ PROMPT HỖ TRỢ BỘ NHỚ HỘI THOẠI (CHAT MEMORY) ---
system_prompt = (
    "Bạn là một trợ lý luật sư chuyên nghiệp tại Việt Nam.\n"
    "Hãy sử dụng các đoạn văn bản luật được cung cấp và lịch sử cuộc trò chuyện dưới đây để trả lời câu hỏi của người dùng một cách logic, chính xác.\n"
    "Nếu không biết câu trả lời hoặc dữ liệu luật không đề cập, hãy nói không biết, không được tự bịa ra câu trả lời.\n\n"
    "[LỊCH SỬ TRÒ CHUYỆN]:\n{chat_history}\n\n"
    "[VĂN BẢN LUẬT LÀM CĂN CỨ]:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 3. Quản lý và hiển thị lịch sử trò chuyện (Chat History) trên giao diện
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là Trợ lý AI được huấn luyện dựa trên toàn bộ 16 chương của Luật Lao động 2019. Bạn cần tôi tư vấn điều gì?"}
    ]

# Hiển thị các tin nhắn cũ ra màn hình
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Xử lý khi người dùng gõ câu hỏi mới
if user_query := st.chat_input("Nhập câu hỏi của bạn về luật lao động tại đây..."):
    # Hiển thị câu hỏi của user
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # AI phản hồi
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Thưa bạn, tôi đang tra cứu và phân tích điều luật...")
        
        # Nâng cấu hình k=25 để FAISS quét rộng rãi, bao quát được nhiều chương hơn
        retriever = vector_db.as_retriever(search_kwargs={"k": 25})
        retrieved_docs = retriever.invoke(user_query)
        
        # --- THUẬT TOÁN HẬU LỌC THÔNG MINH THEO TỪNG CHƯƠNG TỰ ĐỘNG ---
        if selected_filter != "Tất cả":
            start_art, end_art = CHAPTER_MAPPING[selected_filter] # Lấy dải phạm vi (Ví dụ: 11 và 51)
            filtered_docs = []
            
            for doc in retrieved_docs:
                article_match = re.search(r"[Đđ]iều\s+(\d+)", doc.page_content)
                if article_match:
                    article_num = int(article_match.group(1))
                    # Kiểm tra xem số điều bóc tách được có nằm trong dải điều của Chương đang chọn không
                    if start_art <= article_num <= end_art:
                        filtered_docs.append(doc)
                else:
                    # Phương án dự phòng: Quét từ khóa tên Chương thuần túy
                    chuong_keyword = selected_filter.split(":")[0].strip()
                    if chuong_keyword in doc.page_content:
                        filtered_docs.append(doc)
                        
            retrieved_docs = filtered_docs[:7]
            
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # ĐÓNG GÓI LỊCH SỬ CHAT (Lấy các câu thoại cũ làm ngữ cảnh)
        history_text = ""
        for m in st.session_state.messages[-5:-1]:
            history_text += f"{m['role'].upper()}: {m['content']}\n"
        
        # Khởi tạo chuỗi xích xử lý LLM
        chain = prompt | ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2, google_api_key=MY_GOOGLE_API_KEY) | StrOutputParser()
        
        try:
            response = chain.invoke({"context": context_text, "chat_history": history_text, "input": user_query})
        except Exception as e:
            # Cơ chế chịu lỗi tự động chuyển sang máy chủ dự phòng
            fallback_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2, google_api_key=MY_GOOGLE_API_KEY)
            fallback_chain = prompt | fallback_llm | StrOutputParser()
            response = fallback_chain.invoke({"context": context_text, "chat_history": history_text, "input": user_query})
            response = "⚠️ *(Hệ thống đang dùng máy chủ dự phòng)*\n\n" + response
        
        # Hiển thị câu trả lời lên giao diện web
        message_placeholder.markdown(response)
        
        # --- TÍNH NĂNG 1: HIỂN THỊ NGUỒN TRÍCH DẪN LUẬT ---
        if retrieved_docs:
            with st.expander("📚 Xem các phân đoạn Luật gốc được hệ thống đối chiếu (Nguồn trích dẫn)"):
                for idx, doc in enumerate(retrieved_docs):
                    source_name = doc.metadata.get("source", f"Phân đoạn luật thứ {idx+1}")
                    st.markdown(f"**[{idx+1}] Cơ sở tài liệu: {source_name}**")
                    st.caption(doc.page_content)
                    st.markdown("---")
                    
        st.session_state.messages.append({"role": "assistant", "content": response})
