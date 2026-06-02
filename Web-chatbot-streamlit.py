import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CẤU HÌNH GIAO DIỆN WEB (Bản rộng - Wide để chứa Sidebar) ---
st.set_page_config(page_title="Trợ lý Luật Lao Động AI", page_icon="⚖️", layout="wide")

# --- TÍNH NĂNG 3: BỘ LỌC SIDEBAR (METADATA FILTERING) ---
st.sidebar.header("⚖️ BỘ LỌC NÂNG CAO")
st.sidebar.markdown("Giới hạn phạm vi tìm kiếm dữ liệu để tăng độ chính xác.")
selected_filter = st.sidebar.selectbox(
    "Chọn Chương cần tra cứu:",
    ["Tất cả", "Chương II: Hợp đồng lao động", "Chương VII: Thời giờ làm việc, nghỉ ngơi", "Chương XII: Kỷ luật lao động"]
)

st.title("⚖️ Trợ Lý Ảo Tư Vấn Luật Lao Động 2019 VinMinh ăn cức")
st.caption("Đồ án tốt nghiệp Khoa học Máy tính - Hệ thống RAG nâng cao")

# 1. ĐIỀN THẲNG KEY CỦA BẠN VÀO ĐÂY ĐỂ CHẠY TRÊN VS CODE (GIỐNG CODE CŨ)
MY_GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

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
        {"role": "assistant", "content": "Xin chào! Tôi là Trợ lý AI được huấn luyện dựa trên Luật Lao động 2019. Bạn cần tôi tư vấn điều gì?"}
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
        
        # Áp dụng bộ lọc Sidebar vào cấu hình truy vấn của FAISS
        search_kwargs = {"k": 7}
        if selected_filter != "Tất cả":
            search_kwargs["filter"] = {"chuong": selected_filter}
            
        retriever = vector_db.as_retriever(search_kwargs=search_kwargs)
        
        # TRUY VẤN DỮ LIỆU THÔ (Để lấy được metadata làm nguồn trích dẫn)
        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # ĐÓNG GÓI LỊCH SỬ CHAT (Lấy các câu thoại cũ làm ngữ cảnh)
        history_text = ""
        for m in st.session_state.messages[-5:-1]: # Lấy tối đa 4 câu thoại gần nhất
            history_text += f"{m['role'].upper()}: {m['content']}\n"
        
        # Gọi mô hình và truyền trực tiếp biến MY_GOOGLE_API_KEY ở dòng 20 vào
        chain = prompt | ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2, google_api_key=MY_GOOGLE_API_KEY) | StrOutputParser()
        
        try:
            response = chain.invoke({"context": context_text, "chat_history": history_text, "input": user_query})
        except Exception as e:
            # Nếu nghẽn mạch (503), tự động chuyển sang bản dự phòng 3.1-flash-lite
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
