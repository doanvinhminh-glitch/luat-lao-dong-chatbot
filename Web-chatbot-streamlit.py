import os
import streamlit as object
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cấu hình giao diện Web
st.set_page_config(page_title="Trợ lý Luật Lao Động AI", page_icon="⚖️", layout="centered")
st.title("⚖️ Trợ Lý Ảo Tư Vấn Luật Lao Động 2019")
st.caption("Đồ án tốt nghiệp Khoa học Máy tính - Hệ thống RAG nâng cao")

# 1. Cấu hình API Key (Nhớ thay key thật của bạn vào đây)
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"

# 2. Dùng @st.cache_resource để NẠP MODEL ĐÚNG 1 LẦN (Giải quyết triệt để lỗi chạy lâu)
@st.cache_resource
def init_rag_system():
    DB_DIR = "./faiss_db" # Đổi tên đường dẫn sang thư mục FAISS
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    
    # Nạp FAISS DB thay vì Chroma (Thêm tham số allow_dangerous_deserialization=True để chạy trên Cloud)
    vector_db = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    system_prompt = (
        "Bạn là một trợ lý luật sư chuyên nghiệp tại Việt Nam. "
        "Hãy sử dụng các đoạn văn bản luật được cung cấp sau đây để trả lời câu hỏi của người dùng. "
        "Nếu không biết câu trả lời, hãy nói không biết, không được tự bịa ra câu trả lời.\n\n"
        "Văn bản luật làm căn cứ:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return retriever, prompt

retriever, prompt = init_rag_system()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. Quản lý lịch sử trò chuyện (Chat History) trên giao diện
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
        
        # Gọi mô hình chính (3.5-flash)
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        
        try:
            response = rag_chain.invoke(user_query)
        except Exception as e:
            # Nếu nghẽn mạch (503), tự động chuyển sang bản dự phòng 1.5-flash
            fallback_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
            fallback_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | prompt | fallback_llm | StrOutputParser()
            )
            response = fallback_chain.invoke(user_query)
            response = "⚠️ *(Hệ thống đang dùng máy chủ dự phòng)*\n\n" + response
        
        # Hiển thị câu trả lời lên giao diện web
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
