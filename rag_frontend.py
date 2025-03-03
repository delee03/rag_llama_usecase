import streamlit as st
import rag_backend as demo
import os

# 🚀 1. Thiết lập giao diện
st.set_page_config(page_title="Chatbot AI + RAG 🔥")

st.markdown("""
    <h1 style="text-align:center; color: green;">
        Chatbot với LLaMA 3 và RAG từ PDF 📚
    </h1>
""", unsafe_allow_html=True)

# 📥 2. Nút tải file PDF
uploaded_file = st.file_uploader("📂 Tải file PDF để AI học", type="pdf")

if uploaded_file is not None:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF đã được tải lên thành công!")

    # 🛠 3. Tạo FAISS index từ PDF
    with st.spinner("🔄 Đang xử lý tài liệu..."):
        demo.create_pdf_index("uploaded_file.pdf")
    
    st.success("🚀 AI đã học xong nội dung PDF!")

# 📌 4. Kiểm tra nếu chưa có bộ nhớ hội thoại thì khởi tạo
if "memory" not in st.session_state:
    st.session_state.memory = demo.demo_memory()

# 📩 5. Ô nhập liệu cho người dùng
user_input = st.text_area("📝 Nhập câu hỏi của bạn:")

# 🎯 6. Nút gửi tin nhắn
if st.button("Gửi"):
    if not user_input.strip():
        st.warning("⚠️ Vui lòng nhập nội dung!")
    else:
        with st.spinner("🤖 AI đang suy nghĩ..."):
            response = demo.rag_conversation(user_input, st.session_state.memory)
            st.session_state.memory.save_context({"input": user_input}, {"output": response})

        # 📩 7. Hiển thị phản hồi
        st.write("🗨️ **Chatbot:**", response)

