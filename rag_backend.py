
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
import os

# 🚀 1. Load & Index dữ liệu từ PDF
def create_pdf_index(pdf_path):
    """Load tài liệu PDF, tạo embeddings và lưu vào FAISS"""
    # 1️⃣ Load tài liệu PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2️⃣ Chia nhỏ tài liệu thành các đoạn văn
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # 3️⃣ Tạo embeddings sử dụng Amazon Titan
    embeddings = BedrockEmbeddings(
        region_name="us-east-1",
        credentials_profile_name="default",
        model_id="amazon.titan-embed-text-v2:0"
    )

    # 4️⃣ Tạo FAISS VectorStore và lưu index
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")

    return vectorstore  # Trả về FAISS index để sử dụng ngay

# 🚀 2. Load FAISS index (nếu đã tồn tại)
def load_pdf_index():
    """Load FAISS index nếu đã tồn tại, nếu không sẽ trả về None"""
    embeddings = BedrockEmbeddings(
        region_name="us-east-1",
        credentials_profile_name="default",
        model_id="amazon.titan-embed-text-v2:0"
    )

    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        return None  # Nếu chưa có index, cần chạy create_pdf_index()

# 🚀 3. Kết nối AWS Bedrock LLaMA 3
def demo_chatbot():
    """Tạo kết nối tới LLaMA 3 trên AWS Bedrock"""
    return ChatBedrock(
        credentials_profile_name='default',
        model_id='meta.llama3-70b-instruct-v1:0',
        model_kwargs={
            "max_gen_len": 2048,
            "temperature": 0.3,
            "top_p": 0.9
        }
    )

# 🚀 4. Tạo bộ nhớ hội thoại với LLM
def demo_memory():
    """Tạo Conversation Memory với Bedrock"""
    llm_data = demo_chatbot()  # Gọi LLaMA 3 từ AWS Bedrock
    memory = ConversationSummaryBufferMemory(llm=llm_data, max_token_limit=300)
    return memory

# 🚀 5. Kết hợp FAISS + LLaMA 3 để trả lời câu hỏi (RAG)
def rag_conversation(user_input, memory):
    """Truy vấn FAISS, kết hợp với LLaMA 3 để trả lời câu hỏi"""
    faiss_index = load_pdf_index()
    
    if faiss_index is None:
        return "⚠️ Không tìm thấy index! Vui lòng tải PDF trước."

    # 🔍 1. Tìm tài liệu phù hợp với câu hỏi
    retrieved_docs = faiss_index.similarity_search(user_input, k=3)
    retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])

    # 📌 2. Tạo prompt với thông tin tìm được
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là trợ lý AI chuyên nghiệp. Dưới đây là nội dung từ tài liệu PDF:\n{context}\n"),
        ("human", "Câu hỏi: {question}\nTrả lời:")
    ])

    # 🔄 3. Kết hợp dữ liệu với model
    chain = prompt | demo_chatbot()

    # 🔥 4. Gọi model LLaMA 3 với dữ liệu
    response = chain.invoke({
        "context": retrieved_texts,
        "question": user_input
    })

    # 📝 5. Cập nhật bộ nhớ hội thoại
    memory.save_context({"input": user_input}, {"output": response.content})

    return response.content  # Trả về phản hồi của AI
