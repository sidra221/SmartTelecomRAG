import os
import qdrant_client
import gradio as gr

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.memory import ChatMemoryBuffer


# =========================
# 1. Load Documents
# =========================
documents = SimpleDirectoryReader("data").load_data()


# =========================
# 2. Embedding Model (Free & Local)
# =========================
embed_model = FastEmbedEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)
Settings.embed_model = embed_model


# =========================
# 3. Vector Store (Qdrant Local)
# =========================
client = qdrant_client.QdrantClient(
    location=":memory:"
)

vector_store = QdrantVectorStore(
    collection_name="telecom_docs",
    client=client,
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
)


# =========================
# 4. LLM (OpenRouter – Free Model)
# =========================
llm = OpenRouter(
    model="mistralai/mistral-7b-instruct",
    max_tokens=256,
    context_window=4096,
)

Settings.llm = llm


# =========================
# 5. Memory
# =========================
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000
)


# =========================
# 6. Chat Engine (STRICT RAG)
# =========================
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="""
You are a telecom customer support assistant.

You MUST answer using ONLY the provided context.
You are NOT allowed to use any external or general knowledge.

If the answer is NOT explicitly found in the context, reply EXACTLY with:
"I do not have this information in my data."

Do not explain, guess, or add extra details.
"""
)


# =========================
# 7. Chat Function (Gradio Compatible)
# =========================
def chat_with_ai(user_input, history):
    if history is None:
        history = []
    
    response = chat_engine.chat(user_input)

    answer = response.response.strip()

    # Confidence indicator (project feature)
    if answer == "I do not have this information in my data.":
        answer += "\n\n🔒 Source: Not found in documents"
    else:
        answer += "\n\n✅ Source: Internal telecom documents"

    # Use dictionary format for Gradio Chatbot (messages format)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})

    return history, ""


# =========================
# 8. Gradio UI
# =========================
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 📡 Smart Telecom RAG Chatbot")

        chatbot = gr.Chatbot(
            label="Telecom Assistant"
        )

        user_input = gr.Textbox(
            placeholder="Ask a telecom-related question...",
            label="Your Question"
        )

        send_btn = gr.Button("Send")

        send_btn.click(
            chat_with_ai,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input],
        )

        user_input.submit(
            chat_with_ai,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input],
        )

    return demo


# =========================
# 9. Run App
# =========================
if __name__ == "__main__":
    build_ui().launch(debug=True)
