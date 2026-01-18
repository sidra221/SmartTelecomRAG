# 📡 Smart Telecom RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for telecom customer support built with LlamaIndex, Qdrant, FastEmbed, and OpenRouter.

## 🌟 Features

- **RAG-based Q&A**: Answers questions using only the provided documents
- **Local Embeddings**: Uses FastEmbed (BAAI/bge-base-en-v1.5) for free, local embeddings
- **Vector Database**: Qdrant for efficient document storage and retrieval
- **LLM Integration**: OpenRouter API for language model inference
- **Web Interface**: Beautiful Gradio UI for easy interaction
- **Strict Mode**: Only answers from provided context, no hallucination

## 🛠️ Tech Stack

- **LlamaIndex**: RAG framework and document processing
- **Qdrant**: Vector database (in-memory for local use)
- **FastEmbed**: Local embedding model (no API costs)
- **OpenRouter**: LLM API access
- **Gradio**: Web UI framework

## 📋 Prerequisites

- Python 3.8+
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))

## 🚀 Installation

1. **Clone the repository** (or navigate to the project directory):
```bash
cd newrag
```

2. **Create a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

1. **Add your documents** to the `data/` folder:
   - `faq.txt` - Frequently asked questions
   - `terms.txt` - Terms and conditions
   - `plans.txt` - Service plans information
   - Add any other `.txt` files you want to include

2. **Set your OpenRouter API key**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your OpenRouter API key:
     ```bash
     OPENROUTER_API_KEY=your-api-key-here
     ```
   - Get your free API key at: https://openrouter.ai

## 🎯 Usage

1. **Activate the virtual environment** (if not already active):
```bash
source venv/bin/activate
```

2. **Run the application**:
```bash
python rag.py
```

3. **Open your browser** and navigate to:
```
http://127.0.0.1:7860
```

4. **Start chatting!** Ask questions about your telecom documents.

## 📁 Project Structure

```
newrag/
├── rag.py              # Main application file
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
├── .env.example       # Example environment file
├── README.md          # This file
└── data/              # Document folder
    ├── faq.txt
    ├── terms.txt
    └── plans.txt
```

## 🔧 How It Works

1. **Document Loading**: Reads all `.txt` files from the `data/` folder
2. **Embedding**: Converts documents to vectors using FastEmbed
3. **Indexing**: Stores vectors in Qdrant vector database
4. **Query Processing**: When you ask a question:
   - Converts question to embedding
   - Searches similar documents in Qdrant
   - Sends context + question to LLM via OpenRouter
   - Returns answer based only on provided context

## ⚠️ Important Notes

- **Strict RAG Mode**: The chatbot is configured to answer ONLY from provided documents
- **No Hallucination**: If information is not found, it will reply: "I do not have this information in my data."
- **In-Memory Qdrant**: Uses `:memory:` location, so data is lost when the app stops
- **API Costs**: OpenRouter API usage may incur costs (check their pricing)

## 🎨 Customization

### Change the LLM Model

Edit `rag.py`:
```python
llm = OpenRouter(
    model="your-preferred-model",  # Change this
    max_tokens=256,
    context_window=4096,
)
```

### Modify System Prompt

Edit the `system_prompt` in `rag.py`:
```python
system_prompt="""
Your custom instructions here...
"""
```

### Change Embedding Model

Edit `rag.py`:
```python
embed_model = FastEmbedEmbedding(
    model_name="your-preferred-model"  # Change this
)
```

## 📝 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests.

## 📧 Support

For questions or issues, please open an issue on the repository.

---

**Built with  using LlamaIndex, Qdrant, FastEmbed, and OpenRouter**
