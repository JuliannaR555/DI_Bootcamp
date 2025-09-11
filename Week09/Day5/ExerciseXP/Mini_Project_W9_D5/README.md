# 🤖 Local AI Chatbot with Memory

A fully functional local chatbot powered by **Mistral 7B** that runs entirely on your machine with conversational memory capabilities.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-v0.0.350+-green.svg)

## ✨ Features

- 🧠 **Conversational Memory** - Remembers previous messages in the conversation
- 🎭 **Customizable Personality** - Modify system prompts to change AI behavior
- 🚀 **Local Processing** - Everything runs locally (no external API calls)
- ⚡ **Real-time Streaming** - Watch responses generate in real-time
- 🎨 **Beautiful UI** - Clean Streamlit interface with chat components
- 🔒 **Privacy First** - No data leaves your machine

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd Mini_Project_W9_D5
python -m venv chatbot-env
source chatbot-env/bin/activate  # On Windows: chatbot-env\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Chatbot
```bash
streamlit run main.py
```

Your browser will open automatically at `http://localhost:8501`

**Note**: On first run, the app will download the Mistral 7B model (~3.8GB). This may take 5-10 minutes.

## 🎮 Usage

### Basic Chat
1. Type your message in the chat input
2. Press Enter and watch the AI respond in real-time
3. Ask follow-up questions to test the memory

### Personality Customization
Use the sidebar to try different personalities:
- 🏴‍☠️ **Sarcastic Pirate** - "Ahoy! Let me explain that like a true sea dog!"
- 👨‍🏫 **Concise Tutor** - Brief, educational responses
- 🧠 **Philosopher** - Deep, thoughtful answers
- 🔬 **Science Enthusiast** - Fun explanations with analogies

### Memory Testing
```
You: "My name is Alice and I love programming"
AI: "Nice to meet you Alice! Programming is fascinating..."
You: "What's my name and what do I love?"
AI: "Your name is Alice and you love programming!"
```

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit for web interface
- **AI Engine**: Mistral 7B (quantized Q4_0) via llama.cpp
- **Memory**: LangChain ConversationBufferMemory
- **Inference**: llama-cpp-python for local execution

### Model Information
- **Model**: Mistral-7B-Instruct-v0.1 (quantized)
- **Size**: ~3.8GB (Q4_0 quantization)
- **Context**: 4,096 tokens
- **Source**: TheBloke's optimized GGUF format

## 📋 Requirements

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space (4GB for model + dependencies)
- **OS**: Windows 10+, macOS 10.14+, or Linux

### Python Dependencies
See `requirements.txt` for full list:
- `streamlit>=1.28.0` - Web interface
- `langchain>=0.0.350` - Memory and prompt management
- `llama-cpp-python>=0.2.11` - Local model inference
- `huggingface_hub>=0.17.0` - Model downloading

## 🔧 Troubleshooting

### llama-cpp-python Installation Issues
**Windows users** may need to install from pre-built wheels:
```bash
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp310-cp310-win_amd64.whl
```

### Performance Optimization
- Close unnecessary applications during first run
- Reduce `max_tokens` in config.py for faster responses
- Use SSD storage for better model loading speed

### Memory Issues
- Ensure 8GB+ RAM available
- Clear conversation history if responses become slow
- Restart the app if memory usage grows too high

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the base model
- [TheBloke](https://huggingface.co/TheBloke) for model quantization
- [LangChain](https://langchain.com/) for memory management
- [Streamlit](https://streamlit.io/) for the web interface

---

**⭐ Star this repo if you found it helpful!**