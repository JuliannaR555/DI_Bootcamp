# PDF Chatbot with Google Gemini

This project creates an intelligent chatbot that can read and answer questions about PDF documents using Google Gemini's conversational AI and vector search with LangChain and FAISS.

## 🌟 Features

- 📄 Upload and process multiple PDF files simultaneously
- 🔍 Extract and chunk text from PDFs using PyPDF2
- 🧠 Generate embeddings using Google Gemini AI
- 🗃️ Create vector store using FAISS for efficient similarity search
- 💬 Interactive chat interface built with Streamlit
- 📞 User contact information collection system
- 🗂️ Chat history management with clear functionality
- 🔒 Virtual environment support for isolated dependencies

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Google API Key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 1. Create and Activate Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# For Command Prompt users:
# .venv\Scripts\activate.bat
```

### 2. Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

### 3. Configure Google API Key

1. Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Update the `.env` file with your actual API key:

```env
GOOGLE_API_KEY=your_actual_google_api_key_here
```

### 4. Run the Application

```powershell
# Make sure virtual environment is activated
streamlit run app.py
```

The app will be available at: `http://localhost:8501` or `http://localhost:8502`

## 📖 How to Use

### Step 1: Upload PDFs
- Use the sidebar menu to upload one or more PDF files
- Supported formats: PDF files of any size
- Multiple files can be processed simultaneously

### Step 2: Process Documents
- Click "Submit & Process" to extract text and create embeddings
- This may take a few moments depending on file size
- A success message will appear when processing is complete

### Step 3: Ask Questions
- Use the chat interface to ask questions about your documents
- The AI will provide detailed answers based only on the uploaded content
- If information isn't available, the bot will clearly state this

### Step 4: Special Features
- **Contact Request**: Type "call me" to provide your contact information
- **Clear History**: Use the sidebar button to reset the conversation
- **Persistent Storage**: Vector embeddings are saved for future sessions

## 🏗️ Project Structure

```
Mini_Project_W9_D4/
├── .venv/                  # Virtual environment (isolated dependencies)
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python package dependencies
├── .env                   # Environment variables (Google API key)
├── user_info.csv          # User contact information storage
├── faiss_index/           # FAISS vector store (created after first use)
└── README.md              # Project documentation
```

## 🔧 Key Functions & Components

### Core Functions
- **`get_pdf_text(pdf_docs)`**: Extracts text from uploaded PDF files using PyPDF2
- **`get_text_chunks(text)`**: Splits text into manageable chunks with overlap using RecursiveCharacterTextSplitter
- **`get_vector_store(chunks)`**: Creates FAISS vector store with Google Gemini embeddings
- **`get_conversational_chain()`**: Sets up the QA chain with custom prompt template
- **`user_input(user_question)`**: Processes questions and returns AI responses
- **`save_user_info(name, phone, email)`**: Saves contact information to CSV

### Technologies Used
- **🤖 Google Gemini**: Advanced language model and embedding generation
- **🌐 Streamlit**: Interactive web interface and user experience
- **🔗 LangChain**: Document processing and conversational AI chains
- **🔍 FAISS**: Fast similarity search and clustering of dense vectors
- **📄 PyPDF2**: PDF text extraction and processing
- **⚙️ Python-dotenv**: Environment variable management

## 🛠️ Virtual Environment Management

### Activating the Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat

# You should see (.venv) prefix in your terminal
```

### Deactivating the Environment
```powershell
deactivate
```

### Installing New Packages
```powershell
# Always activate environment first
.\.venv\Scripts\Activate.ps1

# Install new packages
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt
```

## 🔍 Troubleshooting

### Common Issues

**1. API Key Problems**
- ✅ Verify your Google API key is valid and active
- ✅ Check that the `.env` file is properly formatted
- ✅ Ensure no extra spaces or quotes around the API key

**2. Module Import Errors**
- ✅ Activate the virtual environment before running
- ✅ Reinstall dependencies: `pip install -r requirements.txt`
- ✅ Check Python version compatibility (3.10+)

**3. PDF Processing Issues**
- ✅ Ensure PDFs contain extractable text (not scanned images)
- ✅ Large PDFs may take longer to process
- ✅ Check file permissions and accessibility

**4. Vector Store Problems**
- ✅ The `faiss_index` folder is created after first PDF processing
- ✅ Delete `faiss_index` folder to reset vector store
- ✅ Ensure sufficient disk space for embeddings

**5. Port Conflicts**
- ✅ If port 8501 is busy, Streamlit will automatically use 8502
- ✅ Check the terminal output for the correct URL
- ✅ Close other Streamlit instances if needed

### Performance Tips
- 📊 **Large PDFs**: Process documents in smaller batches for better performance
- 🔄 **Vector Store**: Reuse existing embeddings by keeping the `faiss_index` folder
- 💾 **Memory**: Monitor system memory usage with very large document sets
- 🔧 **Chunk Size**: Adjust chunk_size in `get_text_chunks()` for different document types

## 🚀 Advanced Usage

### Customizing the Prompt Template
Edit the `prompt_template` in `get_conversational_chain()` to modify AI behavior:

```python
prompt_template = """
Your custom instructions here...
Context: {context}
Question: {question}
Answer:
"""
```

### Adjusting Chunk Parameters
Modify text splitting parameters in `get_text_chunks()`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,    # Adjust for longer/shorter chunks
    chunk_overlap=1000   # Adjust overlap for context preservation
)
```

### Adding New File Types
Extend `get_pdf_text()` to support additional document formats.

## 📊 Data Privacy & Security

- 🔒 **Local Processing**: All document processing happens locally
- 🗃️ **Data Storage**: User information stored in local CSV file
- 🔑 **API Security**: Google API key stored in environment variables
- 🚫 **No Cloud Storage**: Documents are not uploaded to external services

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Update documentation as needed
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure your Google API key is valid
4. Check the terminal output for specific error messages

---

**Happy chatting with your PDFs! 🎉📚**
