"""
Quick installation verification for Local AI Chatbot
"""
try:
    import streamlit
    import langchain
    import huggingface_hub
    print("✅ All dependencies installed correctly!")
    print("🚀 Run: streamlit run main.py")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("📦 Install with: pip install -r requirements.txt")