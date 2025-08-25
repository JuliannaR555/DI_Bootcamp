@echo off
echo Smart Data Scout - Quick Start
echo ==============================

echo.
echo Setting up Python environment...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Creating necessary directories...
if not exist "data" mkdir data
if not exist "logs" mkdir logs

echo.
echo Setting up environment file...
if not exist ".env" (
    copy ".env.example" ".env"
    echo Please edit .env file with your API keys
)

echo.
echo Creating sample data...
python setup_data.py

echo.
echo ==============================
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: streamlit run app.py
echo.
echo For Groq backend: Set GROQ_API_KEY in .env
echo For Ollama backend: Install Ollama and run 'ollama pull llama3'
echo ==============================

pause
