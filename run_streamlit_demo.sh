#!/bin/bash
# Quick script to run both FastAPI backend and Streamlit frontend

echo "🚀 Starting Enrollment Chatbot Demo"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "✓ Using: $PYTHON_CMD"

# Check if streamlit is installed
if ! $PYTHON_CMD -c "import streamlit" &> /dev/null; then
    echo "📦 Installing Streamlit dependencies..."
    pip install -r requirements-streamlit.txt
fi

# Start FastAPI backend in background
echo ""
echo "1️⃣  Starting FastAPI backend..."
cd enrollment_chatbot
$PYTHON_CMD main.py &
API_PID=$!
cd ..

# Wait for API to start
echo "   Waiting for API to initialize..."
sleep 5

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✅ API is running on http://localhost:8000"
else
    echo "   ⚠️  API might not be running. Check for errors above."
fi

# Start Streamlit
echo ""
echo "2️⃣  Starting Streamlit app..."
echo ""
streamlit run streamlit_app.py

# Cleanup when Streamlit exits
echo ""
echo "🛑 Stopping services..."
kill $API_PID 2>/dev/null
echo "✓ Done!"
