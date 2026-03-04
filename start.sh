#!/bin/sh

# Start FastAPI in background
uvicorn api:app --host 0.0.0.0 --port 8001 &

# Wait for FastAPI to be ready before starting Streamlit
echo "⏳ Waiting for FastAPI to start..."
until wget -q --spider http://127.0.0.1:8001/docs 2>/dev/null; do
  sleep 1
done
echo "✅ FastAPI is up!"

# Start Streamlit in foreground
exec streamlit run app.py --server.port=8000 --server.address=0.0.0.0