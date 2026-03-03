#!/bin/sh

# Start FastAPI in background
uvicorn api:app --host 0.0.0.0 --port 8001 &

# Start Streamlit in foreground (replace shell)
exec streamlit run app.py --server.port=7860 --server.address=0.0.0.0