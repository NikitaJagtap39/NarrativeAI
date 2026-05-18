#!/bin/sh

# Start FastAPI in background, restart if it crashes
uvicorn api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Start Streamlit in background
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
STREAMLIT_PID=$!

# If either process dies, kill the other and exit
# so Docker/EC2 knows the container is unhealthy
wait -n
kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null