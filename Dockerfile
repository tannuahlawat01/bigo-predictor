FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0