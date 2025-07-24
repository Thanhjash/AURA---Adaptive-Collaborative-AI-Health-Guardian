FROM python:3.11-slim  
RUN pip install fastapi uvicorn httpx
COPY . .
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
