FROM vgxtj/aura-base:latest
WORKDIR /app
COPY services/ecg_interpreter/ .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]