FROM vgxtj/aura-base:latest
WORKDIR /app
COPY services/aura_main/ .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]