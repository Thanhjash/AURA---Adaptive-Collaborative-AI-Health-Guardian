FROM vgxtj/aura-base:latest
WORKDIR /app
COPY services/radiology_vqa/ .
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]