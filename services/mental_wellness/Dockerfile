FROM vgxtj/aura-base:latest
WORKDIR /app
COPY services/mental_wellness/ .
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]