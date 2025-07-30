FROM vgxtj/aura-base:latest
WORKDIR /app
COPY services/ai_server/ .
EXPOSE 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "120"]