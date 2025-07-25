FROM aura-base:latest

WORKDIR /app
COPY . .

EXPOSE 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "120"]