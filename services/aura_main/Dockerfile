FROM aura-base:latest

# Copy environment and secrets from project root
COPY .env /app/.env
COPY config/secrets /app/config/secrets

# Copy service code
COPY services/aura_main /app

WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]