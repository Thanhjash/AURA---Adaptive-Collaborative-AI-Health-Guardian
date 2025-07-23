# AURA System Architecture

## Microservices
- **aura_main** (8000): Central orchestrator
- **ecg_interpreter** (8001): OTSN ECG analysis
- **radiology_vqa** (8002): Medical image Q&A
- **mental_wellness** (8003): Mental health support

## Tech Stack
- Backend: Python, FastAPI, Docker
- Frontend: Next.js, React, TypeScript
- AI: MedGemma, Custom OTSN
- Data: Supabase, Firebase
- Infrastructure: AWS EC2, GitHub Actions
