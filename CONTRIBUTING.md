# Contributing to SKETCH2SHAPE AI

Thank you for your interest in contributing to **SKETCH2SHAPE AI**! We welcome community contributions, bug reports, feature enhancements, and documentation improvements.

---

## Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors. Please be respectful and constructive in all discussions and pull requests.

---

## Development Setup

### 1. Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18.x or higher
- **Package Managers**: `pip` and `npm`
- **Git**: Installed and configured

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/sketch2shape-ai.git
cd sketch2shape-ai
```

### 3. Backend Setup
Create and activate a virtual environment:
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure your environment variables:
```bash
cp .env.example .env
# Edit .env with your optional API keys (e.g. GROQ_API_KEY)
```

Run the backend server:
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Frontend Setup
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
The frontend application will be available at `http://127.0.0.1:5173`.

---

## Branching & Commit Guidelines

1. **Branch Naming**:
   - Features: `feat/feature-name`
   - Bug fixes: `fix/bug-description`
   - Documentation: `docs/documentation-topic`
   - Performance: `perf/optimization-description`

2. **Commit Messages**: Follow the Conventional Commits specification:
   - `feat: add dual-contouring mesh extraction option`
   - `fix: resolve normal orientation on thin-walled objects`
   - `docs: update setup instructions for WSL2`
   - `test: add unit tests for vehicle blueprint generation`

---

## Coding Standards

### Backend (Python / FastAPI)
- **Formatting & Style**: Follow PEP 8 guidelines.
- **Type Annotations**: Enforce Python type hints on all public functions and Pydantic schemas.
- **Dependencies**: Keep dependencies clean; record any new package additions in `requirements.txt`.
- **Testing**: Add unit tests in `backend/tests/` for any new geometry pipelines or router endpoints.

### Frontend (React / TypeScript / Three.js)
- **TypeScript**: Strict type checking is enabled. Avoid `any`; use strongly-typed interfaces in `frontend/src/types/`.
- **Three.js**: Clean up disposable Three.js resources (geometries, materials, textures) on unmount.
- **Linting & Build**: Ensure `npm run build` passes with zero errors before submitting a pull request.

---

## Running Tests

Run the complete backend test suite:
```bash
pytest backend/tests/ -v
```

Run frontend type checking and production build:
```bash
cd frontend
npm run build
```

---

## Submitting Pull Requests

1. Fork the repository and create your feature branch from `main`.
2. Ensure all backend tests (`pytest`) and frontend builds (`npm run build`) pass cleanly.
3. Verify that no `.env` files, temporary artifacts, or private credentials are included.
4. Open a Pull Request with a clear description of the problem solved, changes made, and visual screenshots or test results where applicable.
