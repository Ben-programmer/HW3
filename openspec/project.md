# Project Context

## Purpose
This repository (HW3) appears to be a small development project that mixes Python code and OpenSpec-driven design work. The primary goal is to maintain clear, machine-validated specifications (OpenSpec) alongside code and use those specs to drive change proposals and implementation.

## Tech Stack
- Python 3.x (used for scripts and application logic)
- Node.js (for CLI tools such as `openspec` and other developer tooling)
- OpenSpec (spec-driven change management stored under `openspec/`)
- Optional/typical: pytest for Python tests, black/isort/mypy for Python linting/type checks; ESLint/Prettier for any JS/TS code

## Project Conventions

### Code Style
- Python: follow Black formatting, isort for imports, and use type hints where practical. Enforce with pre-commit hooks.
- JavaScript/TypeScript: use Prettier and ESLint with a shared config.
- Commit messages: use conventional commits (e.g., `feat:`, `fix:`, `chore:`) to make changelogs easier.

### Architecture Patterns
- Small single-repo layout with a clear `src/` (or top-level modules) and `tests/` directory for Python code.
- Keep OpenSpec artifacts under `openspec/` (specs are the source of truth for behavior).
- Prefer small, single-purpose modules. Extract shared utilities only when duplication or complexity justifies it.

### Testing Strategy
- Unit tests: pytest for Python, aim for fast, deterministic unit tests.
- Integration tests: if external services are required, use recorded fixtures or a local test harness.
- CI: use GitHub Actions (or equivalent) to run linting, type checks, and tests on PRs.

### Git Workflow
- Branching: GitHub Flow — create branches per feature or change (e.g., `add-...`, `fix-...`).
- Pull requests for all changes. Link PRs to OpenSpec `change-id` when implementing proposals.
- Use reviewers and merge when green and approved.

## Domain Context
Add domain-specific notes here for the assistant (examples):
- Primary users and roles
- Expected scale (dev/test vs production)
- Data sensitivity and privacy constraints

_Assumption_: domain specifics are not yet provided. Please populate the section below with details (business rules, key entities, expected workflows) so the AI assistant can author accurate specs and proposals.

## Important Constraints
- Avoid storing secrets in the repository.
- Target compatibility: macOS development environment (local), Python 3.10+ recommended.
- Keep changes small and easily testable; avoid heavy-weight frameworks unless needed.

## External Dependencies
- `openspec` CLI (installed via npm): used to validate and manage proposals.
- Any third-party APIs used by the application should be documented here (auth, rate limits, endpoints).

## Contacts / Ownership
- Repo owner: (please add)
- Primary reviewers: (please add)

---

If you want, I can: populate the Domain Context with information you provide, generate a `pyproject.toml` / `requirements.txt`, or scaffold `tests/` and CI configs.

_Notes on assumptions_: I filled sensible defaults for a small code+specs repo because there was no explicit project metadata. Tell me any corrections and I'll update this file accordingly.
