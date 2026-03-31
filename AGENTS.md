# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`. Use `src/models/` for grounding model components, `src/datasets/` for dataset adapters, `src/losses/` and `src/metrics/` for training logic, and `src/annotation/` for auto-labeling and placement planning. The `src/annotation/free_bbox/` package contains placement-specific geometry, filtering, clustering, and state tracking. Keep reusable helpers in `src/utils/`. Store runtime configs in `configs/`, entry scripts in `scripts/`, utility CLIs in `tools/`, tests in `tests/`, and longer process notes in `docs/`.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`, then install the package in editable mode with `pip install -e .`. Run the test suite with `pytest tests -v`. Typical entry points are:

- `python scripts/train.py`
- `python scripts/evaluate.py`
- `python scripts/inference.py`
- `python tools/run_placement.py --config configs/annotation/placement.yaml --status --output outputs/placement`

Write generated artifacts to project-local folders such as `outputs/` instead of external temp directories.

## Coding Style & Naming Conventions
Follow Python conventions already used in the repo: 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes, and concise docstrings for files and public functions. Keep config names descriptive, for example `configs/experiments/baseline.yaml` or `configs/annotation/placement_housecat6d.yaml`. Put training or evaluation entrypoints in `scripts/`; place operator/debug utilities in `tools/`.

## Testing Guidelines
Use `pytest` and keep tests under `tests/` with names like `test_model.py` and functions like `test_grounding_model_forward`. Add or update tests whenever changing model outputs, dataset adapters, loss behavior, or placement pipeline branches. Prefer targeted tests first, then run `pytest tests -v` before submitting.

## Commit & Pull Request Guidelines
Recent history follows short prefixed subjects such as `feat:` and `style:`. Keep that format and write specific summaries, for example `feat: add HouseCat6D adapter` or `fix: handle failed placement retries`. Pull requests should include purpose, affected paths, test commands run, and sample outputs or screenshots when changing visualization or placement results.

## Agent-Specific Instructions
Before coding, align on the implementation approach. When adding a CLI script or new flags, include a concrete usage example in the file docstring and in the PR description.
