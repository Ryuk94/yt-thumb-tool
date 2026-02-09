# Contributing Guide

## Branching Model

Use GitHub Flow with short-lived branches and pull requests into `main`.

### Branch Names

- `feature/grid-ui`
- `feature/backend-api`
- `feature/pattern-detection`
- `feature/onboarding-flow`
- `feature/spike-pricing`
- `feature/bookmark-patterns`
- `hotfix/<short-description>`

Rules:

- lowercase only
- scope prefix + slash
- short, clear description

## Workflow

1. Sync `main` before starting:
   - `git checkout main`
   - `git pull --rebase origin main`
2. Create a feature branch:
   - `git checkout -b feature/<name>`
3. Commit focused changes with clear messages.
4. Push and open a pull request to `main`.
5. Merge only after checks pass.

## Pull Request Checklist

- [ ] Branch follows naming convention.
- [ ] Change is scoped to one deliverable area.
- [ ] Build/tests pass locally.
- [ ] No secrets or local env files included.
- [ ] Screenshots added for UI changes.
