# AGENTS.md

## Overview

This repository contains the AIC toolkit and multiple project-specific work areas.

- Work from the nearest relevant module or subproject, not from assumptions formed at the superproject root.
- Treat local documentation and workspace instructions as operational context, not optional reading.
- When a task is confined to one area, make the change there and avoid cross-repo spillover.

## Working Model

Use this decision order by default:

1. Identify the real boundary of the task.
2. Read the nearest `AGENTS.md`, `README.md`, and the key entrypoint files for that area.
3. Separate facts from assumptions before proposing architecture or refactors.
4. Prefer the simplest change that preserves current contracts and is easy to verify.
5. Validate at the narrowest meaningful level first, then broaden verification if the change is wide.

## Context Gathering

- Gather context from these sources first, in this order when available:
  1. the nearest local `AGENTS.md`
  2. the nearest `README.md` and workspace-specific setup notes
  3. the key entrypoint, package, or module files for the area being changed
  4. test files covering that area
  5. build, launch, or environment configuration that affects runtime behavior
- For AIC challenge work, also read the relevant files under `docs/` before proposing changes that affect policy behavior, evaluation flow, compliance, or submission assumptions.
- The default challenge-document set is: `docs/overview.md`, `docs/getting_started.md`, `docs/policy.md`, `docs/scoring.md`, `docs/challenge_rules.md`, and any task-specific document such as `docs/scoring_tests.md`, `docs/scene_description.md`, or `docs/submission.md`.
- In multi-workspace tasks, gather context separately for each touched boundary before designing a cross-cutting change.
- Start by locating the owning package, workspace, or service boundary.
- Build context from code layout, architecture notes, runtime commands, and environment assumptions together.
- Do not infer runtime behavior from one layer alone when deployment, bootstrap, or external service wiring may matter.
- For multi-part changes, identify which interfaces or contracts cross module boundaries before editing.
- If a repository contains both research or planning artifacts and implementation code, keep their purposes separate.

## Engineering Principles

- Preserve clear responsibility boundaries between orchestration, application logic, schema or data setup, and infrastructure wiring.
- Prefer explicit flows, deterministic behavior, and debuggable interfaces over clever indirection.
- Keep sources of truth singular and visible; avoid duplicating derived state unless there is a strong operational reason.
- Keep validation and input-shape checks close to system boundaries.
- Do not move business logic into migration, bootstrap, infrastructure, or documentation layers.
- Reuse established patterns and shared helpers before introducing new abstractions.

## Change Discipline

- If behavior changes, update tests, local documentation, and operational guidance in the same iteration.
- If runtime assumptions change, check whether build, deployment, secrets, stage handling, or environment wiring must change too.
- Fix issues at the owning layer instead of documenting workarounds in a higher-level layer.
- For changes that span modules, verify that contracts remain aligned on both sides of the interface.
- Call out meaningful risks, uncertainties, and rollback paths when the impact is non-trivial.

## Verification

- Prefer focused verification first: targeted tests, lint, or local checks for the area you changed.
- Run broader validation when the change affects shared interfaces, startup paths, packaging, or deployment behavior.
- State clearly when complete verification depends on tools, environments, or systems outside the current repository.
- Do not automatically run large end-to-end checks that depend on simulators, hardware loops, or heavyweight external environments unless the user explicitly asks for that run.
- For simulation-dependent validation, prepare the code and the exact manual verification steps, then leave execution to the user.

## Documentation Guidance

- Keep `AGENTS.md` files practical: scope, architecture, commands, environment notes, verification, and change guidance.
- Store durable project context close to the code that owns it.
- Update documentation when commands, architecture boundaries, operational expectations, or user-visible workflows change.
- Use repository-relative paths in documentation and examples unless an absolute path is strictly required by the runtime.
