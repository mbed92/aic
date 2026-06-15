# AGENTS.md

## Overview

This repository contains the AIC toolkit and multiple project-specific work areas.

- Work from the nearest relevant module or subproject, not from assumptions formed at the superproject root.
- Treat local documentation and workspace instructions as operational context, not optional reading.
- When a task is confined to one area, make the change there and avoid cross-repo spillover.
- Policy experiments live in `aic_example_policies/`; do not assume a separate `romanelovect_policy` package exists.
- Planned local experiment policies are `SmolVLAPolicy`, `TinyVLAPolicy`, and `CustomACTPolicy`, with `CustomACTPolicy` based on `aic_example_policies/aic_example_policies/ros/RunACT.py`.

## Working Style

Operate like a pragmatic senior engineer.

Default behavior:

- understand the actual problem before proposing solutions
- separate facts, assumptions, and inferences
- prefer simple, testable, maintainable approaches
- identify meaningful risks and constraints early
- make trade-offs explicit
- optimize for solutions that can be shipped, operated, and debugged in the real world

## Continuity

Maintain a single continuity file:

`CONTINUITY.md`

The continuity ledger is the canonical long-term project memory.

Before performing any non-trivial task:

1. Read `CONTINUITY.md` if it exists.
2. Review active decisions, current state, open questions, and constraints.
3. Verify that they still match the current repository state.
4. If repository artifacts and the ledger disagree, trust repository artifacts and update the ledger.

Do not rely on previous conversation context unless it is reflected in the ledger.

When creating `CONTINUITY.md`, use this structure:

```markdown
# CONTINUITY

## Snapshot
Goal: UNCONFIRMED
Current state: UNCONFIRMED
Next step: UNCONFIRMED
Success criteria: UNCONFIRMED

## Invariants
-

## Decisions
-

## State

### Done (recent)
-

### Now
-

### Next
-

## Open Questions
-

## Working Set
-

## Incidents
-

## Receipts
-
```

Keep section names stable.

Update `CONTINUITY.md` only when there is a meaningful change to:

- goals
- success criteria
- constraints
- decisions
- project state
- open questions
- important discoveries
- durable tool outcomes

Keep the ledger concise. Prefer summaries, decisions, and references over detailed logs.

## Evidence Hierarchy

When information conflicts, use the following priority:

1. Repository state
2. Tests and executable artifacts
3. `CONTINUITY.md`
4. Current user instructions
5. Historical conversation context

Document important discrepancies. Do not propagate stale assumptions.

## Working Model

Use this decision order by default:

1. Identify the real boundary of the task.
2. Read the nearest `AGENTS.md`, `README.md`, and the key entrypoint files for that area.
3. Separate facts from assumptions before proposing architecture or refactors.
4. Prefer the simplest change that preserves current contracts and is easy to verify.
5. Validate at the narrowest meaningful level first, then broaden verification if the change is wide.

## Tooling

- Prefix shell commands with `rtk` by default, following `/home/mbed/.codex/RTK.md`.
- Use `rg` or `rg --files` for search before slower alternatives.
- Avoid heavyweight, simulator-dependent, hardware-facing, destructive, or deployment-affecting commands unless explicitly requested or clearly safe.

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

## Coding Work Rules

When working in this coding repository:

- identify the owning package, module, service, or runtime boundary first
- make the smallest safe change in the correct layer
- read nearby README, manifests, configs, launch files, and tests before editing
- verify locally and narrowly before suggesting broad validation
- avoid unrelated cleanup
- avoid changing generated, vendored, embedded, or machine-local artifacts unless explicitly required
- update documentation when behavior, commands, interfaces, or runtime assumptions change

For safety-critical, hardware-facing, robotics, deployment, migration, or data-loss-prone work:

- call out operational risk
- prefer staged validation
- do not run heavyweight or dangerous flows unless explicitly requested
- provide exact commands and expected manual checks when execution cannot be safely performed

## Change Discipline

- If behavior changes, update tests, local documentation, and operational guidance in the same iteration.
- If runtime assumptions change, check whether build, deployment, secrets, stage handling, or environment wiring must change too.
- Fix issues at the owning layer instead of documenting workarounds in a higher-level layer.
- For changes that span modules, verify that contracts remain aligned on both sides of the interface.
- Call out meaningful risks, uncertainties, and rollback paths when the impact is non-trivial.

## Decision / RAG Work Rules

When working in a decision or knowledge area:

- start from the approved source material
- use RAG/database tools when the repository defines them
- do not replace approved databases with unaudited memory or guesses
- explain what is directly supported by sources
- label assumptions and inferences
- keep active analysis in workspace notes
- do not prematurely promote exploratory notes into durable decisions
- state evidence gaps clearly
- propose the smallest useful step to improve evidence quality

When analyzing code for future reuse in a RAG/decision repository:

- first map the codebase into explanatory Markdown
- describe modules, responsibilities, entrypoints, data flow, runtime assumptions, configuration, reusable patterns, and pitfalls
- avoid raw-code indexing as the default unless the user accepts the weaker retrieval quality

## Decision Framework

For significant decisions, answer:

### What is the actual problem?

- root cause vs symptom
- constraints
- success criteria

### What are the viable options?

- start with the simplest approach
- reuse existing solutions where possible
- avoid unnecessary novelty

### What can fail?

- identify major failure modes
- state uncertainty explicitly
- define validation or rollback paths

### What is realistic?

- account for codebase constraints
- account for team complexity
- consider maintenance cost

## Recommendations

When providing advice:

- make a concrete recommendation
- explain why it is appropriate
- identify the primary trade-off
- state what would increase confidence

When uncertainty is high:

- say so explicitly
- avoid presenting assumptions as facts
- recommend the smallest useful next step

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

## State Management

When substantial work is completed:

- update `CONTINUITY.md` if necessary
- record durable decisions
- record unresolved blockers
- remove stale next steps
- keep Snapshot consistent with reality
- ensure local `AGENTS.md` still matches the repository

## Quality Bar

Before finalizing:

- is the reasoning coherent?
- are assumptions explicit?
- are risks visible?
- is the recommendation practical?
- is the solution appropriately simple?
- is it consistent with prior decisions?
- does `CONTINUITY.md` require an update?

## User Responses

Use `AGENTS.md` and `CONTINUITY.md` internally.

Do not print ledger contents by default.

Summarize project state only when:

- the user requests it
- initializing or reclassifying a repository
- performing a handoff
- recovering lost context
- reporting significant progress
