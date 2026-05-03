# docs

Project documentation. Three layers, increasing in depth:

## 1. Project-level (start here)

| Doc | What it covers |
|---|---|
| [tern-scoping.md](tern-scoping.md) | Product vision, target users, what tern is and isn't |
| [tern-architecture.md](tern-architecture.md) | System architecture: engine, .bin format, runtime model |
| [tern-model-sizing.md](tern-model-sizing.md) | Parameter budget breakdown, why 5 MB is the target |
| [tern-monorepo.md](tern-monorepo.md) | Repo layout rationale, contributor layers |
| [tern-future-work.md](tern-future-work.md) | Open questions, deferred optimizations |

## 2. Phase-level (how we got here)

| Doc | What it covers |
|---|---|
| [tern-phase1-prototype.md](tern-phase1-prototype.md) | Phase 1 spec: distillation training prototype |
| [tern-phase2-prototype.md](tern-phase2-prototype.md) | Phase 2 spec: Wasm engine build |

## 3. Deep technical (training internals)

In [training/](training/):

| Doc | What it covers |
|---|---|
| [training/design.md](training/design.md) | Distillation prototype design Q&A |
| [training/setup.md](training/setup.md) | Environment setup, dependencies |
| [training/milestones.md](training/milestones.md) | Phase 1 milestone breakdown |
| [training/milestones-phase2.md](training/milestones-phase2.md) | Phase 2 milestone breakdown |
| [training/implementation-guide.md](training/implementation-guide.md) | Reference map: which library to use, minimum theory |
| [training/model-internals.md](training/model-internals.md) | **Canonical:** forward pass + backprop + distillation math |
| [training/scaled-training.md](training/scaled-training.md) | Scaling from POC to robust checkpoint |
| [training/phase-1-conclusion.md](training/phase-1-conclusion.md) | Phase 1 wrap-up and results |
| [training/postmortem-bitlinear-asymmetry.md](training/postmortem-bitlinear-asymmetry.md) | The engine bug Phase 2 caught and how |

## 4. Eval methodology

In [eval/](eval/):

| Doc | What it covers |
|---|---|
| [eval/methodology.md](eval/methodology.md) | What each scorecard metric measures, how to reproduce it |

## Reading order for a new contributor

Cold-start onboarding:

1. [tern-scoping.md](tern-scoping.md) — what tern is, why it exists
2. [tern-architecture.md](tern-architecture.md) — how the system fits together
3. [tern-monorepo.md](tern-monorepo.md) — where everything lives in the repo
4. [training/phase-1-conclusion.md](training/phase-1-conclusion.md) — what was built and how it turned out
5. [training/model-internals.md](training/model-internals.md) — the math (read selectively based on what you're touching)
6. [training/postmortem-bitlinear-asymmetry.md](training/postmortem-bitlinear-asymmetry.md) — important cautionary tale about parity testing

Skip 5 and 6 unless you're touching the engine or training code.