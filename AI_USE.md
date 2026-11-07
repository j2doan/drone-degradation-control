# AI Usage Disclosure

This file describes how our team used AI tools in this project, how we reviewed and verified the results, and where our
prompts are recorded.

## Policy and standards

We use AI tools as long as we understand the resulting work. Our standard is that every team member maintains a thorough understanding of the repository, its structure, functionality, implementation, and underlying design decisions. Each member must be able to independently explain and justify the work, demonstrate an understanding of how the system operates, and take ownership of their contributions without relying on AI to provide that understanding. We do not claim line-by-line recall of every line of code.

## Tools and models

| Member | Models |
|---|---|
| Jacob Doan | Claude Opus 5.5, Claude Sonnet 5.5 |
| Bryan Zhang | GPT 6.1 Sol |
| Yi-Hsuan Kuo | Claude Opus 5.5 |
| Jeremy Cho | Claude Sonnet 5.5 |

AI was used as coding agents. With the user's permission, they have access to read specified projects, edit files and run
commands such as tests, simulations and lint.

## How we used AI

AI assistants drafted most of the code, tests and documentation, and helped with debugging and refactoring. The research question, the experiment design and the design decisions are our own, informed by feedback from our advisors. We typically started from a written plan of what we wanted. For larger tasks, we documented our plans in plan.md to establish the intended scope and approach. For smaller fixes, adjustments and clarifications, we communicated directly with the AI through conversation. The AI asked clarifying questions, which we answered, and proposed an implementation, which we reviewed, revised or rejected. The AI also reviewed drafts of our reports and suggested rewrites of some passages, which we edited and approved.

All GitHub management was handled exclusively by team members. The AI had no permission to commit or push changes, and no commits or pushes were made by the AI. Experimental code was exchanged among team members in a separate shared scratch repository, so this repository holds only finished milestones and deliverables.

## Review and verification

We did not accept AI output without checking it. Team members read each proposed change, then ran it on their own
machines to confirm it worked. A change was kept only after it passed these checks:
- **Code review:** we read the proposed changes. Where we did not understand or agree with a change, we asked the AI
  to explain it or revise it.
- **Self-tests:** modules that do not depend on a simulator have their own tests, which we ran after each change.
- **Simulation runs:** we ran flights in Isaac Sim and MuJoCo and inspected the output plots to confirm the drone
  behaved as expected.
- **Traceable results:** each run saves its settings and a hash of the code that produced it, so every result can be
  traced to the exact version of the code.
- **Code quality:** all code in this repo passed an automated linter (ruff).

## Prompt appendix

The prompt appendix in this directory archive the team's major prompts used throughout the project. These files document our original intended instructions, plans, approaches, and guidance provided to AI assistants during development.

The appendix is not a collection of raw, unmodified files or prompts. Some documents have been lightly edited and polished with AI assistance for clarity, grammar, and consistency as part of preparing a final deliverable. However, the original content, intent, and wording of the team's plans have been preserved to reflect our own thinking and decision-making.

The appendix is intended to provide transparency into how we planned and directed AI-assisted work, rather than reproduce complete conversation histories or transcripts. It contains the bulk of the prompts and planning documents used throughout the project; however, some prompts may be missing because their original records were not retained or could no longer be recovered.

