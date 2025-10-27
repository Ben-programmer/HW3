## ADDED Requirements

### Requirement: Project Context Documented
The repository SHALL include a filled `openspec/project.md` that documents purpose, tech stack, code conventions, testing strategy, git workflow, important constraints, and external dependencies.

#### Scenario: Project context is present for new contributors
- **GIVEN** a new contributor clones the repository
- **WHEN** they read `openspec/project.md`
- **THEN** they see the project's purpose, tech stack, conventions, contacts, and constraints sufficient to start a small change proposal

### Requirement: Project file references OpenSpec usage
The `openspec/project.md` SHALL reference the OpenSpec workflow and link to `openspec/AGENTS.md` so that change authors understand how to create proposals.

#### Scenario: Author needs to create a proposal
- **GIVEN** an author wants to propose a change
- **WHEN** they read `openspec/project.md`
- **THEN** they are pointed to `openspec/AGENTS.md` and the `changes/` directory scaffold for how to proceed
