# Architecture Documentation Guide

This document captures our research on how to write effective architecture documentation,
how Claude Code consumes and maintains that documentation, and the specific file structure
and enforcement mechanisms we will use for the ROC project.

## Table of Contents

- [Part 1: Literature Review -- What Architecture Docs Should Contain](#part-1-literature-review----what-architecture-docs-should-contain)
  - [Industry Standards](#industry-standards)
  - [Key Books](#key-books)
  - [Frameworks and Templates](#frameworks-and-templates)
  - [Consensus: Essential vs Optional Sections](#consensus-essential-vs-optional-sections)
  - [Anti-Patterns](#anti-patterns)
  - [Key Principles](#key-principles)
- [Part 2: What Claude Code Needs](#part-2-what-claude-code-needs)
  - [How CLAUDE.md Loading Works](#how-claudemd-loading-works)
  - [The Instruction Budget Problem](#the-instruction-budget-problem)
  - [What to Include vs Exclude](#what-to-include-vs-exclude)
  - [What Claude Can vs Cannot Infer](#what-claude-can-vs-cannot-infer)
  - [Structural Patterns That Help](#structural-patterns-that-help)
  - [Enforcement Hierarchy](#enforcement-hierarchy)
- [Part 3: Our File Structure](#part-3-our-file-structure)
  - [Available Mechanisms](#available-mechanisms)
  - [Chosen Approach: Hybrid](#chosen-approach-hybrid)
  - [What Goes Where](#what-goes-where)
  - [Architecture Doc Sections](#architecture-doc-sections)
- [Part 4: Keeping Docs in Sync](#part-4-keeping-docs-in-sync)
  - [Enforcement Levels](#enforcement-levels)
  - [Hook Configurations](#hook-configurations)
  - [CLAUDE.md Instructions](#claudemd-instructions)
  - [Architecture Fitness Tests](#architecture-fitness-tests)
- [Sources and References](#sources-and-references)

---

## Part 1: Literature Review -- What Architecture Docs Should Contain

### Industry Standards

#### IEEE 1471-2000 / ISO/IEC 42010:2011

The foundational international standard for architecture description of software-intensive
systems. Key concepts:

- **Architecture Description (AD)**: A work product used to express an architecture. Must
  identify the system of interest, its stakeholders, and concerns.
- **Stakeholders and Concerns**: The standard mandates that an AD explicitly identify
  stakeholders (developers, operators, acquirers, users, maintainers) and the concerns each
  has (performance, security, modifiability, etc.). This is a first-class requirement.
- **Viewpoints and Views**: The central organizing mechanism. A viewpoint is a convention for
  constructing and interpreting a view (like a template). A view is a representation of the
  system from a particular perspective, addressing a specific set of concerns. The standard
  requires at least one view but does not prescribe which viewpoints to use.
- **Consistency and Correspondence**: The standard requires that relationships between views
  be documented, including any known inconsistencies. "Correspondence rules" define
  constraints across views.
- **Rationale**: Architecture rationale (why decisions were made) must be recorded.
- **Model Kinds**: Each viewpoint identifies the model kinds (notations, diagrams, tables)
  used within its views.

The 2011 revision added the concept of architecture frameworks (collections of viewpoints for
a domain) and architecture description languages (ADLs).

### Key Books

#### "Documenting Software Architectures: Views and Beyond" (2nd ed., 2010)

Authors: Clements, Bachmann, Bass, Garlan, Ivers, Little, Merson, Nord, Stafford

The definitive reference, often called the "Views and Beyond" approach. Organizes
architectural views into three categories:

**1. Module Views** -- code-time structure (how the system is decomposed into implementation
units):
- Decomposition view: containment/nesting of modules
- Uses view: dependency relationships (which modules use which)
- Generalization view: inheritance/specialization
- Layered view: layers and allowed-to-use relationships
- Data model view: data entities and their relationships (ER diagrams)

**2. Component-and-Connector (C&C) Views** -- runtime structure (processes, threads,
services, data flows):
- Pipe-and-filter view: data streams through processing elements
- Client-server view: request/response interactions
- Publish-subscribe view: event-based communication
- Shared-data view: components accessing shared data stores
- Peer-to-peer view: symmetric interaction patterns

**3. Allocation Views** -- how software maps to non-software structures:
- Deployment view: software elements mapped to hardware/infrastructure
- Implementation view: software elements mapped to file/directory structure
- Work assignment view: software elements mapped to teams

**"Beyond" Section** -- additional documentation that does not fit neatly into a single view:
- Context diagram: system scope and external interfaces
- Architecture overview/summary: quick introduction combining multiple views
- Rationale: why alternatives were rejected
- Behavior documentation: sequence diagrams, state machines, use-case maps
- Interfaces: detailed specification of element interfaces
- Variability guide: where and how the architecture can be configured or extended

**Seven Rules for Sound Documentation:**
1. Write documentation from the reader's point of view
2. Avoid unnecessary repetition
3. Avoid ambiguity
4. Use a standard organization
5. Record rationale
6. Keep documentation current but not too current
7. Review documentation for fitness of purpose

**Choosing Views**: Not all views are needed for every system. The book recommends a
stakeholder-driven approach: list stakeholders, identify their concerns, then select views
that address those concerns.

#### "Software Architecture in Practice" (4th ed., 2021)

Authors: Bass, Clements, Kazman

The most widely used textbook on software architecture. Key contributions:

- **Quality Attribute Scenarios**: A structured way to specify quality requirements with six
  parts: stimulus source, stimulus, artifact, environment, response, response measure. These
  drive architecture decisions and should be documented.
- **Architecture Drivers**: The combination of quality attribute scenarios, functional
  requirements, constraints, and business goals that shape the architecture. The architecture
  makes no sense without understanding what drove it.
- **ADD (Attribute-Driven Design)**: An iterative method that selects architectural drivers,
  chooses tactics/patterns to address them, and documents the results.
- **Tactics Catalog**: Architectural tactics organized by quality attribute (availability,
  performance, security, modifiability, testability, usability). Documenting which tactics
  were chosen and why is part of architecture rationale.
- **Architecture Evaluation**: The ATAM (Architecture Tradeoff Analysis Method) uses
  architecture documentation as input. If the documentation is insufficient for evaluation,
  it is insufficient for development.

#### "Design It! From Programmer to Software Architect" (2017)

Author: Michael Keeling

- **Architecture Haiku**: A one-page summary covering context, functional overview, quality
  attributes, key design decisions, and risks.
- **Architecture Decision Records (ADRs)**: Lightweight documents capturing context,
  decision, status, and consequences. Originally proposed by Michael Nygard in 2011. Now an
  industry best practice.
- **Minimum Viable Architecture**: Document just enough to reduce risk. Avoid
  over-documentation. The right amount depends on the team and project.
- **Fitness Functions**: Architectural properties verified through automated tests. These
  serve as living documentation of architectural constraints.

#### "Just Enough Software Architecture" (2010)

Author: George Fairbanks

- **Risk-driven architecture**: Document and design in proportion to risk.
- **Architecture Hoisting**: Pull essential constraints up to the architecture level; push
  details down. Document only what is architecturally significant.
- **Encapsulation and Designation**: Focus documentation on boundaries (interfaces) and the
  names/identifiers used to refer to elements.

#### "Software Systems Architecture" (2nd ed., 2012)

Authors: Rozanski and Woods

Defines architectural perspectives (cross-cutting quality properties) and viewpoints:

**Core Viewpoints:**
- Functional: what the system does
- Information: how the system stores/manipulates/distributes information
- Concurrency: how functional elements map to concurrent units of execution
- Development: how the system is organized for development
- Deployment: how the system maps to hardware/infrastructure
- Operational: how the system is operated, administered, and supported

**Perspectives** (applied across viewpoints):
Security, Performance and Scalability, Availability and Resilience, Evolution,
Accessibility, Internationalization, Location, Regulation, Usability, Development Resource

### Frameworks and Templates

#### arc42 (Starke and Hruschka)

The most widely used architecture documentation template in Europe. Defines 12 sections:

| # | Section | Purpose | Essential? |
|---|---------|---------|------------|
| 1 | Introduction and Goals | Requirements overview, quality goals, stakeholders | Yes |
| 2 | Architecture Constraints | Technical, organizational, and political constraints | Yes |
| 3 | System Scope and Context | Business context and technical context | Yes |
| 4 | Solution Strategy | Fundamental technology decisions and approaches | Yes |
| 5 | Building Block View | Static decomposition (hierarchical, top-down) | Yes |
| 6 | Runtime View | Important runtime scenarios, interactions, behavior | Yes |
| 7 | Deployment View | Technical infrastructure, mapping to hardware | Situational |
| 8 | Cross-cutting Concepts | Patterns spanning multiple building blocks | Yes |
| 9 | Architecture Decisions | Important decisions with rationale (or ADR links) | Yes |
| 10 | Quality Requirements | Quality tree and quality scenarios | Yes |
| 11 | Risks and Technical Debt | Known risks and technical debt items | Yes |
| 12 | Glossary | Domain and technical terminology | Situational |

#### C4 Model (Simon Brown)

A hierarchical approach to diagramming software architecture at four levels of abstraction:

1. **System Context diagram (Level 1)**: The system as a box surrounded by users and other
   systems. Answers: "What are we building and who/what does it interact with?"
2. **Container diagram (Level 2)**: High-level technical building blocks (applications, data
   stores, microservices, message buses). Answers: "What are the major technical pieces and
   how do they communicate?"
3. **Component diagram (Level 3)**: Internal components of a single container. Answers: "What
   are the key structural building blocks inside this container?"
4. **Code diagram (Level 4)**: Code-level detail (class diagrams, entity diagrams). Generally
   auto-generated and considered optional.

Key principles:
- Each level is independent and meaningful on its own
- Most teams need Levels 1-3 and rarely need Level 4
- Supplement with Dynamic diagrams for behavior and Deployment diagrams for infrastructure
- Notation-agnostic (boxes and arrows, UML, or Structurizr DSL all work)

#### 4+1 View Model (Kruchten, 1995)

One of the earliest and most influential viewpoint frameworks:

1. **Logical View**: Object model, key abstractions, classes and relationships
2. **Process View**: Concurrency, synchronization, distribution
3. **Development View**: Module organization, package structure, build dependencies
4. **Physical View**: Mapping to hardware/networks, deployment topology
5. **+1: Scenarios/Use Cases**: Tie the four views together; validate and illustrate

This model directly influenced IEEE 1471 and remains the conceptual foundation for many later
frameworks.

### Consensus: Essential vs Optional Sections

Synthesizing across all sources:

#### Universally Essential

| Section | Why | Sources |
|---------|-----|---------|
| Context and Scope | Without knowing boundaries, nothing else makes sense | All (C4 L1, arc42 s3, 4+1 scenarios, V&B context diagram) |
| Quality Attribute Goals | Architecture exists to achieve quality goals; without these, decisions have no justification | Bass et al., arc42 s10, IEEE 42010 |
| Key Stakeholders and Concerns | Determines what to document and for whom | IEEE 42010, V&B, arc42 s1 |
| Static Structure / Building Block View | Shows what the system is made of | All (C4 L2-3, arc42 s5, V&B module views, 4+1 logical/development) |
| Runtime Behavior | Shows how elements interact at runtime | All (C4 dynamic, arc42 s6, V&B C&C views, 4+1 process) |
| Key Design Decisions with Rationale | The "why" behind the architecture; most frequently missing, most often needed | IEEE 42010, arc42 s9, V&B, Keeling, Bass et al. |
| Cross-cutting Concerns | Patterns spanning multiple components | arc42 s8, Rozanski & Woods, V&B |

#### Strongly Recommended

| Section | Why | Sources |
|---------|-----|---------|
| Constraints | Frames what is negotiable vs fixed | arc42 s2, Bass et al. |
| Solution Strategy | High-level approach before diving into details | arc42 s4 |
| Interfaces | Contracts between components and with external systems | V&B, Fairbanks |
| Risks and Technical Debt | Honest accounting of known problems | arc42 s11, Keeling |
| Deployment View | How software maps to infrastructure | C4 deployment, 4+1 physical, V&B, arc42 s7 |
| Glossary / Terminology | Ensures shared vocabulary | arc42 s12, DDD |

#### Situational / Optional

| Section | When needed | Sources |
|---------|-------------|---------|
| Code-level diagrams (C4 Level 4) | Rarely; auto-generate if needed | C4 |
| Detailed data models | When data structures are architecturally significant | V&B |
| Work assignment view | Multi-team projects | V&B |
| Variability guide | Product lines, highly configurable systems | V&B |
| Development/build view | Complex build systems, many repositories | 4+1, V&B |
| Performance models | Performance-critical systems | Rozanski & Woods |

### Anti-Patterns

These are the common failure modes identified across the literature:

**From "Documenting Software Architectures" (Clements et al.):**
- **Big Ball of Mud documentation**: No clear views, no separation of concerns. Everything
  dumped into one massive document with no organizing principle.
- **The Dusty Tome**: A 300-page document written once and never updated. Immediately stale.
  Nobody reads it.
- **All Diagrams, No Prose**: Pretty pictures without explanation of what the elements are,
  what they do, and why they are there. Diagrams without supporting text are ambiguous.
- **No Rationale**: Showing what the architecture IS without explaining WHY. The single most
  common and damaging omission. Without rationale, nobody knows which decisions are
  fundamental vs accidental.
- **Technology-First Documentation**: Documenting the technology stack without explaining the
  architectural structures, responsibilities, and interactions. Technology choices are
  decisions, not architecture.
- **Audience Amnesia**: Writing for yourself instead of your readers.

**From Simon Brown (C4 Model):**
- **The Model-Code Gap**: Architecture diagrams that do not match the actual code. Happens
  when diagrams are created once and never maintained.
- **Inconsistent Abstraction Levels**: Mixing high-level containers with low-level classes in
  the same diagram. Each diagram should operate at one level of abstraction.
- **Missing Legends/Keys**: Diagrams using shapes, colors, and line styles without explaining
  what they mean.
- **Implied Relationships**: Lines between boxes with no labels. What does the arrow mean --
  data flow? Dependency? Runtime call? Control flow?

**From Starke (arc42):**
- **Perfectionism**: Trying to document everything leads to documenting nothing.
- **Architecture Astronautics**: Over-abstraction. Documenting theoretical frameworks instead
  of the actual system.
- **Copy-Paste Syndrome**: Reusing another project's architecture document by changing names.

**From Fairbanks ("Just Enough Software Architecture"):**
- **Uniform Documentation Density**: Spending equal effort on all parts regardless of risk or
  complexity. High-risk areas deserve deep documentation; low-risk areas need only a sketch.
- **Ignoring the Boring Parts**: Not documenting cross-cutting concerns (error handling,
  configuration, logging) because they feel mundane. These are often where bugs and confusion
  accumulate.

**General (multiple sources):**
- **Orphaned ADRs**: Decision records that exist but are not linked to the architecture
  documentation.
- **UML-or-Nothing**: Insisting on formal UML when informal boxes-and-arrows communicate
  more effectively. The goal is communication, not notational purity.
- **Documentation as Afterthought**: Writing architecture documentation after development is
  complete. By then, decisions and rationale are forgotten.
- **No Versioning**: Architecture evolves but documentation does not track which version is
  described.

### Key Principles

Synthesizing across all sources, the core principles are:

1. **Stakeholder-driven**: Start by identifying who will read the documentation and what they
   need to know.
2. **Multiple views, not one diagram**: No single diagram can capture an architecture. Use
   complementary views at different levels of abstraction.
3. **Rationale is non-negotiable**: Documenting WHY is more important than documenting WHAT.
   The system itself shows what it is; only documentation can explain why.
4. **Quality attributes drive architecture**: Document the quality attribute requirements that
   shaped decisions. Without these, the architecture looks arbitrary.
5. **Just enough, kept current**: Document in proportion to risk and complexity. A living,
   lightweight document beats a comprehensive, stale one.
6. **Diagrams plus prose**: Every diagram needs accompanying text explaining what it shows,
   what the elements are, and what relationships mean.
7. **Separate concerns into views**: Static structure, runtime behavior, deployment, and
   cross-cutting concerns each deserve distinct treatment.

---

## Part 2: What Claude Code Needs

### How CLAUDE.md Loading Works

Claude Code loads project context through a tiered system. Understanding the loading order
is critical for deciding what goes where.

**Loading order** (position in context window, earliest to latest):
1. System prompt (~4,200 tokens) -- invisible to user
2. Auto memory (MEMORY.md) (~680 tokens)
3. Environment info (~280 tokens)
4. MCP tool names (~120 tokens, deferred)
5. Skill descriptions (~450 tokens)
6. `~/.claude/CLAUDE.md` (~320 tokens) -- personal/global instructions
7. Project `CLAUDE.md` (~1,800 tokens) -- the main project instructions
8. `.claude/rules/` files (unconditional ones)
9. User prompt
10. Path-specific rules load on-demand when Claude reads matching files

**Loading behavior by location:**

| Location | When Loaded |
|----------|-------------|
| Managed policy CLAUDE.md (system-level) | Every session, cannot be excluded |
| `./CLAUDE.md` (project root) | Full load at session start |
| `~/.claude/CLAUDE.md` (user global) | Every session |
| Subdirectory CLAUDE.md files | On-demand when Claude reads files in that directory |
| `.claude/rules/*.md` | On-demand based on glob patterns in YAML frontmatter |

CLAUDE.md files in the directory hierarchy above the working directory are loaded in full at
launch. CLAUDE.md files in subdirectories load on demand when Claude reads files in those
subdirectories.

**Compaction behavior**: CLAUDE.md fully survives compaction (`/compact`). After compaction,
Claude re-reads CLAUDE.md from disk and re-injects it fresh. Instructions given only in
conversation will be lost. You can add a "When compacting, preserve:" section to CLAUDE.md
to control what is preserved.

### The Instruction Budget Problem

Frontier LLMs can reliably follow approximately 150-200 instructions. Claude Code's system
prompt already consumes roughly 50 of those slots, leaving 100-150 for your CLAUDE.md.
Beyond that threshold, compliance degrades uniformly -- every low-value rule dilutes the
compliance probability of every high-value rule.

Anthropic's official recommendation: target under 200 lines per CLAUDE.md file. Community
consensus puts the ideal range at 60-200 lines, with 300 as an absolute ceiling.

A poorly-structured CLAUDE.md can perform worse than having none at all (per analysis
referencing arxiv 2602.11988).

### What to Include vs Exclude

**INCLUDE (things Claude cannot infer from code):**
- Bash commands Claude cannot guess (build, test, deploy)
- Code style rules that differ from defaults (non-standard conventions only)
- Testing instructions and preferred test runners
- Repository etiquette (branch naming, PR conventions)
- Architectural decisions specific to your project
- Developer environment quirks (required env vars, ports, services)
- Common gotchas or non-obvious behaviors
- Project purpose and tech stack (1-2 lines)
- Directory structure map (what lives where)

**EXCLUDE (things Claude can figure out or that waste tokens):**
- Anything Claude can figure out by reading code
- Standard language conventions Claude already knows
- Detailed API documentation (link to docs instead)
- Information that changes frequently
- Long explanations or tutorials
- File-by-file descriptions of the codebase
- Self-evident practices like "write clean code"
- Code style rules enforced by linters (let the linter enforce them)

The litmus test from Anthropic: "For each line, ask: Would removing this cause Claude to
make mistakes? If not, cut it."

### What Claude Can vs Cannot Infer

**Claude CAN infer from code:**
- General language patterns and idioms
- Common library usage and standard APIs
- Standard testing frameworks and their patterns
- Basic project structure from file names and imports
- Type information from type annotations
- What code does (the "what")

**Claude CANNOT infer and needs explicitly documented:**
- Why architectural decisions were made (rationale, tradeoffs, alternatives considered)
- Business rules and domain-specific logic
- Non-obvious constraints (e.g., "this API must stay backward-compatible")
- Event flow and invisible coupling between components (especially event-driven architectures
  where the coupling is not visible in import statements)
- What NOT to do (anti-patterns specific to the project)
- Runtime behavior that is not obvious from reading static code
- Cross-component invariants that span multiple files
- Config-driven behavior where the code path depends on runtime settings
- Historical context ("we used to do X, switched to Y because Z")

Research finding (Arize): repository-localized CLAUDE.md instructions outperform generalized
ones -- investing in tailored, project-specific context yields a measurable +10.87% accuracy
boost vs +5.19% for generic cross-repository instructions.

### Structural Patterns That Help

**Progressive Disclosure**: Load the minimum context needed for the current task, not
everything all the time. Use:
- Root `CLAUDE.md` for universally applicable rules (keep small)
- Subdirectory `CLAUDE.md` files for path-scoped context (load on demand)
- `@path/to/file` imports for detailed reference docs (load at launch)
- Separate docs that Claude reads when needed (no special loading)

**Specificity over vagueness**:
- "Use 2-space indentation" beats "Format code properly"
- "Run `npm test` before committing" beats "Test your changes"
- "API handlers live in `src/api/handlers/`" beats "Keep files organized"

**Emphasis for critical rules**: Adding "IMPORTANT" or "YOU MUST" to instructions
demonstrably improves adherence per Anthropic's official docs.

**The WHY/WHAT/HOW pattern**:
1. WHAT: Technology stack, project structure, codebase architecture
2. WHY: The project's purpose and function of different components
3. HOW: Instructions for meaningful work (build, test, deploy commands)

**Use markdown headers and bullets**: Claude scans structure the same way readers do.
Organized sections are easier to follow than dense paragraphs.

### Enforcement Hierarchy

Understanding what goes where in Claude Code's enforcement stack:

| Level | Mechanism | Compliance | Best For |
|-------|-----------|-----------|----------|
| Deterministic | `settings.json` (permissions, sandbox) | 100% | Security, tool restrictions, env vars |
| Deterministic | Hooks (shell commands on events) | 100% | Linting, formatting, test-running, doc sync |
| Behavioral | CLAUDE.md instructions | ~80% | Architectural context, conventions, design decisions |
| Passive | Co-located docs (subdirectory CLAUDE.md) | N/A | Reducing drift by proximity |

If something must happen every time without exception, make it a hook, not a CLAUDE.md
instruction.

---

## Part 3: Our File Structure

### Available Mechanisms

We evaluated four approaches for organizing architecture documentation:

**1. `.claude/rules/` files** -- Path-scoped rules with YAML frontmatter glob patterns.
Load on demand when Claude reads matching files.
- Pro: Fine-grained path scoping with glob patterns
- Con: Lives in `.claude/` which we do not commit to the repo, so rules are not
  version-controlled or shared

**2. Subdirectory CLAUDE.md files** -- CLAUDE.md files placed in source subdirectories.
Load on demand when Claude reads files in that directory.
- Pro: Version-controlled, co-located with code, on-demand loading
- Con: Less precise scoping than glob patterns (directory-level only)

**3. `@path/to/file` imports** -- References in root CLAUDE.md that expand at launch.
- Pro: Can reference any file, supports relative/absolute paths, 5-hop recursion
- Con: All imports expand at launch (no on-demand loading), consumes instruction budget

**4. Standalone docs in `design/`** -- Files Claude reads when it needs them, not
automatically loaded.
- Pro: Zero instruction budget cost, unlimited length, version-controlled
- Con: Claude must be told or choose to read them; not automatically injected

### Chosen Approach: Hybrid

Since we do not commit `.claude/`, we use a combination of mechanisms 2, 3, and 4:

```
CLAUDE.md                              ~200 lines, universal rules
  @design/architecture.md             Imported at launch (always loaded)

roc/pipeline/CLAUDE.md                 On-demand: pipeline flow, bus topology, sync rules
roc/perception/CLAUDE.md               On-demand: feature extractors, PHYSICAL vs RELATIONAL
roc/db/CLAUDE.md                       On-demand: Node/Edge patterns, schema constraints
roc/reporting/CLAUDE.md                On-demand: dashboard architecture, API overview
roc/framework/CLAUDE.md                On-demand: ExpMod system, component lifecycle
tests/CLAUDE.md                        On-demand: test organization, fixtures, markers
dashboard-ui/CLAUDE.md                 On-demand: React/Mantine patterns, server architecture

design/                                Standalone docs, read when needed
  architecture.md                      Core architecture doc (imported by root CLAUDE.md)
  *.md                                 Existing design docs (ADRs, research, plans)
```

**Rationale for each tier:**

- **Root CLAUDE.md (~200 lines)**: Contains only what applies to every single task regardless
  of which area of the codebase Claude is touching. Architectural invariants, common
  commands, code style, "what NOT to do."

- **`@design/architecture.md` (imported at launch)**: The architecture overview is genuinely
  universal -- Claude should know system boundaries, quality goals, and key decisions
  regardless of which directory it is working in. Worth the instruction budget cost.

- **Subdirectory CLAUDE.md files (on-demand)**: Domain-specific details that only matter when
  working in that area. Pipeline flow diagram only loads when editing pipeline code. Test
  conventions only load when writing tests. This is the key mechanism for staying within the
  instruction budget.

- **`design/*.md` (standalone)**: Detailed design docs, ADRs, research documents. Not
  automatically loaded. Claude reads these when it needs deep context on a specific topic.
  These serve as de facto ADRs and do not need restructuring.

### What Goes Where

| Content | Location | Loading | Why |
|---------|----------|---------|-----|
| Architectural invariants | Root `CLAUDE.md` | Always | Must never be violated, applies everywhere |
| "What NOT to Do" rules | Root `CLAUDE.md` | Always | Highest-value content per research |
| Package layout map | Root `CLAUDE.md` | Always | Universal reference |
| Common commands | Root `CLAUDE.md` | Always | Universal reference |
| Code style (4 lines) | Root `CLAUDE.md` | Always | Universal |
| System context and quality goals | `@design/architecture.md` | Always | Universal but too long for root |
| Key decisions with rationale | `@design/architecture.md` | Always | Universal context |
| Cross-cutting concerns | `@design/architecture.md` | Always | Spans all components |
| Pipeline flow diagram | `roc/pipeline/CLAUDE.md` | On-demand | Only needed when editing pipeline |
| Bus topology table | `roc/pipeline/CLAUDE.md` | On-demand | Only needed when editing pipeline |
| Pipeline sync rules | `roc/pipeline/CLAUDE.md` | On-demand | Only needed when editing pipeline |
| ExpMod modtype table | `roc/framework/CLAUDE.md` | On-demand | Only needed when editing ExpMods |
| Component lifecycle | `roc/framework/CLAUDE.md` | On-demand | Only needed when editing framework |
| Node/Edge patterns | `roc/db/CLAUDE.md` | On-demand | Only needed when editing graph code |
| Schema constraints | `roc/db/CLAUDE.md` | On-demand | Only needed when editing graph code |
| Feature extractor patterns | `roc/perception/CLAUDE.md` | On-demand | Only needed for perception work |
| PHYSICAL vs RELATIONAL | `roc/perception/CLAUDE.md` | On-demand | Only needed for perception work |
| Test organization tree | `tests/CLAUDE.md` | On-demand | Only needed when writing tests |
| Test fixtures and markers | `tests/CLAUDE.md` | On-demand | Only needed when writing tests |
| Dashboard patterns | `dashboard-ui/CLAUDE.md` | On-demand | Only needed for frontend work |
| API overview | `roc/reporting/CLAUDE.md` | On-demand | Only needed for reporting work |
| Debugging tools decision tree | `roc/reporting/CLAUDE.md` | On-demand | Only needed for debugging |
| Detailed design docs (ADRs) | `design/*.md` | Manual | Claude reads when it needs deep context |

### Architecture Doc Sections

Based on the literature review and Claude Code's specific needs, our architecture document
(`design/architecture.md`) should contain these sections:

**Essential (from literature consensus + Claude Code needs):**

1. **Context and Scope** -- ROC's boundaries with external systems (Gymnasium/NetHack,
   Memgraph, DuckLake/DuckDB, OpenTelemetry, Dashboard UI). What is inside the system vs
   outside. This is the C4 Level 1 / arc42 Section 3.

2. **Quality Attribute Goals** -- The non-functional requirements that drove the architecture.
   Why event-driven? Why plugin architecture (ExpMod)? Why game-agnostic core? Why
   component-based? These are the "why" that Claude cannot infer from code.

3. **Solution Strategy** -- The fundamental approach: component-based event-driven pipeline,
   reactive streams (RxPY), graph database for state, plugin system for algorithms. High-level
   decisions before diving into details.

4. **Static Structure** -- Package layout, component relationships. We have this in the
   current CLAUDE.md. The root CLAUDE.md keeps the directory tree; the architecture doc
   elaborates on relationships.

5. **Runtime Behavior** -- The pipeline flow diagram showing how data moves through the
   system. Currently in CLAUDE.md; should move to `roc/pipeline/CLAUDE.md` for on-demand
   loading, with a summary in the architecture doc.

6. **Key Decisions with Rationale** -- The WHY behind each architectural invariant and major
   design choice. Link to detailed design docs in `design/` where they exist. This is the
   section the literature identifies as most frequently missing and most often needed.

7. **Cross-cutting Concerns** -- Config singleton pattern, logging (Loguru + OTel),
   observability stack, threading model (RxPY ThreadPoolScheduler), error handling patterns.

**Strongly Recommended:**

8. **Constraints** -- Python 3.13, Memgraph dependency, single-threaded DuckDB, uv package
   manager, ruff formatter, port range 9000-9099.

9. **Risks and Technical Debt** -- Known problems worth flagging for future work.

10. **Glossary** -- Domain terms that have specific meaning in ROC: FeatureGroup,
    Transformable, ExpMod, Frame, Transform, IntrinsicNode, SaliencyMap, etc.

---

## Part 4: Keeping Docs in Sync

The literature warns extensively about "The Dusty Tome" and "The Model-Code Gap" anti-
patterns. We use a layered enforcement approach:

### Enforcement Levels

| Level | Mechanism | Compliance | Purpose |
|-------|-----------|-----------|---------|
| Deterministic | `Stop` hook (prompt type) | 100% | Gate Claude before finishing; verify docs updated |
| Deterministic | `PostToolUse` hook (command type) | 100% | Remind Claude after editing architectural files |
| Behavioral | CLAUDE.md instruction | ~80% | Always-present guidance to update docs |
| Passive | Co-located subdirectory CLAUDE.md | N/A | Reduce drift by proximity |
| Verification | Architecture fitness tests | 100% in CI | Catch violations after the fact |

### Hook Configurations

Hooks live in `settings.json` (either `.claude/settings.local.json` for local-only or
`~/.claude/settings.json` for user-wide). They fire on Claude Code events and can block
actions or inject messages.

**Stop Hook -- Gate on Completion**

This fires when Claude finishes responding. A prompt-type hook asks a lightweight LLM to
check whether architectural changes were made without corresponding doc updates. If the
check fails, Claude receives the reason and continues working.

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Check if any files in roc/pipeline/, roc/framework/, or roc/db/ were modified. If architectural files were changed (new EventBus, new Component, changed Edge schema, new ExpMod), verify that the corresponding architecture docs were also updated. Return {\"ok\": true} if docs are in sync or no arch changes were made, or {\"ok\": false, \"reason\": \"description of what needs updating\"} if not."
          }
        ]
      }
    ]
  }
}
```

**PostToolUse Hook -- Remind After Architectural Edits**

This fires after every Edit or Write to files in architectural directories. It injects a
reminder into Claude's context. It does not block -- it nudges.

Script (`hooks/check-arch-docs.sh`):

```bash
#!/bin/bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ "$FILE_PATH" =~ roc/(pipeline|framework|db)/ ]]; then
  echo "Architectural file modified: $FILE_PATH"
  echo "If this changes EventBus topology, Component registration, Edge schemas,"
  echo "or ExpMod types, update the corresponding architecture documentation."
fi
exit 0
```

Configuration:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "./hooks/check-arch-docs.sh"
          }
        ]
      }
    ]
  }
}
```

**Hook API Reference**

Key details for implementing hooks:

- **Exit code 0**: Allow the action (stdout may contain JSON or context text)
- **Exit code 2**: Block the action (stderr becomes Claude's feedback)
- **JSON output**: Can include `hookSpecificOutput` with `permissionDecision` (allow/deny/ask)
  for PreToolUse, or `decision` (block) for other events
- **Input via stdin**: JSON with `session_id`, `cwd`, `hook_event_name`, `tool_name`,
  `tool_input` (tool-specific arguments including `file_path` for Edit/Write)
- **Environment**: `CLAUDE_SESSION_ID`, `CLAUDE_PROJECT_DIR`, `CLAUDE_ENV_FILE`, standard
  shell vars

### CLAUDE.md Instructions

Add to the root CLAUDE.md (behavioral guidance, ~80% compliance):

```markdown
## Documentation Maintenance

When modifying architectural code, update the corresponding documentation:
- New/removed EventBus -> update bus topology in roc/pipeline/CLAUDE.md
- New/removed Component -> update component list in design/architecture.md
- New/removed ExpMod modtype -> update ExpMod table in roc/framework/CLAUDE.md
- Changed Edge allowed_connections -> update schema docs in roc/db/CLAUDE.md
- Changed pipeline flow -> update pipeline diagram in roc/pipeline/CLAUDE.md
- New architectural invariant -> add to root CLAUDE.md Architectural Invariants section

When adding a new architectural invariant, add it to both the Architectural Invariants
section AND the "What NOT to Do" section if applicable.
```

### Architecture Fitness Tests

Tests that verify architectural invariants programmatically. These serve as living
documentation that cannot go stale because they fail when violated. The test names and
docstrings themselves become documentation.

Examples of fitness tests for ROC:

```python
# tests/unit/test_architecture.py

def test_all_eventbuses_are_class_attributes():
    """EventBuses must be declared as class-level attributes on Components.

    Invariant #4: No ad-hoc bus creation in functions, methods, or module scope.
    """
    # Scan for EventBus() instantiation outside class bodies
    ...

def test_no_game_imports_outside_perception():
    """Only perception/ and reporting/state.py may import nle or gymnasium.

    Invariant #2: Game-specific code is confined to the perception layer.
    """
    # Check import statements across the codebase
    ...

def test_pipeline_components_use_expmod():
    """Resolution, action, saliency must go through ExpMod.get().

    Invariant #3: Algorithm selection goes through ExpMod, not hardcoded.
    """
    ...

def test_no_direct_pipeline_cross_imports():
    """Pipeline components must not import other pipeline component classes.

    Invariant #1: All inter-component communication flows through EventBuses.
    Importing a component class to access its .bus class attribute is the only
    allowed cross-component reference.
    """
    ...

def test_edge_subclasses_define_allowed_connections():
    """Every Edge subclass must define allowed_connections.

    Invariant #8: Edge connections must satisfy schema constraints.
    """
    ...
```

These tests complement the documentation -- the docs explain the "why" and the tests
enforce the "what." Together they prevent both the "Dusty Tome" (stale docs) and the
"Model-Code Gap" (docs that do not match reality).

---

## Sources and References

### Standards
- ISO/IEC/IEEE 42010:2011 -- "Systems and software engineering -- Architecture description."
  Originally IEEE Std 1471-2000.

### Books
- Clements, P., Bachmann, F., Bass, L., Garlan, D., Ivers, J., Little, R., Merson, P.,
  Nord, R., & Stafford, J. (2010). *Documenting Software Architectures: Views and Beyond*
  (2nd ed.). Addison-Wesley.
- Bass, L., Clements, P., & Kazman, R. (2021). *Software Architecture in Practice*
  (4th ed.). Addison-Wesley.
- Keeling, M. (2017). *Design It! From Programmer to Software Architect*. Pragmatic
  Bookshelf.
- Fairbanks, G. (2010). *Just Enough Software Architecture: A Risk-Driven Approach*.
  Marshall & Brainerd.
- Rozanski, N. & Woods, E. (2012). *Software Systems Architecture: Working with
  Stakeholders Using Viewpoints and Perspectives* (2nd ed.). Addison-Wesley.
- Brown, S. (2018). *Software Architecture for Developers* (Vol. 1 & 2). Leanpub.
- Starke, G. (2022). *arc42 by Example* (2nd ed.). Leanpub.

### Frameworks
- arc42: https://arc42.org/overview
- C4 Model: https://c4model.com
- Kruchten, P. (1995). "The 4+1 View Model of Architecture." *IEEE Software*, 12(6), 42-50.

### Claude Code Official Documentation
- Best Practices for Claude Code: https://code.claude.com/docs/en/best-practices
- How Claude remembers your project: https://code.claude.com/docs/en/memory
- Explore the context window: https://code.claude.com/docs/en/context-window

### Community Sources
- Arize Blog: "Claude MD Best Practices Learned from Optimizing Claude Code with Prompt
  Learning" -- https://arize.com/blog/claude-md-best-practices-learned-from-optimizing-claude-code-with-prompt-learning/
- ShareUHack: "Claude Code CLAUDE.md Setup Guide 2026" --
  https://www.shareuhack.com/en/posts/claude-code-claude-md-setup-guide-2026
- HumanLayer Blog: "Writing a Good Claude MD" --
  https://www.humanlayer.dev/blog/writing-a-good-claude-md
- How Anthropic teams use Claude Code:
  https://claude.com/blog/how-anthropic-teams-use-claude-code
- FreeCodeCamp: "Claude Code Handbook" --
  https://www.freecodecamp.org/news/claude-code-handbook/
