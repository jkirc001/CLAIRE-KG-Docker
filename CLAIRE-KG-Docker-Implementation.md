# CLAIRE-KG-Docker Implementation Document

## 1. Project Overview

CLAIRE-KG-Docker is a containerized distribution of the CLAIRE-KG system (Cybersecurity LLM Assisted Information Retrieval Engine - Knowledge Graph). It provides a pre-ingested Neo4j knowledge graph and a Python CLI application packaged as Docker services. Users need only Docker and an OpenAI API key to start the stack and ask natural language questions about cybersecurity frameworks.

The system uses an LLM to generate Cypher queries against a knowledge graph containing six cybersecurity datasets (CVE, CWE, CAPEC, ATT&CK, NICE, DCWF) with crosswalk relationships, then enhances raw query results into cited natural language answers. Testing showed a 100% automated pass rate and 95% human-validated pass rate on cross-framework queries, compared to 10%/15% for CLAIRE-DirectLLM and 0%/5% for CLAIRE-RAG.

### Repository

- GitHub: https://github.com/jkirc001/CLAIRE-KG-Docker
- Source pinned to commit `22fc17204c2e1f3094b64c63a3176c07f54652ee` of the main [CLAIRE-KG](https://github.com/jkirc001/CLAIRE-KG) repository.

---

## 2. Development Environment

### Hardware

- **Machine:** MacBook Pro 16-inch, 2019 ([specs](https://support.apple.com/en-us/111932))
- **CPU:** 2.3 GHz 8-Core Intel Core i9
- **Graphics:** AMD Radeon Pro 5500M 4 GB, Intel UHD Graphics 630 1536 MB
- **Memory:** 32 GB 2667 MHz DDR4
- **OS:** macOS Sequoia 15.7.4 (Darwin 24.6.0)

### Software

- **Docker:** Docker Desktop for Mac with Docker Compose v2
- **Python (container):** 3.13-slim (Debian base)
- **Build system:** hatchling
- **Package manager (source repo):** uv
- **Container registry:** GitHub Container Registry (GHCR) at `ghcr.io/jkirc001/claire-kg-app`
- **CI/CD:** GitHub Actions

---

## 3. Architecture

### 3.1 Knowledge Graph Pipeline (3 Phases)

1. **Phase 1: Cypher Generation** -- Classifies the user's question using rule-based regex patterns (no LLM) to identify relevant datasets, intent type, and complexity. Then sends the question, a dynamically-discovered Neo4j schema, and classification metadata to OpenAI (default: `gpt-4o`) to generate a Cypher query. The query is validated, post-processed with domain-specific fixes, and executed against Neo4j.

2. **Phase 2: Answer Enhancement** -- Takes the raw Neo4j query results and the original question, constructs a grounding prompt that requires claims to be sourced only from the database results with UID citations (e.g., [CWE-79], [CAPEC-100], [T1059]), and sends it to OpenAI (default: `gpt-4o`) to generate a natural language answer.

3. **Phase 3: Evaluation (Optional)** -- Runs DeepEval metrics (Relevancy, Faithfulness, GEval) against the generated answer to score answer quality. Disabled by default; enabled with the `--eval` flag.

### 3.2 Server Architecture

The system runs as a two-container Docker Compose stack:

- **Neo4j service (`claire-kg-docker-graphrag`):** Neo4j 5.26.12 with APOC and Graph Data Science plugins. Stores the pre-ingested knowledge graph. Exposes port 7475 (Browser) and 7688 (Bolt), offset from standard ports to avoid conflicts with the main CLAIRE-KG repo.
- **App service (`app`):** Python 3.13-slim container running the CLAIRE-KG CLI. Connects to Neo4j via the Docker network service name. Runs as one-off commands (`docker compose run --rm`), not as a persistent server.

Unlike CLAIRE-RAG-Docker (which runs a persistent FastAPI server to avoid ML model cold starts), CLAIRE-KG-Docker uses the `docker compose run --rm` pattern. The application has no ML models to load -- the only latency sources are the Neo4j connection and OpenAI API calls.

### 3.3 Data Flow

```
User -> ./ask script -> docker compose run --rm app ask "question"
                            |
                            v
                      QuestionClassifier.classify()
                            | (datasets, intents, complexity, schema pack)
                            v
                      CypherGenerator.generate_cypher()
                            | (schema discovery, special cases, LLM call, validation, post-processing)
                            v
                      Neo4j: execute_cypher() -> raw results (JSON)
                            |
                            v
                      LLMOrchestrator Phase 2: Enhance answer with citations
                            | (grounding instructions, question-type handlers)
                            v
                      (Optional) Phase 3: DeepEval evaluation
                            |
                            v
                      User Answer (natural language + UID citations)
```

---

## 4. Package Structure

```
CLAIRE-KG-Docker/
  src/claire_kg/                       # Application source (copied from main repo)
    __init__.py                        # Package init, version 0.1.0
    __main__.py                        # CLI entry point (delegates to cli.py)
    cli.py                             # Typer CLI (ask, query, setup, test commands)
    database.py                        # Neo4j connection, schema validation, status
    cypher_generator.py                # LLM-based Cypher query generation (OpenAI SDK)
    question_classifier.py             # Rule-based question analysis (no LLM)
    query_orchestrator.py              # Phase 1 only: Cypher generation + execution
    llm_orchestrator.py                # Full 3-phase pipeline (Cypher -> execute -> enhance)
    query_validator.py                 # Query structure validation
    rag_search.py                      # RAG/vector similarity search (embeddings)
    runner.py                          # Subprocess runner for external callers
    ingest.py                          # Dataset ingestion engine (APOC-based)
    evaluator.py                       # Phase 3: DeepEval evaluation (Relevancy, Faithfulness, GEval)
    schema_knowledge.py                # Dynamic schema discovery from Neo4j
    curated_schema_builder.py          # Classification-driven schema optimization
    dataset_metadata.py                # Cached metadata (role names, node counts, baseline examples)
    archive/                           # Legacy/archived modules (not active)
      orchestrator.py
      llm_orchestrator.py.archived
      debug_system.py
      vocab.py
      README.md
  docker/
    neo4j-entrypoint.sh                # Dump restore on first start
  scripts/
    fetch-dump.sh                      # Downloads dump from GitHub Release
  config/ (none)                       # No config directory; env vars drive configuration
  .github/
    workflows/
      build-push.yml                   # Multi-arch GHCR build on release
  ask                                  # User-facing wrapper script
  Dockerfile
  docker-compose.yml
  docker-compose.override.yml          # Local build override (gitignored)
  pyproject.toml
  .env.example
  .gitignore
  .dockerignore
  VERSION                              # Pinned source commit SHA
  LICENSE                              # MIT
  README.md
```

---

## 5. Dependencies and Library Choices

### 5.1 Core Dependencies

| Library | Version Constraint | Purpose | Decision Rationale |
|---------|-------------------|---------|-------------------|
| `neo4j` | `>=5.0.0` | Neo4j Python driver | Connects to Neo4j 5.x via Bolt protocol. Required for all Cypher query execution. |
| `openai` | `>=2.6.1` | OpenAI API client | v2.x API for chat completions. Used for Cypher generation (Phase 1) and answer enhancement (Phase 2). |
| `tiktoken` | `>=0.12.0` | Token counting for cost tracking | Accurate token counting for prompt optimization and cost comparison between full and minimal prompts. |
| `python-dotenv` | `>=1.0.0` | Environment variable loading from .env files | Loads OPENAI_API_KEY and NEO4J_* variables. Silently does nothing if no .env file exists. |
| `typer` | `>=0.9.0` | CLI framework | Provides command structure (ask, query, setup subcommands) with argument parsing and help text. |
| `rich` | `>=13.0.0` | Terminal formatting | Colored output, panels, tables for query analysis display and result formatting. |
| `httpx` | `>=0.24.0` | HTTP client | Async HTTP support used by the OpenAI SDK internally. |
| `tqdm` | `>=4.65.0` | Progress bars | Used during dataset ingestion and embedding generation for progress display. |
| `scikit-learn` | `>=1.3.2` | Vector similarity search | Cosine similarity computation for RAG fallback search when embeddings are available. |
| `deepeval` | `>=3.6.9` | Phase 3 evaluation metrics | Relevancy, Faithfulness, and GEval metrics for answer quality scoring. Disabled by default. |

### 5.2 Build Dependencies

| Tool | Purpose |
|------|---------|
| `hatchling` | Python build backend (PEP 517) |

### 5.3 Key Difference from CLAIRE-RAG-Docker and CLAIRE-DirectLLM-Docker

CLAIRE-KG-Docker has **no PyTorch dependency** and **no ML models to load**. Unlike CLAIRE-RAG-Docker (which requires sentence-transformers, cross-encoder models, and CPU-only PyTorch at ~2.93 GB image size), CLAIRE-KG-Docker relies on the OpenAI API for LLM operations and Neo4j for data retrieval. This eliminates cold start latency from model deserialization.

However, the `deepeval` dependency pulls a significant dependency tree (including pydantic, pandas, and various evaluation libraries), which increases the image size beyond what the core application requires. DeepEval is used only for optional Phase 3 evaluation and is not needed for the primary ask workflow.

### 5.4 No Config Directory

Unlike CLAIRE-RAG-Docker and CLAIRE-DirectLLM-Docker (which ship `config/settings.yaml` and `config/models.yaml`), CLAIRE-KG-Docker has no config directory in the Docker distribution. All configuration is driven by environment variables:

- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` for database connection
- `OPENAI_API_KEY` for LLM access
- `PHASE1_MODEL`, `PHASE2_MODEL` for model override (optional)

The application code loads `.env` via `python-dotenv` with path resolution from `__file__`; when the package is installed in site-packages, this path does not resolve to a project-root config directory, so container environment variables are the correct approach.

---

## 6. Data

### 6.1 Knowledge Graph

The Neo4j knowledge graph is pre-built from six cybersecurity datasets with crosswalk relationships:

| Dataset | Description | Approximate Node Count |
|---------|-------------|----------------------|
| CVE | Common Vulnerabilities and Exposures | ~38,000 |
| CWE | Common Weakness Enumeration | ~1,000 |
| CAPEC | Common Attack Pattern Enumeration and Classification | ~500 |
| ATT&CK | MITRE ATT&CK framework (techniques, tactics, mitigations) | ~1,000 |
| NICE | NICE Cybersecurity Workforce Framework | ~100 |
| DCWF | DoD Cyber Workforce Framework (work roles, tasks, knowledge) | ~100 |

### 6.2 Crosswalk Relationships

The graph includes relationships linking entities across frameworks:

| Crosswalk | Relationship | Description |
|-----------|-------------|-------------|
| CVE-CWE | `HAS_WEAKNESS` | CVE linked to its root-cause CWE |
| CVE-Asset | `AFFECTS` | CVE linked to affected vendor/product |
| CVE-ATT&CK | via CWE/CAPEC chain | CVE to techniques via weakness/attack pattern |
| CAPEC-ATT&CK | `USES_TECHNIQUE` | Attack patterns linked to ATT&CK techniques |
| CAPEC-CWE | `EXPLOITS_WEAKNESS` | Attack patterns linked to target weaknesses |
| NICE-ATT&CK | via shared knowledge areas | Workforce roles to relevant techniques |
| NICE-DCWF | `MAPS_TO` | NICE framework roles to DCWF work roles |
| CWE-Mitigation | `MITIGATED_BY` | Weaknesses linked to mitigation strategies |

### 6.3 Neo4j Database Dump

| Property | Value |
|----------|-------|
| File | `backups/neo4j.dump` |
| Size | 553 MB |
| Contents | All six datasets + crosswalks (no embeddings) |
| Neo4j Version | 5.26.12 |
| Download URL | https://github.com/jkirc001/CLAIRE-KG-Docker/releases/download/v0.1.0-data/neo4j.dump |

### 6.4 Dump Distribution

The dump is hosted as a GitHub Release asset (`v0.1.0-data`). Users download it with `scripts/fetch-dump.sh` or manually from the release page.

Decision: The dump is not baked into the Docker image because:
- It would increase the Neo4j image size by 553 MB on every pull
- It changes independently of the application code
- Users may want to inspect or replace the dump without rebuilding
- The Neo4j base image is already 848 MB; adding the dump would push it past 1.4 GB

### 6.5 Dump Restore

The dump is restored automatically on first container start via a custom entrypoint script (`docker/neo4j-entrypoint.sh`). The script:

1. Checks for a `/data/.restore-complete` flag file
2. If absent, verifies `/backups/neo4j.dump` exists (exits with error if not)
3. Runs `neo4j-admin database load neo4j --from-path=/backups --overwrite-destination=true`
4. Creates the flag file to prevent re-restore on subsequent starts
5. Chains to Neo4j's official entrypoint (`/startup/docker-entrypoint.sh`)

The backups directory is mounted read-only (`:ro`) to prevent the container from modifying the host's dump file.

### 6.6 Embeddings

The dump ships without embeddings. The main "ask" path uses Cypher generation and does not require embeddings. Embeddings are used only for RAG-based vector similarity search, which is an optional feature. Users can generate embeddings after setup by running `docker compose run --rm app setup embeddings` (requires OpenAI API credits).

---

## 7. Docker Implementation

### 7.1 App Container

#### Base Image

`python:3.13-slim` was chosen because:
- CLAIRE-KG has no PyTorch dependency, so Python 3.13 compatibility is not a concern
- The `-slim` variant minimizes image size
- No C/C++ compilers needed (no native extensions to build)

#### Dockerfile

The Dockerfile is minimal (11 lines):

1. Copy `pyproject.toml`, `README.md`, and `src/`
2. Install the package via `pip install --no-cache-dir .`
3. Set entrypoint to `python -m claire_kg` with default command `ask --help`

No uv is installed in the image. The source repo uses uv for development, but the Docker image installs via pip from `pyproject.toml` using the hatchling build backend.

### 7.2 Neo4j Container

#### Image and Plugins

`neo4j:5.26.12` with two plugins:
- **APOC** -- Required for data loading procedures used during ingestion
- **Graph Data Science (GDS)** -- Licensed plugin; requires `NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"`

Plugins are auto-downloaded on first start and cached in a named volume (`claire-kg-docker-graphrag-plugins`).

#### Memory Configuration

Reduced from the main repo's ingestion-tuned settings (18 GB total) to a read-only optimized profile:

| Setting | Value | Purpose |
|---------|-------|---------|
| `heap_initial__size` | 2 GB | JVM heap starting size |
| `heap_max__size` | 4 GB | JVM heap maximum |
| `pagecache_size` | 2 GB | Neo4j page cache for graph data |
| `transaction_total_max` | 2 GB | Maximum memory for concurrent transactions |

Total Neo4j allocation: ~10 GB, workable on a 16 GB machine for single-user CLI queries.

#### Custom Entrypoint

The Neo4j container uses a custom entrypoint (`docker/neo4j-entrypoint.sh`) that restores the database dump on first start, then chains to Neo4j's official entrypoint. This gives users a "just works" experience with no manual restore step.

#### Healthcheck

`cypher-shell -u neo4j -p graphrag-password "RETURN 1"` at 30-second intervals with a 5-minute start period. The start period accounts for the first-start dump restore, during which Neo4j is not yet running and the healthcheck fails.

### 7.3 Port Mapping

| Service | Container Port | Host Port | Purpose |
|---------|---------------|-----------|---------|
| Neo4j Browser | 7474 | 7475 | Web UI for graph inspection |
| Neo4j Bolt | 7687 | 7688 | Cypher query protocol |

Non-standard host ports (7475/7688 instead of 7474/7687) avoid conflicts with the main CLAIRE-KG repo's Neo4j instance, allowing both to run simultaneously during development.

### 7.4 Multi-Architecture Support

The CI/CD pipeline builds for both `linux/amd64` and `linux/arm64`. ARM64 support is achieved via QEMU emulation on GitHub Actions. The Neo4j 5.26.12 image also supports both architectures natively. This allows the stack to run on:

- Linux x86_64 servers
- Apple Silicon Macs (M1/M2/M3/M4) via Docker Desktop

### 7.5 Docker Compose Configuration

The `docker-compose.yml` defines two services:

**Neo4j (`claire-kg-docker-graphrag`):**
- Image: `neo4j:5.26.12`
- Restart policy: `unless-stopped`
- Volumes: data, logs, import, plugins (named), backups and entrypoint (bind-mounted, read-only)
- Custom entrypoint for dump restore

**App (`app`):**
- Image: `ghcr.io/jkirc001/claire-kg-app:latest`
- Environment: `env_file: .env` for `OPENAI_API_KEY`, plus hardcoded `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- Depends on: Neo4j (`condition: service_healthy`)
- No ports, no volumes, no healthcheck (run-and-exit pattern)

A `docker-compose.override.yml` (gitignored) is available for local development builds that adds `build: .` to the app service.

### 7.6 Named Volumes

| Volume | Mount Point | Purpose |
|--------|-------------|---------|
| `claire-kg-docker-graphrag-data` | `/data` | Graph storage (persists across restarts) |
| `claire-kg-docker-graphrag-logs` | `/logs` | Neo4j diagnostic logs |
| `claire-kg-docker-graphrag-import` | `/import` | Dataset files for ingestion (empty in Docker distribution) |
| `claire-kg-docker-graphrag-plugins` | `/plugins` | Cached APOC and GDS plugin JARs |

All volumes use explicit `name:` fields to avoid Compose project-name prefixing.

---

## 8. CI/CD Pipeline

### 8.1 GitHub Actions Workflow

The workflow (`build-push.yml`) triggers on GitHub Release publish events and manual `workflow_dispatch`. It:

1. Checks out the repository
2. Sets up QEMU (for ARM64 cross-compilation)
3. Sets up Docker Buildx (multi-platform builder)
4. Logs into GHCR using `GITHUB_TOKEN`
5. Extracts image tags from the release (semver + `latest`)
6. Builds and pushes the multi-arch image with GitHub Actions cache (`type=gha,mode=max`)

### 8.2 Image Tags

- `ghcr.io/jkirc001/claire-kg-app:latest` -- latest release
- `ghcr.io/jkirc001/claire-kg-app:<version>` -- specific release (e.g., `0.1.0`)

### 8.3 GHCR Package Visibility

The GHCR package was manually changed to public via the GitHub web UI to allow unauthenticated pulls. GitHub creates GHCR packages as private by default, even for public repositories.

---

## 9. Wrapper Script (`./ask`)

The `ask` bash script provides the user-facing interface. It:

1. Loads environment variables from `.env` (if it exists)
2. Validates `OPENAI_API_KEY` is set; exits with a clear error if not
3. Auto-starts Neo4j with `docker compose up -d claire-kg-docker-graphrag`
4. Checks if Neo4j is healthy; if not, polls `docker inspect` every 5 seconds until healthy
5. Only displays the "Waiting for Neo4j..." message if Neo4j is not yet healthy (avoids noise on subsequent runs)
6. Runs `docker compose run --rm app ask "$@"` to pass the question through

### 9.1 Supported CLI Flags

The `ask` command (via `cli.py`) supports:

| Flag | Default | Description |
|------|---------|-------------|
| `--debug` | false | Show Cypher query, reasoning, confidence, cost, execution time |
| `--eval` | false | Enable Phase 3 DeepEval evaluation |
| `--phase1` | false | Phase 1 only (Cypher generation + execution, no answer enhancement) |
| `--phase2` | true | Phase 2 answer enhancement (default behavior) |
| `--model`, `-m` | gpt-4o | Override model for all phases |
| `--phase1-model` | gpt-4o | Override model for Cypher generation |
| `--phase2-model` | gpt-4o | Override model for answer enhancement |
| `--phase3-model` | gpt-4o | Override model for evaluation grading |
| `--limit`, `-l` | 25 | Maximum results to return from Neo4j |
| `--save` | false | Save results to JSON/MD/debug files |
| `--compare` | false | Run with and without schema optimization, compare results |
| `--class` | false | Show question classification details |
| `--examples` | (none) | Show example questions by dataset (e.g., `--examples CVE`, `--examples all`) |
| `--select` | false | Interactive example question selection |
| `--no-geval` | false | Disable GEval metric when `--eval` is used |

### 9.2 Additional CLI Commands

| Command | Description |
|---------|-------------|
| `query` | Phase 1 only (Cypher generation + execution) |
| `setup ingest <dataset>` | Ingest one dataset |
| `setup ingest-all` | Ingest all datasets |
| `setup crosswalk <type>` | Create relationships between datasets |
| `setup embeddings` | Generate vector embeddings (requires OpenAI API credits) |
| `setup clean` | Delete all graph data |
| `setup status` | Show node/relationship counts |

---

## 10. Application Pipeline Details

### 10.1 Question Classification

The `QuestionClassifier` uses rule-based regex patterns (no LLM call) to analyze questions. It returns a `ClassificationResult` containing:

- **Primary datasets:** Which datasets are relevant (CVE, CWE, CAPEC, ATT&CK, NICE, DCWF)
- **Crosswalk groups:** Which cross-dataset relationships are needed
- **Intent type:** lookup, count, list, traverse, filter, similarity_search, path_find, etc.
- **Complexity:** easy, medium, or hard
- **Failure patterns:** Known query patterns that tend to fail (count, ordering, cross-domain, correlation)
- **Expected schema pack:** Which subset of the Neo4j schema the Cypher generator needs

This classification drives schema selection for the Cypher generator, reducing prompt size by ~80-85% compared to sending the full schema.

### 10.2 Cypher Generation

The `CypherGenerator` uses the OpenAI API to translate natural language questions into Cypher queries.

**Schema Discovery:** The generator dynamically queries Neo4j for the current schema (node labels, relationship types, properties) and caches it. A `CuratedSchemaBuilder` further filters the schema based on the classification result, sending only relevant schema elements to the LLM.

**Prompt Modes:**
- **Minimal prompt** (default): Schema + question + limit only. ~62% cost reduction compared to full prompt. Works well for simple queries.
- **Full prompt**: Includes Cypher syntax rules, domain-specific rules, and filtered example queries from the baseline test suite. Used when classification indicates path_find or complete_chain complexity.

**Special Cases:** Several query types are handled directly without an LLM call:
- "Everything about CVE-2024-X" -- pre-built comprehensive Cypher with full joins
- "What knowledge is required for [role]?" -- direct knowledge-for-role query
- "Tasks associated with work role 441" -- direct tasks-for-role query
- "Top N most common CWEs" -- direct aggregation query

**Post-Processing:** Generated queries undergo validation and auto-fixing:
- Variable scope validation (WITH clause includes all needed variables)
- Missing properties auto-added (uid, name, description, severity, cvss_v31)
- Tactic filtering for ATT&CK queries via `TACTIC_LOWER_TO_CANONICAL` mapping
- Workforce-specific fixes (work_role property, COALESCE for title)
- CVE-specific fixes (severity as CVSS only; vulnerability keyword queries use HAS_WEAKNESS)
- Mitigation relationship detection

**Token Tracking:** Accurate token counting via tiktoken with before/after optimization comparison stored in the result.

### 10.3 Answer Enhancement (Phase 2)

The `LLMOrchestrator` sends the raw Neo4j results and original question to the LLM with grounding instructions:

- Answer ONLY from database results; no extrapolation
- Cite UIDs (e.g., [CWE-79], [CAPEC-100]) for every claim
- For limited result sets, say "From the knowledge graph:" and list only what was returned
- Do not add information not present in the query results

**Question-Type Handlers:** Specialized answer builders for different result shapes:
- Count questions: "There are N [entity type]."
- CVE affects vendor/product: structured affected-product lists
- Similarity questions: "The following N vulnerabilities are similar to [CVE]."
- Mitigation lists: formatted with source entity and UIDs
- Technique lists: tactics listed per technique
- Workforce/tasks: work role name + task list
- Attack chains: multi-hop results with intermediate entities

### 10.4 Evaluation (Phase 3)

When enabled (`--eval`), the `evaluator.py` module runs DeepEval metrics:

| Metric | Purpose | Grader Model |
|--------|---------|-------------|
| Relevancy | Does the answer address the question? | gpt-4o (default) |
| Faithfulness | Is every claim supported by database results? | gpt-4o (default) |
| GEval (optional) | Custom metric via LLM | gpt-4o (default) |

Scores range 0.0-1.0 for each metric. GEval can be disabled with `--no-geval`. A multimodal kwarg crash in DeepEval's Relevancy metric was fixed by patching the metric before evaluation.

---

## 11. Configuration

### 11.1 Environment Variables

**Required:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (mandatory for ask/query) |

**Optional (with defaults):**

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` (host) / `bolt://claire-kg-docker-graphrag:7687` (Docker) | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `graphrag-password` | Neo4j password |
| `PHASE1_MODEL` | `gpt-4o` | Cypher generation model |
| `PHASE2_MODEL` | `gpt-4o` | Answer enhancement model |

### 11.2 Neo4j Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `NEO4J_AUTH` | `neo4j/graphrag-password` | Database credentials |
| `NEO4J_PLUGINS` | `["apoc","graph-data-science"]` | Required plugins |
| `NEO4J_ACCEPT_LICENSE_AGREEMENT` | `yes` | Required for GDS plugin |

### 11.3 Config Path Resolution

The application code in `database.py` uses `load_dotenv(path)` with a path built by calling `os.path.dirname` three times on `__file__` (from `database.py` -> `claire_kg/` -> `src/` -> project root), then appending `config/.env`. When the package is installed in site-packages inside the Docker image, this path does not resolve to a valid `.env` file. `load_dotenv` silently does nothing when the file is missing, so container environment variables work correctly without modification.

---

## 12. Decisions Made During Implementation

### 12.1 Two-Container Stack (Not Single Container)

Unlike CLAIRE-DirectLLM-Docker (single container, no database) and CLAIRE-RAG-Docker (single container with file-based ChromaDB), CLAIRE-KG-Docker requires a separate Neo4j container. Neo4j is a full database server with its own JVM, memory management, and plugin system. Running it in-process with the Python application is not feasible.

### 12.2 Run-and-Exit Pattern (Not Persistent Server)

CLAIRE-RAG-Docker uses a persistent FastAPI server to avoid the ~11-second cold start from PyTorch/HuggingFace model loading. CLAIRE-KG-Docker has no ML models to load, so `docker compose run --rm` is appropriate. The app starts in <1 second; the dominant latency is the OpenAI API call.

### 12.3 Dump Restore via Entrypoint (Not Baked into Image)

The Neo4j dump is not baked into a custom Neo4j image. A 553 MB dump inside the image would increase pull size on every update and require rebuilding the image to update data. Instead, the dump is downloaded separately and restored on first start via a custom entrypoint script. This separates data updates from code updates.

### 12.4 Read-Only Backups Mount

The `./backups` directory is mounted as `:ro` on the Neo4j container, preventing the container from modifying or deleting the host's dump file. This is the opposite of CLAIRE-RAG-Docker's vectorstore mount, which requires read-write access due to ChromaDB's SQLite WAL mode.

### 12.5 Non-Standard Host Ports

Host ports 7475/7688 (instead of 7474/7687) were chosen to avoid conflicts with the main CLAIRE-KG repo's Neo4j instance, which uses the standard ports. This allows both to run simultaneously during development.

### 12.6 Python 3.13 (Not 3.11)

Unlike CLAIRE-RAG-Docker (which requires Python 3.11 for PyTorch compatibility), CLAIRE-KG-Docker has no PyTorch dependency and can use the latest Python. Python 3.13-slim provides a smaller base image.

### 12.7 hatchling Build Backend (Not uv_build)

The source repo uses uv for development, but `pyproject.toml` specifies hatchling as the build backend. This allows standard `pip install .` in the Docker image without installing uv. This is a simpler approach than CLAIRE-DirectLLM-Docker's workaround of parsing `pyproject.toml` with `tomllib` to bypass the uv_build backend.

### 12.8 No Config Directory in Docker Distribution

The main CLAIRE-KG repo uses `config/.env` for environment variables. The Docker distribution omits the config directory entirely and relies on container environment variables. This is the correct approach because `database.py`'s path resolution from `__file__` does not work when the package is installed in site-packages.

### 12.9 DeepEval Multimodal Kwarg Fix

DeepEval's Relevancy metric crashed when passed a multimodal keyword argument. A fix was applied in the evaluator to patch the metric before evaluation, allowing Phase 3 to work correctly.

### 12.10 Plugin Volume Persistence

Neo4j plugins (APOC, GDS) are stored in a named volume that persists across container restarts. This avoids re-downloading ~100 MB of plugin JARs on every start. However, if the Neo4j image version is updated, stale plugin JARs in the volume may cause compatibility issues. Users must remove the plugins volume and restart to resolve this.

### 12.11 Wrapper Script Auto-Start

The `./ask` wrapper script auto-starts the Neo4j container if it is not running and waits for it to be healthy before sending the query. This means users can run `./ask "question"` without remembering to start the stack first. The "Waiting for Neo4j..." message is only shown when Neo4j is not yet healthy, avoiding noise on subsequent runs.

### 12.12 No Auto-Pull of App Image

An earlier version of the wrapper script pulled the latest app image before each run. This was removed because it added latency on every invocation and was unnecessary for most users. Users who want the latest image can run `docker compose pull` manually.

---

## 13. Comparison with CLAIRE-RAG-Docker and CLAIRE-DirectLLM-Docker

| Aspect | CLAIRE-KG-Docker | CLAIRE-RAG-Docker | CLAIRE-DirectLLM-Docker |
|--------|-----------------|-------------------|------------------------|
| **Purpose** | Knowledge graph approach (Cypher queries) | RAG baseline (vector search + re-ranking) | Direct LLM baseline (no retrieval) |
| **Database** | Neo4j 5.26.12 (separate container) | ChromaDB (file-based, in-process) | None |
| **ML Models** | None | sentence-transformers + cross-encoder (~160 MB) | None |
| **PyTorch** | Not required | CPU-only torch 2.1.2 | Not required |
| **Python Version** | 3.13 | 3.11 (torch compatibility) | 3.13 |
| **Docker Services** | 2 (Neo4j + app) | 1 (persistent server) | 1 (run and exit) |
| **Runtime Mode** | CLI per invocation (`docker compose run --rm`) | Persistent FastAPI server | CLI per invocation (`docker compose run --rm`) |
| **Cold Start** | <1s (no models to load) | ~10s (PyTorch + HF model loading) | <1s (no models to load) |
| **Per-Query Time** | 3-8s (Neo4j + 2 OpenAI API calls) | 3-6s (retrieval + ranking + OpenAI API) | 1-3s (OpenAI API only) |
| **Data Download** | neo4j.dump (553 MB) | vectorstore.tar.gz (176 MB) | None |
| **Volume Mounts** | 4 named + 2 bind (Neo4j data, plugins, backups, entrypoint) | ./vectorstore (read-write) | None |
| **Ports** | 7475, 7688 (Neo4j Browser + Bolt) | 8000 (FastAPI) | None |
| **Healthcheck** | cypher-shell validation (30s interval, 5m start period) | HTTP GET /health (10s interval, 30s start) | None |
| **Build Backend** | hatchling | hatchling | uv_build (bypassed) |
| **OpenAI SDK** | v2.x (>=2.6.1) | v1.x (>=1.0.0) | v2.x (>=2.11.0) |
| **Wrapper Scripts** | `./ask` | `./ask` | `./ask`, `./compare` |
| **Test Pass Rate (auto/human)** | 100% / 95% | 0% / 5% | 10% / 15% |

---

## 14. Performance

### 14.1 Stack Startup

| Phase | Time | Notes |
|-------|------|-------|
| First start (dump restore) | 2-5 min | One-time; depends on disk speed |
| Subsequent starts (Neo4j) | 30-60s | JVM + plugin loading + page cache warm-up |
| App container startup | <1s | No models to load |

### 14.2 Per-Query Performance

| Component | Time |
|-----------|------|
| Question classification (regex) | <10 ms |
| Schema discovery (cached) | <100 ms |
| Cypher generation (OpenAI API) | 1-3s |
| Neo4j query execution | 100-500 ms |
| Answer enhancement (OpenAI API) | 2-5s |
| **Total (Phase 1 + 2)** | **3-8s** |
| Phase 3 evaluation (optional) | +5-15s |

### 14.3 Minimum System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| Disk | 2 GB (dump + images + graph data) | 5 GB |
| Network | Internet access (OpenAI API) | -- |

---

## 15. Releases

| Release | Purpose | Assets |
|---------|---------|--------|
| `v0.1.0-data` | Pre-ingested database dump | `neo4j.dump` (553 MB) |

---

## 16. Git History

```
f6eaf3e Remove docker-compose.override.yml from git
ea5048c Remove auto-pull from ask script
fb8cb35 Only show waiting message when Neo4j is not yet healthy
752d8a9 Pull latest app image before each ask run
f957378 Fix DeepEval Relevancy metric crash on multimodal kwarg
eaa789f Add embeddings setup instructions to README
d7b7a76 Use jkirc001 GHCR namespace and non-conflicting ports
116c9c9 Add ask wrapper, GHCR workflow, and fix Python version
1cf58d6 Add Neo4j entrypoint restore, fetch-dump script, and VERSION file
1c0e94f Add app source, Dockerfile, Compose, and repo scaffolding
c1e411b Initial commit
```

---

## 17. Files Copied from Source Repository

The following files were copied from the main CLAIRE-KG repository (commit `22fc17204c2e1f3094b64c63a3176c07f54652ee`):

**Included:**
- `src/claire_kg/` -- Complete application source package
- `pyproject.toml` -- Build configuration and dependencies
- `LICENSE` -- MIT

**Excluded:**
- `docs/` -- All documentation, figures, architecture diagrams
- `tests/` -- Full test suite
- `scripts/` -- Development and verification scripts
- `config/` -- Environment files (replaced by container environment variables)
- `final_outputs/`, `tmp_outputs/` -- Run artifacts
- `.cursor/`, `.taskmaster/`, `memory_bank_documents/`, `notes/` -- IDE and project config
- `uv.lock` -- Included in repo for reference but not used in Docker build

**Added for Docker distribution:**
- `docker/neo4j-entrypoint.sh` -- Dump restore entrypoint
- `scripts/fetch-dump.sh` -- Dump download script
- `ask` -- Wrapper script
- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.override.yml`
- `.github/workflows/build-push.yml`
- `.env.example`, `.gitignore`, `.dockerignore`
- `VERSION`
- `README.md`
