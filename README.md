# CLAIRE-KG Docker Distribution

Run the CLAIRE-KG cybersecurity knowledge graph with Docker. No Python, uv, or manual data ingestion required -- only Docker and an OpenAI API key.

This repo is for running CLAIRE-KG with Docker and pre-ingested data. Development and full documentation live in the [main CLAIRE-KG repo](https://github.com/jkirc001/CLAIRE-KG).

## Prerequisites

- Docker and Docker Compose
- An OpenAI API key (from https://platform.openai.com/api-keys)
- Minimum 16GB RAM, sufficient disk for the database dump (~580MB)

## Quick Start

1. Clone this repo:
   ```bash
   git clone https://github.com/jkirc001/CLAIRE-KG-Docker.git
   cd CLAIRE-KG-Docker
   ```

2. Set your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY=sk-your-key-here
   ```

3. Download the database dump:
   ```bash
   ./scripts/fetch-dump.sh
   ```
   Or download manually from the [v0.1.0-data release](https://github.com/jkirc001/CLAIRE-KG-Docker/releases/tag/v0.1.0-data) and place `neo4j.dump` in `./backups/`.

4. Start the stack:
   ```bash
   docker compose up -d
   ```
   On first start, Neo4j restores the database dump. This may take several minutes.

5. Ask a question:
   ```bash
   ./ask "How many work roles are in the DCWF framework?"
   ```
   Or without the wrapper:
   ```bash
   docker compose run --rm app ask "How many work roles are in the DCWF framework?"
   ```

## Alternative ways to set the API key

Export in your shell:
```bash
export OPENAI_API_KEY=sk-your-key-here
docker compose run --rm app ask "Your question?"
```

Inline for a single run:
```bash
OPENAI_API_KEY=sk-your-key-here docker compose run --rm app ask "Your question?"
```

## Stop and Cleanup

Stop the stack:
```bash
docker compose down
```

Remove volumes (deletes the graph and restored data):
```bash
docker compose down -v
```

## Notes

- The pre-ingested graph includes CVE, CWE, CAPEC, ATT&CK, NICE, and DCWF datasets with crosswalks. Embeddings are not included -- only the Cypher-based ask path is supported.
- The image includes the `deepeval` evaluation dependency, which increases image size. It is not needed for asking questions.
- If queries are slow on large result sets, increase Neo4j memory settings in `docker-compose.yml`. See the main repo's `docker/docker-compose.yml` for ingestion-tuned settings.
- If you update the Neo4j image version, remove the plugins volume to avoid stale plugin JARs: `docker volume rm claire-kg-graphrag-plugins`
