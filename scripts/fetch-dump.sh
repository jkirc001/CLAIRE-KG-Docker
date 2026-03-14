#!/bin/bash
set -e

DUMP_URL="https://github.com/jkirc001/CLAIRE-KG-Docker/releases/download/v0.1.0-data/neo4j.dump"
DEST_DIR="./backups"
DEST_FILE="${DEST_DIR}/neo4j.dump"

if [ -f "$DEST_FILE" ]; then
    echo "Dump file already exists at ${DEST_FILE}. Skipping download."
    exit 0
fi

mkdir -p "$DEST_DIR"

echo "Downloading neo4j.dump from GitHub Release..."
curl -L --progress-bar -o "$DEST_FILE" "$DUMP_URL"

echo "Download complete: ${DEST_FILE}"
