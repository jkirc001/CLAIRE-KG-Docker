#!/bin/bash
set -e

DUMP_DIR="/backups"
DUMP_FILE="${DUMP_DIR}/neo4j.dump"
RESTORE_FLAG="/data/.restore-complete"

if [ ! -f "$RESTORE_FLAG" ]; then
    if [ ! -f "$DUMP_FILE" ]; then
        echo "Error: No dump file found at ${DUMP_FILE}."
        echo "Run ./scripts/fetch-dump.sh first."
        exit 1
    fi

    echo "First start detected. Restoring database from ${DUMP_FILE}..."
    neo4j-admin database load neo4j --from-path="$DUMP_DIR" --overwrite-destination=true
    touch "$RESTORE_FLAG"
    echo "Database restore complete."
fi

exec /startup/docker-entrypoint.sh "$@"
