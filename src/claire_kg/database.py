"""
Neo4j database connection and operations for CLAIRE-KG.

Provides a single interface for connecting to Neo4j, running Cypher queries,
and performing schema validation and setup. Used by the ingest pipeline,
CLI (status, clean), Cypher generator (schema discovery), and query orchestrator.

Classes / types:
  - ValidationResult: Result of validate_schema() (valid flag + list of errors).
  - IngestionResult: Result of dataset ingestion (success, counts, optional error).
  - Neo4jConnection: Main connection class; use execute_cypher() for queries,
    get_status() for node/relationship counts (cached for small datasets),
    validate_schema() for golden-schema checks, create_constraints/create_indexes for setup.

Configuration: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (from config/.env).
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv
import logging

# Load environment variables from config/.env
load_dotenv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", ".env"
    )
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Result types for validation and ingestion
# -----------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of schema validation"""

    valid: bool
    errors: List[str]


@dataclass
class IngestionResult:
    """Result of dataset ingestion"""

    success: bool
    nodes_created: int = 0
    relationships_created: int = 0
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Neo4jConnection: connect, execute, status, validate, constraints/indexes
# -----------------------------------------------------------------------------


class Neo4jConnection:
    """Neo4j database connection and operations.

    Connects on construction; use execute_cypher() or execute_cypher_single()
    for queries. get_status() uses cached metadata for small datasets (WorkRole,
    Tactic, etc.) and live queries for large ones (Vulnerability, Asset).
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize Neo4j connection"""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "graphrag-password")

        self.driver = None
        self._connect()

    # --- Connection lifecycle ---

    def _connect(self):
        """Establish connection to Neo4j and verify with a test query."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the driver and release resources."""
        if self.driver:
            self.driver.close()

    # --- Query execution ---

    def execute_cypher(
        self, cypher: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return a list of record dicts. Re-raises with query context on error."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            # Re-raise with query context for debugging
            import traceback

            error_msg = f"{str(e)}\nQuery: {cypher[:200]}..."
            raise Exception(error_msg) from e

    def execute_cypher_single(
        self, cypher: str, parameters: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a Cypher query and return the first record, or None if no results."""
        results = self.execute_cypher(cypher, parameters)
        return results[0] if results else None

    # --- Status and schema validation ---

    def get_status(self) -> Dict[str, int]:
        """Return node/relationship counts; uses cached metadata for small datasets, live queries for large ones."""
        from .dataset_metadata import (
            get_dataset_metadata,
            is_small_dataset,
            WORKROLE_METADATA,
            SPECIALTYAREA_METADATA,
            TACTIC_METADATA,
            ABILITY_METADATA,
            TECHNIQUE_METADATA,
            SUBTECHNIQUE_METADATA,
            WEAKNESS_METADATA,
            ATTACKPATTERN_METADATA,
        )

        # Use cached metadata for small datasets, query for large ones
        # Metrics with None use cache_map (small datasets); others run live Cypher
        status_queries = {
            "Total Nodes": "MATCH (n) RETURN count(n) as count",
            "Total Relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "Vulnerability Nodes": "MATCH (n:Vulnerability) RETURN count(n) as count",
            "Asset Nodes": "MATCH (n:Asset) RETURN count(n) as count",
            "Mitigation Nodes": "MATCH (n:Mitigation) RETURN count(n) as count",  # Query
            "Task Nodes": "MATCH (n:Task) RETURN count(n) as count",  # Query
            "Knowledge Nodes": "MATCH (n:Knowledge) RETURN count(n) as count",  # Query
            "Skill Nodes": "MATCH (n:Skill) RETURN count(n) as count",  # Query
            "Weakness Nodes": None,  # Use cache
            "AttackPattern Nodes": None,  # Use cache
            "SubTechnique Nodes": None,  # Use cache
            "Ability Nodes": None,  # Use cache
            "Technique Nodes": None,  # Use cache
            "WorkRole Nodes": None,  # Use cache
            "Tactic Nodes": None,  # Use cache
            "SpecialtyArea Nodes": None,  # Use cache
        }

        # Cache lookup map
        cache_map = {
            "Weakness Nodes": ("Weakness", WEAKNESS_METADATA),
            "AttackPattern Nodes": ("AttackPattern", ATTACKPATTERN_METADATA),
            "SubTechnique Nodes": ("SubTechnique", SUBTECHNIQUE_METADATA),
            "Ability Nodes": ("Ability", ABILITY_METADATA),
            "Technique Nodes": ("Technique", TECHNIQUE_METADATA),
            "WorkRole Nodes": ("WorkRole", WORKROLE_METADATA),
            "Tactic Nodes": ("Tactic", TACTIC_METADATA),
            "SpecialtyArea Nodes": ("SpecialtyArea", SPECIALTYAREA_METADATA),
        }

        status = {}
        for metric, query in status_queries.items():
            try:
                # Use cached metadata for small datasets
                if query is None and metric in cache_map:
                    label, metadata = cache_map[metric]
                    status[metric] = metadata.get("total_count", 0)
                else:
                    # Query database for large/uncached datasets
                    result = self.execute_cypher_single(query)
                    status[metric] = result["count"] if result else 0
            except Exception as e:
                logger.error(f"Error getting {metric}: {e}")
                status[metric] = 0

        return status

    def validate_schema(self) -> ValidationResult:
        """Check required node labels, relationship types, and properties; return ValidationResult(valid, errors)."""
        errors = []

        # Required node labels (golden schema)
        required_labels = [
            "AttackPattern",
            "Weakness",
            "Technique",
            "SubTechnique",
            "Vulnerability",
            "WorkRole",
            "Category",
            "Infrastructure",
            "Service",
            "System",
        ]

        for label in required_labels:
            try:
                result = self.execute_cypher_single(
                    f"MATCH (n:{label}) RETURN count(n) as count"
                )
                if not result or result["count"] == 0:
                    errors.append(f"No {label} nodes found")
            except Exception as e:
                errors.append(f"Error checking {label} nodes: {e}")

        # Check for required relationship types
        required_rels = [
            "RELATES_TO",
            "WEAKNESS",
            "AFFECTS",
            "IN_TAXONOMY",
            "HAS_EMBEDDING",
        ]

        for rel_type in required_rels:
            try:
                result = self.execute_cypher_single(
                    f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                )
                if not result or result["count"] == 0:
                    errors.append(f"No {rel_type} relationships found")
            except Exception as e:
                errors.append(f"Error checking {rel_type} relationships: {e}")

        # Check for required properties
        required_props = ["uid", "name", "source", "ingested_at"]

        for label in required_labels:
            for prop in required_props:
                try:
                    result = self.execute_cypher_single(
                        f"MATCH (n:{label}) WHERE n.{prop} IS NULL RETURN count(n) as count"
                    )
                    if result and result["count"] > 0:
                        errors.append(f"{label} nodes missing {prop} property")
                except Exception as e:
                    errors.append(f"Error checking {label}.{prop}: {e}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    # --- Schema setup: constraints and indexes ---

    def create_constraints(self):
        """Create unique constraints on uid for main node labels (idempotent with IF NOT EXISTS)."""
        constraints = [
            "CREATE CONSTRAINT attack_pattern_uid IF NOT EXISTS FOR (n:AttackPattern) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT weakness_uid IF NOT EXISTS FOR (n:Weakness) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT technique_uid IF NOT EXISTS FOR (n:Technique) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT subtechnique_uid IF NOT EXISTS FOR (n:SubTechnique) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT vulnerability_uid IF NOT EXISTS FOR (n:Vulnerability) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT workrole_uid IF NOT EXISTS FOR (n:WorkRole) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT category_uid IF NOT EXISTS FOR (n:Category) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT infrastructure_uid IF NOT EXISTS FOR (n:Infrastructure) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT service_uid IF NOT EXISTS FOR (n:Service) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT system_uid IF NOT EXISTS FOR (n:System) REQUIRE n.uid IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                self.execute_cypher(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as e:
                logger.error(f"Error creating constraint: {e}")

    def create_indexes(self):
        """Create name indexes for main node labels (idempotent with IF NOT EXISTS)."""
        indexes = [
            "CREATE INDEX attack_pattern_name IF NOT EXISTS FOR (n:AttackPattern) ON (n.name)",
            "CREATE INDEX weakness_name IF NOT EXISTS FOR (n:Weakness) ON (n.name)",
            "CREATE INDEX technique_name IF NOT EXISTS FOR (n:Technique) ON (n.name)",
            "CREATE INDEX vulnerability_name IF NOT EXISTS FOR (n:Vulnerability) ON (n.name)",
            "CREATE INDEX workrole_name IF NOT EXISTS FOR (n:WorkRole) ON (n.name)",
            "CREATE INDEX category_name IF NOT EXISTS FOR (n:Category) ON (n.name)",
            "CREATE INDEX infrastructure_name IF NOT EXISTS FOR (n:Infrastructure) ON (n.name)",
            "CREATE INDEX service_name IF NOT EXISTS FOR (n:Service) ON (n.name)",
            "CREATE INDEX system_name IF NOT EXISTS FOR (n:System) ON (n.name)",
        ]

        for index in indexes:
            try:
                self.execute_cypher(index)
                logger.info(f"Created index: {index}")
            except Exception as e:
                logger.error(f"Error creating index: {e}")

    def __enter__(self):
        """Context manager entry; returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit; closes the driver."""
        self.close()
