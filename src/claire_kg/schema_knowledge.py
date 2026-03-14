#!/usr/bin/env python3
"""
Dynamic schema knowledge: discover Neo4j schema at runtime for Cypher generation.

Uses db.labels(), db.relationshipTypes(), and sample nodes to build node types,
relationships, field mappings, and query patterns. Optional integration with
dataset_metadata for sample counts and WorkRole relationship stats. Used to
produce schema prompts (get_schema_prompt) and to generate simple Cypher
(generate_cypher_query, detect_dataset_type). Alternative to static curated
schema in curated_schema_builder / CypherGenerator.

Module layout: NodeTypeInfo, RelationshipInfo dataclasses → DynamicSchemaKnowledgeSystem
(connect, discover, get_* accessors, get_schema_prompt, generate_cypher_query).
"""

import json
import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# -----------------------------------------------------------------------------
# NodeTypeInfo, RelationshipInfo, DynamicSchemaKnowledgeSystem
# -----------------------------------------------------------------------------


@dataclass
class NodeTypeInfo:
    """Information about a node type in the schema."""

    label: str
    primary_id_field: str
    common_fields: List[str]
    relationships: List[str]
    description: str
    sample_count: int = 0


@dataclass
class RelationshipInfo:
    """Information about a relationship in the schema."""

    name: str
    from_node: str
    to_node: str
    description: str
    count: int = 0


class DynamicSchemaKnowledgeSystem:
    """Discovers and caches Neo4j schema (labels, properties, relationships) for prompts and Cypher."""

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        verbose: bool = True,
    ):
        """Initialize with Neo4j connection parameters.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            verbose: If False, suppress discovery messages (default: True)
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv(
            "NEO4J_PASSWORD", "graphrag-password"
        )
        self.verbose = verbose

        self.driver = None
        self.schema_cache = {}
        self._connect()
        self._discover_schema()

    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            if self.verbose:
                print("OK: Connected to Neo4j for schema discovery")
        except (ServiceUnavailable, AuthError) as e:
            if self.verbose:
                print(f"ERROR: Failed to connect to Neo4j: {e}")
            raise

    def _discover_schema(self):
        """Discover labels, properties, relationships; build schema_knowledge (node_types, field_mappings, query_patterns)."""
        if self.verbose:
            print(" Discovering database schema...")

        node_labels = self._get_node_labels()
        node_properties = self._get_node_properties(node_labels)
        relationships = self._get_relationships()

        # Build schema knowledge
        self.schema_knowledge = {
            "node_types": {},
            "relationships": {},
            "field_mappings": {},
            "query_patterns": self._get_query_patterns(),
        }

        # Process each node label
        for label in node_labels:
            properties = node_properties.get(label, [])

            # Determine primary ID field
            primary_id_field = self._determine_primary_id_field(properties)

            # Get common fields
            common_fields = self._get_common_fields(properties)

            # Get sample count - use cached metadata for small datasets, query for large ones
            sample_count = self._get_sample_count(label)

            # Create node type info
            self.schema_knowledge["node_types"][label] = NodeTypeInfo(
                label=label,
                primary_id_field=primary_id_field,
                common_fields=common_fields,
                relationships=[],  # Relationships are stored separately
                description=self._get_description(label),
                sample_count=sample_count,
            )

            # Create field mappings
            self.schema_knowledge["field_mappings"][label] = (
                self._create_field_mappings(properties)
            )

        # Process relationships
        for rel_name, rel_info in relationships.items():
            if isinstance(rel_info, dict):
                self.schema_knowledge["relationships"][rel_name] = RelationshipInfo(
                    name=rel_name,
                    from_node=rel_info.get("from_node", ""),
                    to_node=rel_info.get("to_node", ""),
                    description=rel_info.get("description", ""),
                    count=rel_info.get("count", 0),
                )

        if self.verbose:
            print(
                f"OK: Discovered {len(node_labels)} node types and {len(relationships)} relationship types"
            )

    def _get_node_labels(self) -> List[str]:
        """Get all node labels from the database."""
        with self.driver.session() as session:
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            return sorted(labels)

    def _get_node_properties(self, labels: List[str]) -> Dict[str, List[str]]:
        """Get properties for each node label."""
        properties = {}

        for label in labels:
            with self.driver.session() as session:
                # Get sample node to discover properties
                query = f"MATCH (n:{label}) RETURN n LIMIT 1"
                result = session.run(query)
                record = result.single()

                if record:
                    node_data = record["n"]
                    properties[label] = list(node_data.keys())
                else:
                    properties[label] = []

        return properties

    def _get_relationships(self) -> Dict[str, Any]:
        """Get all relationship types and their patterns."""
        relationships = {}

        with self.driver.session() as session:
            # Get all relationship types
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]

            for rel_type in rel_types:
                # Get relationship patterns
                query = f"""
                MATCH (a)-[r:{rel_type}]->(b)
                RETURN DISTINCT labels(a) as from_labels, labels(b) as to_labels, count(r) as count
                LIMIT 10
                """
                result = session.run(query)

                rel_patterns = []
                for record in result:
                    rel_patterns.append(
                        {
                            "from_node": (
                                record["from_labels"][0]
                                if record["from_labels"]
                                else "Unknown"
                            ),
                            "to_node": (
                                record["to_labels"][0]
                                if record["to_labels"]
                                else "Unknown"
                            ),
                            "count": record["count"],
                        }
                    )

                if rel_patterns:
                    relationships[rel_type] = rel_patterns[
                        0
                    ]  # Take the most common pattern

        return relationships

    def _determine_primary_id_field(self, properties: List[str]) -> str:
        """Determine the primary ID field from available properties."""
        # Priority order for ID fields
        id_candidates = ["uid", "id", "name", "title", "identifier"]

        for candidate in id_candidates:
            if candidate in properties:
                return candidate

        # Fallback to first property
        return properties[0] if properties else "id"

    def _get_common_fields(self, properties: List[str]) -> List[str]:
        """Get common fields for display, prioritizing important ones."""
        # Priority order for common fields
        priority_fields = [
            "uid",
            "id",
            "name",
            "title",
            "description",
            "descriptions",
            "severity",
            "cvss_v31",
            "cvss_v30",
            "published",
            "modified",
            "category",
            "tactic",
            "platform",
            "work_role",
            "specialty_area",
        ]

        common_fields = []

        # Add priority fields that exist
        for field in priority_fields:
            if field in properties:
                common_fields.append(field)

        # Add remaining fields
        for field in properties:
            if field not in common_fields:
                common_fields.append(field)

        return common_fields[:10]  # Limit to 10 most relevant fields

    def _get_sample_count(self, label: str) -> int:
        """Get sample count for a label - uses cached metadata for small datasets."""
        try:
            from .dataset_metadata import get_dataset_metadata, is_small_dataset

            if is_small_dataset(label):
                metadata = get_dataset_metadata(label)
                if metadata and "total_count" in metadata:
                    return metadata["total_count"]
        except (ImportError, Exception):
            pass

        with self.driver.session() as session:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

    def _get_description(self, label: str) -> str:
        """Get description for a label based on common patterns."""
        descriptions = {
            "Vulnerability": "Common Vulnerabilities and Exposures (CVE)",
            "Weakness": "Common Weakness Enumeration (CWE)",
            "AttackPattern": "Common Attack Pattern Enumeration (CAPEC)",
            "Technique": "MITRE ATT&CK Framework Techniques",
            "SubTechnique": "MITRE ATT&CK Sub-Techniques",
            "WorkRole": "Cybersecurity Work Roles",
            "Category": "Taxonomy Categories",
            "Infrastructure": "Infrastructure Components",
            "Service": "Service Components",
            "System": "System Components",
            "Task": "Work Tasks",
            "Knowledge": "Knowledge Areas",
            "Skill": "Skills",
            "CompetencyArea": "Competency Areas",
            "KSA": "Knowledge, Skills, Abilities",
            "Element": "Framework Elements",
            "DCWFCode": "DCWF Codes",
            "Ability": "Abilities",
            "SpecialtyArea": "Specialty Areas",
            "Mitigation": "Security Mitigations",
            "Asset": "Infrastructure Assets",
            "Tactic": "ATT&CK Tactics",
        }
        return descriptions.get(label, f"{label} entities")

    def _create_field_mappings(self, properties: List[str]) -> Dict[str, List[str]]:
        """Create field mappings for a node type."""
        mappings = {}

        # Common field mappings
        field_mappings = {
            "id": ["uid", "id", "identifier"],
            "name": ["name", "title", "work_role"],
            "description": ["description", "descriptions", "definition"],
            "severity": ["severity", "cvss_v31", "cvss_v30"],
            "category": ["category", "tactic", "specialty_area"],
            "published": ["published", "created", "modified"],
        }

        for field_type, candidates in field_mappings.items():
            for candidate in candidates:
                if candidate in properties:
                    mappings[field_type] = [candidate]
                    break

        return mappings

    def _get_query_patterns(self) -> Dict[str, str]:
        """Get query patterns for different types of queries."""
        return {
            "single_entity": "MATCH (n:{label} {{{id_field}: '{id}'}}) RETURN n.{fields}",
            "entity_list": "MATCH (n:{label}) RETURN n.{fields} LIMIT {limit}",
            "entity_search": "MATCH (n:{label}) WHERE n.{field} CONTAINS '{search_term}' RETURN n.{fields}",
            "analytical_count": "MATCH (n:{label}) WHERE n.{field} = {value} RETURN COUNT(n) as count",
            "analytical_year": "MATCH (n:{label}) WHERE n.year = {year} RETURN COUNT(n) as count",
            "crosswalk_cve_cwe": "MATCH (v:Vulnerability {{uid: '{cve_id}'}})-[:HAS_WEAKNESS]->(w:Weakness) RETURN w.{fields}",
            "crosswalk_cve_asset": "MATCH (v:Vulnerability {{uid: '{cve_id}'}})-[:AFFECTS]->(a:Asset) RETURN a.{fields}",
            "crosswalk_cwe_cve": "MATCH (w:Weakness {{uid: '{cwe_id}'}})<-[:HAS_WEAKNESS]-(v:Vulnerability) RETURN v.{fields}",
            "multi_hop": "MATCH (a:{start_label})-[:{rel1}]->(b:{mid_label})-[:{rel2}]->(c:{end_label}) WHERE a.{id_field} = '{id}' RETURN c.{fields}",
        }

    def get_node_type_info(self, dataset_type: str) -> Optional[NodeTypeInfo]:
        """Get node type information for a dataset."""
        return self.schema_knowledge["node_types"].get(dataset_type)

    def get_field_mappings(self, dataset_type: str) -> Dict[str, List[str]]:
        """Get field mappings for a dataset."""
        return self.schema_knowledge["field_mappings"].get(dataset_type, {})

    def get_relationship_info(
        self, relationship_name: str
    ) -> Optional[RelationshipInfo]:
        """Get relationship information."""
        return self.schema_knowledge["relationships"].get(relationship_name)

    def get_correct_label(self, dataset_type: str) -> str:
        """Get the correct Neo4j label for a dataset type."""
        node_info = self.get_node_type_info(dataset_type)
        return node_info.label if node_info else dataset_type

    def get_primary_id_field(self, dataset_type: str) -> str:
        """Get the primary ID field for a dataset."""
        node_info = self.get_node_type_info(dataset_type)
        return node_info.primary_id_field if node_info else "uid"

    def get_common_fields(self, dataset_type: str) -> List[str]:
        """Get common fields for a dataset."""
        node_info = self.get_node_type_info(dataset_type)
        return node_info.common_fields if node_info else ["uid", "name", "description"]

    def detect_dataset_type(self, query: str) -> Optional[str]:
        """Detect dataset type from query text."""
        query_lower = query.lower()

        # CVE detection
        if any(
            term in query_lower
            for term in ["cve-", "vulnerability", "cvss", "severity"]
        ):
            return "Vulnerability"

        # CWE detection
        if any(term in query_lower for term in ["cwe-", "weakness", "weaknesses"]):
            return "Weakness"

        # CAPEC detection
        if any(
            term in query_lower
            for term in ["capec-", "attack pattern", "attack patterns"]
        ):
            return "AttackPattern"

        # ATT&CK detection
        if any(
            term in query_lower
            for term in ["attack", "technique", "tactic", "t1059", "t1566"]
        ):
            return "Technique"

        # NICE detection
        if any(
            term in query_lower
            for term in ["nice", "work role", "work roles", "cyber defense analyst"]
        ):
            return "WorkRole"

        # DCWF detection
        if any(
            term in query_lower
            for term in ["dcwf", "specialty area", "forensics analyst"]
        ):
            return "SpecialtyArea"

        return None

    def generate_cypher_query(
        self, query: str, dataset_type: str = None, limit: int = 10
    ) -> str:
        """Generate proper Cypher query using schema knowledge."""
        if not dataset_type:
            dataset_type = self.detect_dataset_type(query)

        if not dataset_type:
            return self._generate_generic_query(query, limit)

        # Get schema information
        label = self.get_correct_label(dataset_type)
        id_field = self.get_primary_id_field(dataset_type)
        common_fields = self.get_common_fields(dataset_type)

        # Generate query based on query type
        if f"{dataset_type}-" in query.upper() or any(
            term in query.upper() for term in ["CVE-", "CWE-", "CAPEC-", "T"]
        ):
            # Specific entity query
            entity_id = self._extract_entity_id(query, dataset_type)
            if entity_id:
                fields_str = ", ".join([f"n.{field}" for field in common_fields])
                return f"MATCH (n:{label} {{{id_field}: '{entity_id}'}}) RETURN {fields_str} LIMIT 1"

        # Generic search query
        fields_str = ", ".join([f"n.{field}" for field in common_fields])
        return f"MATCH (n:{label}) RETURN {fields_str} LIMIT {limit}"

    def _extract_entity_id(self, query: str, dataset_type: str) -> Optional[str]:
        """Extract entity ID from query."""
        import re

        if dataset_type == "Vulnerability":
            match = re.search(r"CVE-\d{4}-\d+", query.upper())
            return match.group(0) if match else None
        elif dataset_type == "Weakness":
            match = re.search(r"CWE-\d+", query.upper())
            return match.group(0) if match else None
        elif dataset_type == "AttackPattern":
            match = re.search(r"CAPEC-\d+", query.upper())
            return match.group(0) if match else None
        elif dataset_type == "Technique":
            match = re.search(r"T\d+", query.upper())
            return match.group(0) if match else None

        return None

    def _generate_generic_query(self, query: str, limit: int) -> str:
        """Generate generic query when dataset type can't be detected."""
        return f"MATCH (n) WHERE n.uid IS NOT NULL RETURN n.uid, n.name, n.description LIMIT {limit}"

    def get_schema_prompt(self) -> str:
        """Get schema information as a prompt for LangChain."""
        schema_info = []

        schema_info.append(
            "CLAIRE-KG Database Schema (DISCOVERED FROM ACTUAL DATABASE):"
        )
        schema_info.append("=" * 70)

        for dataset_type, node_info in self.schema_knowledge["node_types"].items():
            schema_info.append(f"\n{dataset_type} ({node_info.description}):")
            schema_info.append(f"  Label: {node_info.label}")
            schema_info.append(f"  Primary ID: {node_info.primary_id_field}")
            schema_info.append(f"  Sample Count: {node_info.sample_count}")
            schema_info.append(
                f"  Available Properties: {', '.join(node_info.common_fields[:10])}"
            )
            # Emphasize: ONLY use these properties - do not invent property names
            schema_info.append(
                f"  WARNING: STRICT: ONLY use the properties listed above. Do NOT use properties like 'title', 'name', 'description' unless they appear in the list above!"
            )
            schema_info.append(
                f"  Relationships: {', '.join(node_info.relationships[:3])}"
            )
            # Add explicit examples for critical node types
            if node_info.label == "Asset":
                schema_info.append(
                    "  WARNING: CRITICAL: Use 'vendor' property (NOT 'product') for vendor filtering"
                )
                schema_info.append(
                    "     Example: WHERE toLower(a.vendor) = 'microsoft' (NOT a.product = 'Microsoft')"
                )
            elif node_info.label == "AttackPattern":
                schema_info.append(
                    "  WARNING: CRITICAL: Use 'uid' and 'name' properties (NOT 'element_code' or 'element_name')"
                )
                schema_info.append(
                    "     Example: RETURN ap.uid, ap.name (NOT coalesce(ap.element_code, ap.element_name))"
                )
            elif node_info.label == "Category":
                schema_info.append(
                    "  WARNING: CRITICAL: Check available category names - they may differ from expected values"
                )
            elif node_info.label == "WorkRole":
                schema_info.append(
                    "  WARNING: CRITICAL: Use 'work_role' property for role names (NOT 'title')"
                )
                schema_info.append(
                    "     Example: WHERE wr.work_role = 'System Administrator' (NOT wr.title = 'System Administrator')"
                )
                schema_info.append(
                    "     Example: MATCH (wr:WorkRole {work_role: 'System Administrator'}) ..."
                )
                # Add relationship statistics from cache to guide query generation
                try:
                    from .dataset_metadata import get_workrole_relationship_stats

                    rel_stats = get_workrole_relationship_stats()
                    if rel_stats:
                        schema_info.append("  Relationship Availability:")
                        schema_info.append(
                            f"    - PERFORMS: Only {rel_stats.get('PERFORMS', {}).get('percent', 0):.1f}% of WorkRoles have Tasks"
                        )
                        schema_info.append(
                            "      WARNING: IMPORTANT: For 'What does a [WorkRole] do?' questions, return the WorkRole definition directly"
                        )
                        schema_info.append(
                            "      (use wr.definition or wr.text), NOT related Tasks. Only 51% of roles have PERFORMS relationships!"
                        )
                        schema_info.append(
                            "      CORRECT: MATCH (wr:WorkRole {work_role: 'X'}) RETURN wr.uid, wr.work_role, wr.definition"
                        )
                        schema_info.append(
                            "      WRONG: MATCH (wr:WorkRole)-[:PERFORMS]->(t:Task) ... (may return 0 results!)"
                        )
                except (ImportError, Exception):
                    pass  # Cache unavailable, skip relationship stats
            elif node_info.label == "Task":
                schema_info.append(
                    "  WARNING: CRITICAL: Use 'title' or 'text' properties (may vary, use coalesce for robustness)"
                )

        # Add relationship information
        schema_info.append("\n" + "=" * 70)
        schema_info.append("AVAILABLE RELATIONSHIPS:")
        schema_info.append("=" * 70)

        for rel_name, rel_info in self.schema_knowledge["relationships"].items():
            schema_info.append(
                f"{rel_name}: {rel_info.from_node} -> {rel_info.to_node} (count: {rel_info.count})"
            )
            schema_info.append(
                f"  Example: (from:{rel_info.from_node})-[:{rel_name}]->(to:{rel_info.to_node})"
            )

        schema_info.append("\n" + "=" * 70)
        schema_info.append(" CRITICAL PROPERTY USAGE RULES:")
        schema_info.append("=" * 70)
        schema_info.append(
            "1. ONLY use properties that appear in 'Available Properties' for each node type"
        )
        schema_info.append(
            "2. If a property is NOT listed, it does NOT exist - do NOT use it!"
        )
        schema_info.append(
            "3. For WorkRole nodes: Use 'work_role' (NOT 'title') - check Available Properties to confirm"
        )
        schema_info.append(
            "4. Always use the EXACT property names as shown - they are case-sensitive"
        )
        schema_info.append(
            "5. NEVER invent or guess property names - use ONLY what is discovered from the database"
        )
        schema_info.append("")
        schema_info.append(
            "CRITICAL: Always use the EXACT field names and relationships from the schema above!"
        )
        schema_info.append("NEVER invent or guess field names or relationship names!")
        schema_info.append(
            "IMPORTANT: For relationship directions, use the examples above:"
        )
        schema_info.append(
            "  - If relationship shows 'A -> B', use (a:A)-[:REL]->(b:B)"
        )
        schema_info.append(
            "  - If relationship shows 'A -> B', use (b:B)<-[:REL]-(a:A) for reverse traversal"
        )
        schema_info.append("Examples:")
        schema_info.append("  - CWE entities use label 'Weakness', not 'CWE'")
        schema_info.append("  - CVE entities use label 'Vulnerability', not 'CVE'")
        schema_info.append("  - Use 'uid' field for entity IDs")
        schema_info.append("  - Use 'descriptions' field for CVE descriptions")
        schema_info.append(
            "  - Use 'USES_TACTIC' relationship for ATT&CK techniques and tactics"
        )
        schema_info.append(
            "  - For sub-techniques: (t:Technique)<-[:IS_PART_OF]-(st:SubTechnique)"
        )
        schema_info.append(
            "  - Use 'published' field for publication dates (NOT 'published_date')"
        )
        schema_info.append(
            "  - Use 'year' field for year queries (NOT 'published_date')"
        )
        schema_info.append("  - Use 'cvss_v31' field for CVSS scores")

        # Add explicit field mappings for common mistakes
        schema_info.append("\nCOMMON FIELD MAPPINGS (USE THESE EXACT NAMES):")
        schema_info.append("  - Publication date: 'published' (NOT 'published_date')")
        schema_info.append("  - Year filtering: 'year' (NOT 'published_date')")
        schema_info.append("  - CVSS score: 'cvss_v31' (NOT 'cvss_score')")
        schema_info.append("  - CVE descriptions: 'descriptions' (NOT 'description')")
        schema_info.append("  - CWE descriptions: 'description' (NOT 'descriptions')")

        # Add cross-dataset relationship limitations
        schema_info.append("\nCROSS-DATASET RELATIONSHIP LIMITATIONS:")
        schema_info.append(
            "  - CWE weaknesses do NOT directly relate to ATT&CK techniques"
        )
        schema_info.append(
            "  - CVE vulnerabilities do NOT directly relate to ATT&CK techniques"
        )
        schema_info.append(
            "  - CAPEC to ATT&CK: Use (ap:AttackPattern)-[:RELATES_TO]->(t:Technique)"
        )
        schema_info.append(
            "    WARNING: Note: Relationship direction is AP -> Technique (NOT Technique -> AP)"
        )
        schema_info.append(
            "  - CAPEC to Tactic: Go through Technique first: (ap:AttackPattern)-[:RELATES_TO]->(t:Technique)-[:USES_TACTIC]->(ta:Tactic)"
        )
        schema_info.append(
            "  - For CWE to ATT&CK queries: Return 'No direct relationships found'"
        )
        schema_info.append(
            "  - For CVE to ATT&CK queries: Return 'No direct relationships found'"
        )
        schema_info.append(
            "  - For CWE queries: Always use {uid: 'CWE-XXX'} not {name: '...'}"
        )
        schema_info.append(
            "  - For CVE queries: Always use {uid: 'CVE-YYYY-NNNNN'} not {name: '...'}"
        )
        schema_info.append(
            "  - For CAPEC queries: Always use {uid: 'CAPEC-XXX'} not {name: '...'}"
        )
        schema_info.append(
            "  - For ATT&CK queries: Always use {uid: 'TXXXX'} not {name: '...'}"
        )
        try:
            with self.driver.session() as session:
                # Check CWE-ATT&CK relationships
                cwe_attack_result = session.run(
                    "MATCH (w:Weakness)-[r]->(t:Technique) RETURN count(r) as count"
                ).single()
                cwe_attack_count = (
                    cwe_attack_result["count"] if cwe_attack_result else 0
                )

                # Check CAPEC-ATT&CK relationships
                capec_attack_result = session.run(
                    "MATCH (ap:AttackPattern)-[r:RELATES_TO]->(t:Technique) RETURN count(r) as count"
                ).single()
                capec_attack_count = (
                    capec_attack_result["count"] if capec_attack_result else 0
                )

                # Check CVE-ATT&CK relationships
                cve_attack_result = session.run(
                    "MATCH (v:Vulnerability)-[r]->(t:Technique) RETURN count(r) as count"
                ).single()
                cve_attack_count = (
                    cve_attack_result["count"] if cve_attack_result else 0
                )

                schema_info.append(f"  - CWE-ATT&CK relationships: {cwe_attack_count}")
                schema_info.append(
                    f"  - CAPEC-ATT&CK relationships: {capec_attack_count}"
                )
                schema_info.append(f"  - CVE-ATT&CK relationships: {cve_attack_count}")

        except Exception as e:
            schema_info.append(
                f"  - Crosswalk relationship counts: Unable to determine ({str(e)})"
            )

        # Add actual lookup values for common queries
        schema_info.append("\n" + "=" * 70)
        schema_info.append("ACTUAL LOOKUP VALUES (USE THESE EXACT VALUES):")
        schema_info.append("=" * 70)
        try:
            with self.driver.session() as session:
                # Get actual Category names
                cat_result = session.run(
                    "MATCH (c:Category) RETURN c.name AS name ORDER BY c.name LIMIT 15"
                )
                categories = [r["name"] for r in cat_result] if cat_result else []
                if categories:
                    schema_info.append(
                        f"  Category names (for CAPEC): {', '.join(categories)}"
                    )
                    schema_info.append(
                        "    WARNING: Do NOT use category names like 'Injection' - use actual names above"
                    )

                # Get sample vendor values for Asset nodes
                vendor_result = session.run(
                    "MATCH (a:Asset) WHERE a.vendor IS NOT NULL RETURN DISTINCT a.vendor AS vendor ORDER BY vendor LIMIT 10"
                )
                vendors = [r["vendor"] for r in vendor_result] if vendor_result else []
                if vendors:
                    schema_info.append(
                        f"  Sample Asset vendors: {', '.join(vendors[:5])}..."
                    )
                    schema_info.append(
                        "    WARNING: For vendor filtering: Use a.vendor (NOT a.product)"
                    )

        except Exception as e:
            schema_info.append(f"  Unable to fetch lookup values: {str(e)}")

        # Workforce-specific dynamic hints (no hardcoding)
        schema_info.append("\n" + "=" * 70)
        schema_info.append("WORKFORCE SCHEMA HINTS (DYNAMIC):")
        schema_info.append("=" * 70)
        try:
            with self.driver.session() as session:

                def props_for(label: str) -> str:
                    """Return comma-separated property names for label (from one sample node)."""
                    rec = session.run(
                        f"MATCH (n:{label}) RETURN keys(n) AS props LIMIT 1"
                    ).single()
                    return ", ".join(rec["props"]) if rec and rec["props"] else "(none)"

                wr_props = props_for("WorkRole")
                task_props = props_for("Task")
                kn_props = props_for("Knowledge")
                sa_props = props_for("SpecialtyArea")

                schema_info.append(f"  - WorkRole props: {wr_props}")
                schema_info.append(f"  - Task props: {task_props}")
                schema_info.append(f"  - Knowledge props: {kn_props}")
                schema_info.append(f"  - SpecialtyArea props: {sa_props}")

                # Discover existing relationship types from WorkRole
                rels = session.run(
                    "MATCH (:WorkRole)-[r]->() RETURN DISTINCT type(r) AS t ORDER BY t"
                ).values("t")
                rel_list = ", ".join([r for r in rels]) if rels else "(none)"
                schema_info.append(f"  - WorkRole outgoing rels: {rel_list}")

                def rel_count(a: str, rel: str, b: str) -> int:
                    """Return count of (a)-[:rel]->(b) for workforce hint section."""
                    rec = session.run(
                        f"MATCH (:{a})-[:{rel}]->(:{b}) RETURN count(*) AS c"
                    ).single()
                    return rec["c"] if rec else 0

                for trip in [
                    ("WorkRole", "PERFORMS", "Task"),
                    ("WorkRole", "REQUIRES_KNOWLEDGE", "Knowledge"),
                    ("WorkRole", "REQUIRES_SKILL", "Skill"),
                    ("WorkRole", "IN_SPECIALTY_AREA", "SpecialtyArea"),
                ]:
                    c = rel_count(*trip)
                    schema_info.append(
                        f"    · ({trip[0]})-[:{trip[1]}]->({trip[2]}): {c}"
                    )

        except Exception as e:
            schema_info.append(f"  - Workforce hints unavailable: {e}")

        # Query generation rules (robustness)
        schema_info.append("\nROBUST QUERY RULES:")
        schema_info.append(
            "  - Use IS NOT NULL for property existence (NOT exists(n.prop))"
        )
        schema_info.append(
            "  - Prefer discovered props (e.g., WorkRole.title/text) and coalesce where needed"
        )
        schema_info.append(
            "  - Use only discovered relationship types; fall back to OPTIONAL MATCH if unsure"
        )

        return "\n".join(schema_info)

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()


# Usage example
if __name__ == "__main__":
    schema_system = DynamicSchemaKnowledgeSystem()

    # Test schema knowledge
    print("Dynamic Schema Knowledge System Test")
    print("=" * 50)

    # Test CWE detection
    cwe_query = "What does CWE-79 describe?"
    dataset_type = schema_system.detect_dataset_type(cwe_query)
    print(f"CWE Query: {cwe_query}")
    print(f"Detected Dataset: {dataset_type}")
    print(f"Correct Label: {schema_system.get_correct_label(dataset_type)}")
    print(
        f"Generated Cypher: {schema_system.generate_cypher_query(cwe_query, dataset_type)}"
    )

    print("\n" + "=" * 50)

    # Test CVE detection
    cve_query = "What is the CVSS score of CVE-2024-20439?"
    dataset_type = schema_system.detect_dataset_type(cve_query)
    print(f"CVE Query: {cve_query}")
    print(f"Detected Dataset: {dataset_type}")
    print(f"Correct Label: {schema_system.get_correct_label(dataset_type)}")
    print(
        f"Generated Cypher: {schema_system.generate_cypher_query(cve_query, dataset_type)}"
    )

    print("\n" + "=" * 50)
    print("Schema Prompt for LangChain:")
    print(schema_system.get_schema_prompt())

    schema_system.close()
