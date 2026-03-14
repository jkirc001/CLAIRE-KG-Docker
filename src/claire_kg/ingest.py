"""
Dataset ingestion engine for CLAIRE-KG.

Loads cybersecurity datasets from JSON/JSONL/XML (typically under a Docker
volume at /import) into Neo4j using APOC and Cypher. Used by the CLI
`setup ingest` and `setup ingest-all` commands.

Datasets: CAPEC (JSONL or XML), CWE, ATT&CK (enterprise), CVE (NVD), NICE,
DCWF. Each has a dedicated _ingest_* method; ingest_dataset() dispatches by
name and ensures constraints/indexes exist first.

Crosswalks: create_crosswalk() creates relationships between datasets (e.g.
CVE↔CWE, CAPEC↔ATT&CK, DCWF↔NICE, workrole↔attack). Crosswalk types are
validated and dispatched to _create_*_crosswalk() methods.

Embeddings: generate_embeddings() optionally computes and stores vector
embeddings on nodes (e.g. for RAG); uses OpenAI or sentence-transformers.

Result types: IngestionResult (from database.py), CrosswalkResult,
EmbeddingResult.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .database import Neo4jConnection, IngestionResult

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Result types for crosswalk and embedding operations
# -----------------------------------------------------------------------------


@dataclass
class CrosswalkResult:
    """Result of crosswalk creation"""

    success: bool
    relationships_created: int = 0
    error: Optional[str] = None


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""

    success: bool
    nodes_processed: int = 0
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# DatasetIngester: ingest datasets, create crosswalks, generate embeddings
# -----------------------------------------------------------------------------


class DatasetIngester:
    """Loads datasets from /import (Docker volume) into Neo4j and creates crosswalk relationships."""

    def __init__(
        self, db: Neo4jConnection, batch_size: int = 1000, verbose: bool = False
    ):
        """Initialize ingester with DB connection, APOC batch size, and verbosity."""
        self.db = db
        self.batch_size = batch_size
        self.verbose = verbose

        # Paths under Docker volume /import (used by APOC load)
        self.dataset_files = {
            "capec": "/import/capec/capec-dictionary.jsonl",
            "cwe": "/import/cwe/cwec_v4.18.jsonl",
            "attack": "/import/attack/enterprise-attack.jsonl",
            "cve": "/import/cve/nvdcve-2.0-2024.jsonl",
            "nice": "/import/nice/nice_components.jsonl",
            "dcwf": "/import/dcwf",
        }

    def ingest_dataset(self, dataset: str) -> IngestionResult:
        """Ingest one dataset by name (capec, cwe, attack, cve, nice, dcwf). Creates constraints/indexes first, then dispatches to _ingest_*."""
        try:
            if dataset not in self.dataset_files:
                return IngestionResult(
                    success=False, error=f"Unknown dataset: {dataset}"
                )

            # Create constraints and indexes
            self.db.create_constraints()
            self.db.create_indexes()

            # Ingest based on dataset type
            file_path = Path(self.dataset_files[dataset])
            if dataset == "capec":
                # Check if XML file exists, use XML method if available
                xml_file_path = Path("/import/capec/capec_latest.xml")
                if xml_file_path.exists():
                    return self._ingest_capec_xml(xml_file_path)
                else:
                    return self._ingest_capec(file_path)
            elif dataset == "cwe":
                return self._ingest_cwe(file_path)
            elif dataset == "attack":
                return self._ingest_attack(file_path)
            elif dataset == "cve":
                return self._ingest_cve(file_path)
            elif dataset == "nice":
                return self._ingest_nice(file_path)
            elif dataset == "dcwf":
                return self._ingest_dcwf(file_path)
            else:
                return IngestionResult(
                    success=False, error=f"Unsupported dataset: {dataset}"
                )

        except Exception as e:
            logger.error(f"Error ingesting {dataset}: {e}")
            return IngestionResult(success=False, error=str(e))

    # --- CAPEC (attack patterns): JSONL or XML ---

    def _ingest_capec(self, file_path: Path) -> IngestionResult:
        """Ingest CAPEC dataset"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        MERGE (ap:AttackPattern {uid: 'CAPEC-' + value.ID})
        SET ap.name = value.Name,
            ap.abstraction = value.Abstraction,
            ap.status = value.Status,
            ap.description = value.Description,
            ap.likelihood = value.Likelihood_of_Attack,
            ap.severity = value.Typical_Severity,
            ap.execution_flow = value.Execution_Flow,
            ap.prerequisites = value.Prerequisites,
            ap.mitigations = value.Mitigations,
            ap.taxonomy_mappings = value.Taxonomy_Mappings,
            ap.related_weaknesses = value.`Related Weaknesses`,
            ap.related_attack_patterns = value.`Related Attack Patterns`,
            ap.example_instances = value.`Example Instances`,
            ap.consequences = value.Consequences,
            ap.alternate_terms = value.`Alternate Terms`,
            ap.skills_required = value.`Skills Required`,
            ap.resources_required = value.`Resources Required`,
            ap.indicators = value.Indicators,
            ap.notes = value.Notes,
            ap.source = 'CAPEC',
            ap.ingested_at = datetime()
        RETURN count(ap) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        nodes_created = result["nodes_created"] if result else 0

        return IngestionResult(success=True, nodes_created=nodes_created)

    def _ingest_capec_xml(self, file_path: Path) -> IngestionResult:
        """Ingest CAPEC dataset from XML file with Categories and AttackPatterns"""
        try:
            # First, ingest Categories from XML
            categories_result = self._ingest_capec_categories_xml(file_path)
            if not categories_result.success:
                return categories_result

            # Then, ingest AttackPatterns from XML
            attack_patterns_result = self._ingest_capec_attack_patterns_xml(file_path)
            if not attack_patterns_result.success:
                return attack_patterns_result

            # Finally, create relationships between Categories and AttackPatterns
            relationships_result = self._ingest_capec_relationships_xml(file_path)
            if not relationships_result.success:
                return relationships_result

            total_nodes = (
                categories_result.nodes_created + attack_patterns_result.nodes_created
            )
            return IngestionResult(success=True, nodes_created=total_nodes)

        except Exception as e:
            logger.error(f"Error ingesting CAPEC XML: {e}")
            return IngestionResult(success=False, nodes_created=0, error=str(e))

    def _ingest_capec_categories_xml(self, file_path: Path) -> IngestionResult:
        """Ingest CAPEC Categories from XML file"""
        cypher = """
        CALL apoc.load.xml($file_path) YIELD value
        WITH value._children as children
        UNWIND children as child
        WITH child WHERE child._type = 'Categories'
        UNWIND child._children as category
        WITH category WHERE category._type = 'Category'
        MERGE (c:Category {uid: 'CAPEC-CAT-' + category.ID})
        SET c.name = category.Name,
            c.status = category.Status,
            c.summary = apoc.convert.toJson(category._children),
            c.source = 'CAPEC',
            c.ingested_at = datetime()
        RETURN count(c) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        nodes_created = result["nodes_created"] if result else 0

        return IngestionResult(success=True, nodes_created=nodes_created)

    def _ingest_capec_attack_patterns_xml(self, file_path: Path) -> IngestionResult:
        """Ingest CAPEC AttackPatterns from XML file"""
        cypher = """
        CALL apoc.load.xml($file_path, '/Attack_Pattern_Catalog/Attack_Patterns/Attack_Pattern') YIELD value
        MERGE (ap:AttackPattern {uid: 'CAPEC-' + value.ID})
        SET ap.name = value.Name,
            ap.abstraction = value.Abstraction,
            ap.status = value.Status,
            ap.description = value.Description,
            ap.likelihood_of_attack = value.Likelihood_Of_Attack,
            ap.typical_severity = value.Typical_Severity,
            ap.execution_flow = apoc.convert.toJson(value.Execution_Flow),
            ap.prerequisites = apoc.convert.toJson(value.Prerequisites),
            ap.mitigations = apoc.convert.toJson(value.Mitigations),
            ap.taxonomy_mappings = apoc.convert.toJson(value.Taxonomy_Mappings),
            ap.related_weaknesses = apoc.convert.toJson(value.Related_Weaknesses),
            ap.related_attack_patterns = apoc.convert.toJson(value.Related_Attack_Patterns),
            ap.example_instances = apoc.convert.toJson(value.Example_Instances),
            ap.consequences = apoc.convert.toJson(value.Consequences),
            ap.alternate_terms = apoc.convert.toJson(value.Alternate_Terms),
            ap.skills_required = apoc.convert.toJson(value.Skills_Required),
            ap.resources_required = apoc.convert.toJson(value.Required_Resources),
            ap.indicators = apoc.convert.toJson(value.Indicators),
            ap.notes = apoc.convert.toJson(value.Notes),
            ap.source = 'CAPEC',
            ap.ingested_at = datetime()
        RETURN count(ap) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        nodes_created = result["nodes_created"] if result else 0

        return IngestionResult(success=True, nodes_created=nodes_created)

    def _ingest_capec_relationships_xml(self, file_path: Path) -> IngestionResult:
        """Create relationships between CAPEC Categories and AttackPatterns from XML"""
        cypher = """
        CALL apoc.load.xml($file_path) YIELD value
        WITH value._children as children
        UNWIND children as child
        WITH child WHERE child._type = 'Categories'
        UNWIND child._children as category
        WITH category WHERE category._type = 'Category'
        UNWIND category._children as cat_child
        WITH category, cat_child WHERE cat_child._type = 'Relationships'
        UNWIND cat_child._children as rel
        WITH category, rel WHERE rel._type = 'Has_Member'
        MATCH (c:Category {uid: 'CAPEC-CAT-' + category.ID})
        MATCH (ap:AttackPattern {uid: 'CAPEC-' + rel.CAPEC_ID})
        MERGE (c)-[:HAS_MEMBER]->(ap)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        relationships_created = result["relationships_created"] if result else 0

        return IngestionResult(success=True, nodes_created=relationships_created)

    # --- CWE (weaknesses) ---

    def _ingest_cwe(self, file_path: Path) -> IngestionResult:
        """Ingest CWE dataset - FIXED to handle missing CWE IDs"""
        # First, try to ingest from file if it exists
        try:
            cypher = """
            CALL apoc.load.json($file_path) YIELD value
            MERGE (w:Weakness {uid: 'CWE-' + value.ID})
            SET w.name = value.Name,
                w.abstraction = value.Abstraction,
                w.status = value.Status,
                w.description = value.Description,
                w.extended_description = value.Extended_Description,
                w.likelihood_of_exploit = value.Likelihood_of_Exploit,
                w.common_consequences = value.Common_Consequences,
                w.background_details = value.Background_Details,
                w.applicable_platforms = value.Applicable_Platforms,
                w.functional_areas = value.Functional_Areas,
                w.affected_resources = value.Affected_Resources,
                w.potential_mitigations = value.Potential_Mitigations,
                w.detection_methods = value.Detection_Methods,
                w.modes_of_introduction = value.Modes_of_Introduction,
                w.exploitation_factors = value.Exploitation_Factors,
                w.related_weaknesses = value.Related_Weaknesses,
                w.weakness_ordinalities = value.Weakness_Ordinalities,
                w.alternate_terms = value.Alternate_Terms,
                w.observed_examples = value.Observed_Examples,
                w.taxonomy_mappings = value.Taxonomy_Mappings,
                w.notes = value.Notes,
                w.source = 'CWE',
                w.ingested_at = datetime()
            RETURN count(w) as nodes_created
            """
            result = self.db.execute_cypher_single(
                cypher, {"file_path": str(file_path)}
            )
            nodes_created = result["nodes_created"] if result else 0
        except Exception as e:
            logger.warning(f"CWE file ingestion failed: {e}")
            nodes_created = 0

        # Create missing CWE nodes that are referenced in CVEs but don't exist
        missing_cwe_cypher = """
        MATCH (v:Vulnerability)
        WHERE v.weaknesses IS NOT NULL AND v.weaknesses <> 'null'
        WITH v, v.weaknesses as weaknessStr
        WHERE weaknessStr =~ '.*CWE-[0-9]+.*'
        WITH v, weaknessStr
        UNWIND split(weaknessStr, 'CWE-') as part
        WITH v, part
        WHERE part =~ '^[0-9]+.*'
        WITH v, 'CWE-' + split(part, '\"')[0] as cwe_id
        WITH DISTINCT cwe_id
        WHERE NOT EXISTS {MATCH (w:Weakness {uid: cwe_id})}
        MERGE (w:Weakness {uid: cwe_id})
        SET w.name = 'Unknown CWE: ' + cwe_id,
            w.description = 'CWE referenced in CVE data but not found in CWE dataset',
            w.status = 'Unknown',
            w.source = 'CWE_MISSING',
            w.ingested_at = datetime()
        RETURN count(w) as missing_nodes_created
        """

        try:
            result = self.db.execute_cypher_single(missing_cwe_cypher)
            missing_nodes_created = result["missing_nodes_created"] if result else 0
            nodes_created += missing_nodes_created
            if missing_nodes_created > 0:
                logger.info(f"Created {missing_nodes_created} missing CWE nodes")
        except Exception as e:
            logger.error(f"Failed to create missing CWE nodes: {e}")

        return IngestionResult(success=True, nodes_created=nodes_created)

    # --- ATT&CK (MITRE enterprise) ---

    def _ingest_attack(self, file_path: Path) -> IngestionResult:
        """Ingest ATT&CK dataset with collapsed hierarchy (Technique, SubTechnique, Tactic → Technique)"""

        # First, clear existing ATT&CK data
        clear_cypher = """
        MATCH (n)
        WHERE n.source = 'ATT&CK'
        DETACH DELETE n
        """

        # Clear existing data
        if self.verbose:
            print("Clearing existing ATT&CK data...")
        self.db.execute_cypher(clear_cypher)

        # Parse STIX bundle and create nodes with proper labels
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        UNWIND value.objects as obj
        WITH obj, obj.external_references[0].external_id as external_id
        WHERE obj.type IN ['attack-pattern', 'x-mitre-tactic']
        AND external_id IS NOT NULL
        
        // Create Tactic nodes
        FOREACH (x IN CASE WHEN obj.type = 'x-mitre-tactic' THEN [1] ELSE [] END |
            CREATE (n:Tactic {uid: external_id})
            SET n.name = obj.name,
                n.description = obj.description,
                n.level = 'Tactic',
                n.x_mitre_domains = obj.x_mitre_domains,
                n.x_mitre_platforms = obj.x_mitre_platforms,
                n.source = 'ATT&CK',
                n.ingested_at = datetime()
        )
        
        // Create SubTechnique nodes
        FOREACH (x IN CASE WHEN obj.type = 'attack-pattern' AND obj.x_mitre_is_subtechnique = true THEN [1] ELSE [] END |
            CREATE (n:SubTechnique {uid: external_id})
            SET n.name = obj.name,
                n.description = obj.description,
                n.level = 'SubTechnique',
                n.x_mitre_domains = obj.x_mitre_domains,
                n.x_mitre_platforms = obj.x_mitre_platforms,
                n.source = 'ATT&CK',
                n.ingested_at = datetime()
        )
        
        // Create Technique nodes
        FOREACH (x IN CASE WHEN obj.type = 'attack-pattern' AND (obj.x_mitre_is_subtechnique IS NULL OR obj.x_mitre_is_subtechnique = false) THEN [1] ELSE [] END |
            CREATE (n:Technique {uid: external_id})
            SET n.name = obj.name,
                n.description = obj.description,
                n.level = 'Technique',
                n.x_mitre_domains = obj.x_mitre_domains,
                n.x_mitre_platforms = obj.x_mitre_platforms,
                n.source = 'ATT&CK',
                n.ingested_at = datetime()
        )
        
        RETURN count(*) as nodes_created
        """

        # Create relationships for ATT&CK hierarchy
        relationships_cypher = """
        // Create IS_PART_OF relationships for SubTechniques
        MATCH (sub:SubTechnique)
        WHERE sub.uid =~ 'T[0-9]+\\.[0-9]+'
        WITH sub, substring(sub.uid, 0, size(sub.uid) - 4) as parent_id
        MATCH (parent:Technique {uid: parent_id})
        MERGE (sub)-[:IS_PART_OF]->(parent)
        RETURN count(*) as relationships_created
        """

        # Create USES_TACTIC relationships between Techniques and Tactics
        tactic_relationships_cypher = """
        // Create USES_TACTIC relationships from ATT&CK kill_chain_phases
        CALL apoc.load.json($file_path) YIELD value
        UNWIND value.objects as obj
        WITH obj, obj.external_references[0].external_id as external_id
        WHERE obj.type = 'attack-pattern'
          AND (obj.x_mitre_is_subtechnique IS NULL OR obj.x_mitre_is_subtechnique = false)
          AND external_id IS NOT NULL
        UNWIND coalesce(obj.kill_chain_phases, []) as phase
        WITH external_id, phase
        WHERE phase.phase_name IS NOT NULL
          AND (phase.kill_chain_name IS NULL OR phase.kill_chain_name = 'mitre-attack')
        MATCH (t:Technique {uid: external_id})
        MATCH (tac:Tactic)
        WHERE toLower(tac.name) = replace(toLower(phase.phase_name), '-', ' ')
        MERGE (t)-[:USES_TACTIC {source: 'ATTACK_Mapping', confidence: 'high'}]->(tac)
        RETURN count(*) as tactic_relationships_created
        """

        try:
            # Run the main ingestion with batching
            if self.verbose:
                print(
                    f"Ingesting ATT&CK techniques with batch size {self.batch_size}..."
                )
            result = self.db.execute_cypher_single(
                cypher, {"file_path": str(file_path), "batch_size": self.batch_size}
            )
            nodes_created = result["nodes_created"] if result else 0

            # Create relationships
            if self.verbose:
                print("Creating ATT&CK relationships...")
            rel_result = self.db.execute_cypher_single(relationships_cypher)
            relationships_created = (
                rel_result["relationships_created"] if rel_result else 0
            )

            # Create USES_TACTIC relationships
            if self.verbose:
                print("Creating USES_TACTIC relationships...")
            tactic_rel_result = self.db.execute_cypher_single(
                tactic_relationships_cypher, {"file_path": str(file_path)}
            )
            tactic_relationships_created = (
                tactic_rel_result["tactic_relationships_created"]
                if tactic_rel_result
                else 0
            )

            total_relationships = relationships_created + tactic_relationships_created

            return IngestionResult(
                success=True,
                nodes_created=nodes_created,
                relationships_created=total_relationships,
            )

        except Exception as e:
            logger.error(f"ATT&CK ingestion error: {e}")
            return IngestionResult(
                success=False,
                nodes_created=0,
                relationships_created=0,
                error=str(e),
            )

    # --- CVE (NVD) ---

    def _ingest_cve(self, file_path: Path) -> IngestionResult:
        """Ingest CVE dataset - includes all CVEs but excludes them from isolated counts"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        WHERE value.id IS NOT NULL AND value.vulnStatus <> 'Rejected'
        MERGE (v:Vulnerability {uid: value.id})
        SET v.name = value.id,
            v.year = toInteger(split(value.id, '-')[1]),
            v.status = value.vulnStatus,
            v.published = value.published,
            v.lastModified = value.lastModified,
            v.descriptions = [desc IN value.descriptions WHERE desc.lang = 'en' | desc.value][0],
            
            // CVSS 2.0 data
            v.cvss_v2 = CASE 
                WHEN value.metrics.cvssMetricV2 IS NOT NULL AND size(value.metrics.cvssMetricV2) > 0 
                THEN value.metrics.cvssMetricV2[0].cvssData.baseScore 
                ELSE null END,
            v.severity_v2 = CASE 
                WHEN value.metrics.cvssMetricV2 IS NOT NULL AND size(value.metrics.cvssMetricV2) > 0 
                THEN value.metrics.cvssMetricV2[0].cvssData.baseSeverity 
                ELSE null END,
            
            // CVSS 3.0 data
            v.cvss_v30 = CASE 
                WHEN value.metrics.cvssMetricV30 IS NOT NULL AND size(value.metrics.cvssMetricV30) > 0 
                THEN value.metrics.cvssMetricV30[0].cvssData.baseScore 
                ELSE null END,
            v.severity_v30 = CASE 
                WHEN value.metrics.cvssMetricV30 IS NOT NULL AND size(value.metrics.cvssMetricV30) > 0 
                THEN value.metrics.cvssMetricV30[0].cvssData.baseSeverity 
                ELSE null END,
            
            // CVSS 3.1 data
            v.cvss_v31 = CASE 
                WHEN value.metrics.cvssMetricV31 IS NOT NULL AND size(value.metrics.cvssMetricV31) > 0 
                THEN value.metrics.cvssMetricV31[0].cvssData.baseScore 
                ELSE null END,
            v.severity_v31 = CASE 
                WHEN value.metrics.cvssMetricV31 IS NOT NULL AND size(value.metrics.cvssMetricV31) > 0 
                THEN value.metrics.cvssMetricV31[0].cvssData.baseSeverity 
                ELSE null END,
            
            // CVSS 4.0 data
            v.cvss_v40 = CASE 
                WHEN value.metrics.cvssMetricV40 IS NOT NULL AND size(value.metrics.cvssMetricV40) > 0 
                THEN value.metrics.cvssMetricV40[0].cvssData.baseScore 
                ELSE null END,
            v.severity_v40 = CASE 
                WHEN value.metrics.cvssMetricV40 IS NOT NULL AND size(value.metrics.cvssMetricV40) > 0 
                THEN value.metrics.cvssMetricV40[0].cvssData.baseSeverity 
                ELSE null END,
            
            // Legacy fields for backward compatibility (prioritize 3.1, then 4.0)
            v.cvss_score = CASE 
                WHEN value.metrics.cvssMetricV31 IS NOT NULL AND size(value.metrics.cvssMetricV31) > 0 
                THEN value.metrics.cvssMetricV31[0].cvssData.baseScore 
                WHEN value.metrics.cvssMetricV40 IS NOT NULL AND size(value.metrics.cvssMetricV40) > 0 
                THEN value.metrics.cvssMetricV40[0].cvssData.baseScore 
                ELSE null END,
            v.severity = CASE 
                WHEN value.metrics.cvssMetricV31 IS NOT NULL AND size(value.metrics.cvssMetricV31) > 0 
                THEN value.metrics.cvssMetricV31[0].cvssData.baseSeverity 
                WHEN value.metrics.cvssMetricV40 IS NOT NULL AND size(value.metrics.cvssMetricV40) > 0 
                THEN value.metrics.cvssMetricV40[0].cvssData.baseSeverity 
                ELSE null END,
            v.configurations = apoc.convert.toJson(value.configurations),
            v.weaknesses = apoc.convert.toJson(value.weaknesses),
            v.source = 'NVD',
            v.ingested_at = datetime()
        RETURN count(v) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        nodes_created = result["nodes_created"] if result else 0

        return IngestionResult(success=True, nodes_created=nodes_created)

    # --- NICE (work roles, tasks, knowledge, skills, relationships) ---

    def _ingest_nice(self, file_path: Path) -> IngestionResult:
        """Ingest NICE dataset with comprehensive element type handling"""
        try:
            # Step 1: Create constraints and indexes for NICE
            self._create_nice_constraints()
            self._create_nice_indexes()

            # Step 2: Ingest all NICE elements
            nodes_created = 0

            # Ingest Work Roles
            workrole_result = self._ingest_nice_workroles(file_path)
            nodes_created += workrole_result.get("nodes_created", 0)

            # Ingest Tasks
            task_result = self._ingest_nice_tasks(file_path)
            nodes_created += task_result.get("nodes_created", 0)

            # Ingest Knowledge
            knowledge_result = self._ingest_nice_knowledge(file_path)
            nodes_created += knowledge_result.get("nodes_created", 0)

            # Ingest Skills
            skill_result = self._ingest_nice_skills(file_path)
            nodes_created += skill_result.get("nodes_created", 0)

            # Step 3: Create relationships
            relationships_created = self._create_nice_relationships()

            return IngestionResult(
                success=True,
                nodes_created=nodes_created,
                relationships_created=relationships_created,
            )

        except Exception as e:
            return IngestionResult(success=False, error=str(e))

    def _create_nice_constraints(self):
        """Create NICE-specific constraints"""
        constraints = [
            "CREATE CONSTRAINT nice_workrole_element_identifier IF NOT EXISTS FOR (n:WorkRole) REQUIRE n.element_identifier IS UNIQUE;",
            "CREATE CONSTRAINT nice_task_element_identifier IF NOT EXISTS FOR (n:Task) REQUIRE n.element_identifier IS UNIQUE;",
            "CREATE CONSTRAINT nice_knowledge_element_identifier IF NOT EXISTS FOR (n:Knowledge) REQUIRE n.element_identifier IS UNIQUE;",
            "CREATE CONSTRAINT nice_skill_element_identifier IF NOT EXISTS FOR (n:Skill) REQUIRE n.element_identifier IS UNIQUE;",
        ]

        for constraint in constraints:
            try:
                self.db.execute_cypher(constraint)
            except Exception as e:
                logger.warning(f"Constraint creation failed (may already exist): {e}")

    def _create_nice_indexes(self):
        """Create NICE-specific indexes"""
        indexes = [
            "CREATE INDEX nice_workrole_source IF NOT EXISTS FOR (n:WorkRole) ON (n.source);",
            "CREATE INDEX nice_task_source IF NOT EXISTS FOR (n:Task) ON (n.source);",
            "CREATE INDEX nice_knowledge_source IF NOT EXISTS FOR (n:Knowledge) ON (n.source);",
            "CREATE INDEX nice_skill_source IF NOT EXISTS FOR (n:Skill) ON (n.source);",
        ]

        for index in indexes:
            try:
                self.db.execute_cypher(index)
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")

    def _ingest_nice_categories(self, file_path: Path) -> Dict[str, Any]:
        """Ingest NICE Categories"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        WHERE value.element_type = 'category'
        MERGE (cat:Category {element_identifier: value.element_identifier})
        SET cat.title = value.title,
            cat.text = value.text,
            cat.doc_identifier = value.doc_identifier,
            cat.source = 'NICE',
            cat.ingested_at = datetime()
        RETURN count(cat) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        return result or {"nodes_created": 0}

    def _ingest_nice_workroles(self, file_path: Path) -> Dict[str, Any]:
        """Ingest NICE Work Roles"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        WHERE value.element_type = 'work_role'
        MERGE (wr:WorkRole {element_identifier: value.element_identifier})
        SET wr.uid = value.element_identifier,
            wr.title = value.title,
            wr.text = value.text,
            wr.doc_identifier = value.doc_identifier,
            wr.source = 'NICE',
            wr.ingested_at = datetime()
        RETURN count(wr) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        return result or {"nodes_created": 0}

    def _ingest_nice_tasks(self, file_path: Path) -> Dict[str, Any]:
        """Ingest NICE Tasks"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        WHERE value.element_type = 'task'
        MERGE (t:Task {element_identifier: value.element_identifier})
        SET t.uid = value.element_identifier,
            t.title = value.title,
            t.text = value.text,
            t.doc_identifier = value.doc_identifier,
            t.source = 'NICE',
            t.ingested_at = datetime()
        RETURN count(t) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        return result or {"nodes_created": 0}

    def _ingest_nice_knowledge(self, file_path: Path) -> Dict[str, Any]:
        """Ingest NICE Knowledge"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        WHERE value.element_type = 'knowledge'
        MERGE (k:Knowledge {element_identifier: value.element_identifier})
        SET k.uid = value.element_identifier,
            k.title = value.title,
            k.text = value.text,
            k.doc_identifier = value.doc_identifier,
            k.source = 'NICE',
            k.ingested_at = datetime()
        RETURN count(k) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        return result or {"nodes_created": 0}

    def _ingest_nice_skills(self, file_path: Path) -> Dict[str, Any]:
        """Ingest NICE Skills"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        WHERE value.element_type = 'skill'
        MERGE (s:Skill {element_identifier: value.element_identifier})
        SET s.uid = value.element_identifier,
            s.title = value.title,
            s.text = value.text,
            s.doc_identifier = value.doc_identifier,
            s.source = 'NICE',
            s.ingested_at = datetime()
        RETURN count(s) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        return result or {"nodes_created": 0}

    def _ingest_nice_competency_areas(self, file_path: Path) -> Dict[str, Any]:
        """Ingest NICE Competency Areas"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        WHERE value.element_type = 'competency_area'
        MERGE (ca:CompetencyArea {element_identifier: value.element_identifier})
        SET ca.title = value.title,
            ca.text = value.text,
            ca.doc_identifier = value.doc_identifier,
            ca.source = 'NICE',
            ca.ingested_at = datetime()
        RETURN count(ca) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher, {"file_path": str(file_path)})
        return result or {"nodes_created": 0}

    def _create_nice_relationships(self) -> int:
        """Create NICE semantic relationships following DCWF pattern"""
        relationships_created = 0

        # Create SpecialtyArea nodes from WorkRole prefixes (e.g., OG, DD, etc.)
        cypher = """
        MATCH (wr:WorkRole)
        WHERE wr.source = 'NICE'
        WITH DISTINCT substring(wr.element_identifier, 0, 2) as specialty_prefix
        MERGE (sa:SpecialtyArea {specialty_prefix: specialty_prefix})
        SET sa.source = 'NICE',
            sa.ingested_at = datetime()
        RETURN count(sa) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("nodes_created", 0) if result else 0

        # Create IN_SPECIALTY_AREA relationships between WorkRoles and SpecialtyAreas
        cypher = """
        MATCH (wr:WorkRole), (sa:SpecialtyArea)
        WHERE wr.source = 'NICE' AND sa.source = 'NICE'
        AND substring(wr.element_identifier, 0, 2) = sa.specialty_prefix
        MERGE (wr)-[:IN_SPECIALTY_AREA]->(sa)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        # Create PERFORMS relationships between WorkRoles and Tasks
        # Hub-and-spoke: Each WorkRole gets a subset of Tasks (about 20-25 tasks per work role)
        cypher = """
        MATCH (wr:WorkRole), (t:Task)
        WHERE wr.source = 'NICE' AND t.source = 'NICE'
        WITH wr, t, rand() as random
        WHERE random < 0.025  // About 2.5% to get ~25 tasks per work role
        MERGE (wr)-[:PERFORMS]->(t)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        # Create REQUIRES_KNOWLEDGE relationships between WorkRoles and Knowledge
        # Hub-and-spoke: Each WorkRole gets a subset of Knowledge (about 15-20 knowledge per work role)
        cypher = """
        MATCH (wr:WorkRole), (k:Knowledge)
        WHERE wr.source = 'NICE' AND k.source = 'NICE'
        WITH wr, k, rand() as random
        WHERE random < 0.03  // About 3% to get ~20 knowledge per work role
        MERGE (wr)-[:REQUIRES_KNOWLEDGE]->(k)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        # Create REQUIRES_SKILL relationships between WorkRoles and Skills
        # Hub-and-spoke: Each WorkRole gets a subset of Skills (about 12-15 skills per work role)
        cypher = """
        MATCH (wr:WorkRole), (s:Skill)
        WHERE wr.source = 'NICE' AND s.source = 'NICE'
        WITH wr, s, rand() as random
        WHERE random < 0.03  // About 3% to get ~15 skills per work role
        MERGE (wr)-[:REQUIRES_SKILL]->(s)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        return relationships_created

    # --- DCWF (work roles, tasks, KSAs, relationships) ---

    def _ingest_dcwf(self, file_path: Path) -> IngestionResult:
        """Ingest DCWF dataset with comprehensive multi-file handling"""
        try:
            # Step 1: Create constraints and indexes for DCWF
            self._create_dcwf_constraints()
            self._create_dcwf_indexes()

            # Step 2: Ingest all DCWF elements
            nodes_created = 0

            # Ingest Elements (functional areas)

            # Ingest Work Roles from individual role files
            workrole_result = self._ingest_dcwf_workroles(file_path)
            nodes_created += workrole_result.get("nodes_created", 0)

            # Ingest Tasks from individual role files
            task_result = self._ingest_dcwf_tasks(file_path)
            nodes_created += task_result.get("nodes_created", 0)

            # Ingest KSAs from individual role files
            ksa_result = self._ingest_dcwf_ksas(file_path)
            nodes_created += ksa_result.get("nodes_created", 0)

            # Ingest additional Tasks and KSAs from master list
            master_result = self._ingest_dcwf_master_list()
            nodes_created += master_result.get("nodes_created", 0)

            # Step 3: Create relationships
            relationships_created = self._create_dcwf_relationships()

            return IngestionResult(
                success=True,
                nodes_created=nodes_created,
                relationships_created=relationships_created,
            )

        except Exception as e:
            return IngestionResult(success=False, error=str(e))

    def _create_dcwf_constraints(self):
        """Create DCWF-specific constraints"""
        constraints = [
            "CREATE CONSTRAINT dcwf_workrole_dcwf_code IF NOT EXISTS FOR (n:WorkRole) REQUIRE n.dcwf_code IS UNIQUE;",
            "CREATE CONSTRAINT dcwf_task_dcwf_number IF NOT EXISTS FOR (n:Task) REQUIRE n.dcwf_number IS UNIQUE;",
            "CREATE CONSTRAINT dcwf_ksa_dcwf_number IF NOT EXISTS FOR (n:KSA) REQUIRE n.dcwf_number IS UNIQUE;",
            "CREATE CONSTRAINT dcwf_element_element_code IF NOT EXISTS FOR (n:Element) REQUIRE n.element_code IS UNIQUE;",
            "CREATE CONSTRAINT dcwf_dcwfcode_dcwf_code IF NOT EXISTS FOR (n:DCWFCode) REQUIRE n.dcwf_code IS UNIQUE;",
        ]

        for constraint in constraints:
            try:
                self.db.execute_cypher(constraint)
            except Exception as e:
                logger.warning(f"Constraint creation failed (may already exist): {e}")

    def _create_dcwf_indexes(self):
        """Create DCWF-specific indexes"""
        indexes = [
            "CREATE INDEX dcwf_workrole_element IF NOT EXISTS FOR (n:WorkRole) ON (n.element);",
            "CREATE INDEX dcwf_task_core_additional IF NOT EXISTS FOR (n:Task) ON (n.core_additional);",
            "CREATE INDEX dcwf_ksa_core_additional IF NOT EXISTS FOR (n:KSA) ON (n.core_additional);",
            "CREATE INDEX dcwf_ksa_nice_mapping IF NOT EXISTS FOR (n:KSA) ON (n.nice_mapping);",
        ]

        for index in indexes:
            try:
                self.db.execute_cypher(index)
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")

    def _ingest_dcwf_workroles(self, file_path: Path) -> Dict[str, Any]:
        """Ingest DCWF Work Roles from dcwf_roles.jsonl using APOC"""
        cypher = """
        CALL apoc.load.json($file_path) YIELD value
        WHERE value.work_role IS NOT NULL AND value.work_role <> ''
        MERGE (wr:WorkRole {work_role: value.work_role})
        SET wr.uid = value.dcwf_code,
            wr.definition = value.definition,
            wr.dcwf_code = value.dcwf_code,
            wr.element = value.element,
            wr.ncwf_id = value.ncwf_id,
            wr.source = 'DCWF',
            wr.ingested_at = datetime()
        RETURN count(wr) as nodes_created
        """

        result = self.db.execute_cypher_single(
            cypher, {"file_path": "/import/dcwf/dcwf_roles.jsonl"}
        )
        return result or {"nodes_created": 0}

    def _ingest_dcwf_tasks(self, file_path: Path) -> Dict[str, Any]:
        """Ingest DCWF Tasks from individual role files"""
        import subprocess

        try:
            # List files in the DCWF directory
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "claire-kg-graphrag",
                    "find",
                    "/import/dcwf",
                    "-name",
                    "*.jsonl",
                    "-type",
                    "f",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            files = result.stdout.strip().split("\n")
            nodes_created = 0

            for file in files:
                if file.strip():
                    cypher = """
                    CALL apoc.load.json($file_path) YIELD value
                    WITH value,
                         value.header_info AS header_info,
                         value.data AS data
                    UNWIND data AS task_data
                    WITH task_data
                    WHERE task_data.task_ksa_type = 'Task'
                    MERGE (t:Task {dcwf_number: task_data.dcwf_number})
                    SET t.description = task_data.description,
                        t.core_additional = task_data.core_additional,
                        t.source = 'DCWF',
                        t.ingested_at = datetime()
                    RETURN count(t) as nodes_created
                    """

                    result = self.db.execute_cypher_single(cypher, {"file_path": file})
                    nodes_created += result.get("nodes_created", 0) if result else 0

            return {"nodes_created": nodes_created}

        except Exception as e:
            logger.error(f"Error ingesting DCWF tasks: {e}")
            return {"nodes_created": 0}

    def _ingest_dcwf_ksas(self, file_path: Path) -> Dict[str, Any]:
        """Ingest DCWF KSAs from individual role files"""
        import subprocess

        try:
            # List files in the DCWF directory
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "claire-kg-graphrag",
                    "find",
                    "/import/dcwf",
                    "-name",
                    "*.jsonl",
                    "-type",
                    "f",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            files = result.stdout.strip().split("\n")
            nodes_created = 0

            for file in files:
                if file.strip():
                    cypher = """
                    CALL apoc.load.json($file_path) YIELD value
                    WITH value,
                         value.header_info AS header_info,
                         value.data AS data
                    UNWIND data AS ksa_data
                    WITH ksa_data
                    WHERE ksa_data.task_ksa_type = 'KSA'
                    // Create separate node types based on KSA type
                    WITH ksa_data,
                         CASE 
                            WHEN toLower(ksa_data.description) STARTS WITH 'knowledge of' OR toLower(ksa_data.description) STARTS WITH '* knowledge of' OR toLower(ksa_data.description) STARTS WITH 'knowledge in' OR toLower(ksa_data.description) STARTS WITH 'knowledge and' THEN 'Knowledge'
                            WHEN toLower(ksa_data.description) STARTS WITH 'skill in' OR toLower(ksa_data.description) STARTS WITH 'skill to' OR toLower(ksa_data.description) STARTS WITH 'skill of' THEN 'Skill'
                            WHEN toLower(ksa_data.description) STARTS WITH 'ability to' OR toLower(ksa_data.description) STARTS WITH 'abiltiy to' THEN 'Ability'
                            ELSE 'Other'
                         END as ksa_type
                    
                    // Create Knowledge nodes
                    FOREACH (x IN CASE WHEN ksa_type = 'Knowledge' THEN [1] ELSE [] END |
                        MERGE (k:Knowledge {dcwf_number: ksa_data.dcwf_number})
                        SET k.text = ksa_data.description,
                            k.core_additional = ksa_data.core_additional,
                            k.nice_mapping = ksa_data.nice_mapping,
                            k.source = 'DCWF',
                            k.ingested_at = datetime()
                    )
                    
                    // Create Skill nodes
                    FOREACH (x IN CASE WHEN ksa_type = 'Skill' THEN [1] ELSE [] END |
                        MERGE (s:Skill {dcwf_number: ksa_data.dcwf_number})
                        SET s.text = ksa_data.description,
                            s.core_additional = ksa_data.core_additional,
                            s.nice_mapping = ksa_data.nice_mapping,
                            s.source = 'DCWF',
                            s.ingested_at = datetime()
                    )
                    
                    // Create Ability nodes
                    FOREACH (x IN CASE WHEN ksa_type = 'Ability' THEN [1] ELSE [] END |
                        MERGE (a:Ability {dcwf_number: ksa_data.dcwf_number})
                        SET a.text = ksa_data.description,
                            a.core_additional = ksa_data.core_additional,
                            a.nice_mapping = ksa_data.nice_mapping,
                            a.source = 'DCWF',
                            a.ingested_at = datetime()
                    )
                    RETURN count(*) as nodes_created
                    """

                    result = self.db.execute_cypher_single(cypher, {"file_path": file})
                    nodes_created += result.get("nodes_created", 0) if result else 0

            return {"nodes_created": nodes_created}

        except Exception as e:
            logger.error(f"Error ingesting DCWF KSAs: {e}")
            return {"nodes_created": 0}

    def _ingest_dcwf_master_list(self) -> Dict[str, Any]:
        """Ingest DCWF Tasks and KSAs from master_task_&_ksa_list.jsonl"""
        cypher = """
        CALL apoc.load.json("file:///import/dcwf/master_task_&_ksa_list.jsonl") YIELD value
        UNWIND value.data AS item
        WITH item
        WHERE item.task_ksa_type = 'Task'
        MERGE (t:Task {dcwf_number: item.dcwf_number})
        SET t.description = item.description,
            t.task_ksa_type = item.task_ksa_type,
            t.nice_mapping = item.nice_mapping,
            t.source = 'DCWF',
            t.ingested_at = datetime()
        RETURN count(t) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher)
        task_count = result.get("nodes_created", 0) if result else 0

        # Now ingest KSAs from the same file
        cypher = """
        CALL apoc.load.json("file:///import/dcwf/master_task_&_ksa_list.jsonl") YIELD value
        UNWIND value.data AS item
        WITH item
        WHERE item.task_ksa_type = 'KSA'
        // Create separate node types based on KSA type
        WITH item,
             CASE 
                WHEN toLower(item.description) STARTS WITH 'knowledge of' OR toLower(item.description) STARTS WITH '* knowledge of' OR toLower(item.description) STARTS WITH 'knowledge in' OR toLower(item.description) STARTS WITH 'knowledge and' THEN 'Knowledge'
                WHEN toLower(item.description) STARTS WITH 'skill in' OR toLower(item.description) STARTS WITH 'skill to' OR toLower(item.description) STARTS WITH 'skill of' THEN 'Skill'
                WHEN toLower(item.description) STARTS WITH 'ability to' OR toLower(item.description) STARTS WITH 'abiltiy to' THEN 'Ability'
                ELSE 'Other'
             END as ksa_type
        
        // Create Knowledge nodes
        FOREACH (x IN CASE WHEN ksa_type = 'Knowledge' THEN [1] ELSE [] END |
            MERGE (k:Knowledge {dcwf_number: item.dcwf_number})
            SET k.uid = item.dcwf_number,
                k.text = item.description,
                k.core_additional = item.core_additional,
                k.nice_mapping = item.nice_mapping,
                k.source = 'DCWF',
                k.ingested_at = datetime()
        )
        
        // Create Skill nodes
        FOREACH (x IN CASE WHEN ksa_type = 'Skill' THEN [1] ELSE [] END |
            MERGE (s:Skill {dcwf_number: item.dcwf_number})
            SET s.uid = item.dcwf_number,
                s.text = item.description,
                s.core_additional = item.core_additional,
                s.nice_mapping = item.nice_mapping,
                s.source = 'DCWF',
                s.ingested_at = datetime()
        )
        
        // Create Ability nodes
        FOREACH (x IN CASE WHEN ksa_type = 'Ability' THEN [1] ELSE [] END |
            MERGE (a:Ability {dcwf_number: item.dcwf_number})
            SET a.uid = item.dcwf_number,
                a.text = item.description,
                a.core_additional = item.core_additional,
                a.nice_mapping = item.nice_mapping,
                a.source = 'DCWF',
                a.ingested_at = datetime()
        )
        RETURN count(*) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher)
        ksa_count = result.get("nodes_created", 0) if result else 0

        return {"nodes_created": task_count + ksa_count}

    def _create_dcwf_relationships(self) -> int:
        """Create DCWF hierarchical relationships"""
        relationships_created = 0

        # Create SpecialtyArea nodes directly (no need for Element nodes)
        # Normalize property names: set specialty_prefix = element_code for consistency with NICE
        cypher = """
        MATCH (wr:WorkRole)
        WHERE wr.source = 'DCWF' AND wr.element IS NOT NULL
        WITH DISTINCT wr.element as element_name
        MERGE (sa:SpecialtyArea {element_name: element_name})
        SET         sa.element_code = CASE 
            WHEN element_name CONTAINS 'IT' THEN 'IT'
            WHEN element_name CONTAINS 'CS' THEN 'CS' 
            WHEN element_name CONTAINS 'EN' THEN 'EN'
            WHEN element_name CONTAINS 'CE' THEN 'CE'
            WHEN element_name CONTAINS 'IN' THEN 'IN'
            WHEN element_name CONTAINS 'DA' THEN 'DA'
            WHEN element_name CONTAINS 'SE' THEN 'SE'
            ELSE 'OTHER'
        END,
        sa.specialty_prefix = sa.element_code,
        sa.source = 'DCWF',
        sa.ingested_at = datetime()
        RETURN count(sa) as nodes_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("nodes_created", 0) if result else 0

        # Create IN_SPECIALTY_AREA relationships between WorkRoles and SpecialtyAreas
        cypher = """
        MATCH (wr:WorkRole), (sa:SpecialtyArea)
        WHERE wr.source = 'DCWF' AND sa.source = 'DCWF'
        AND (
            wr.element CONTAINS sa.element_code OR
            wr.element CONTAINS sa.element_name OR
            (wr.element CONTAINS 'IT' AND sa.element_code = 'IT') OR
            (wr.element CONTAINS 'CS' AND sa.element_code = 'CS') OR
            (wr.element CONTAINS 'EN' AND sa.element_code = 'EN') OR
            (wr.element CONTAINS 'CE' AND sa.element_code = 'CE') OR
            (wr.element CONTAINS 'IN' AND sa.element_code = 'IN') OR
            (wr.element CONTAINS 'DA' AND sa.element_code = 'DA') OR
            (wr.element CONTAINS 'SE' AND sa.element_code = 'SE')
        )
        MERGE (wr)-[:IN_SPECIALTY_AREA]->(sa)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        # Create PERFORMS relationships between WorkRoles and Tasks
        # For now, we'll create a simplified relationship based on element matching
        cypher = """
        MATCH (wr:WorkRole), (t:Task)
        WHERE wr.source = 'DCWF' AND t.source = 'DCWF'
        // Create relationships based on shared DCWF context
        // This is a simplified approach - in production you'd track actual task assignments
        WITH wr, t
        WHERE rand() < 0.05  // Sample 5% of possible relationships for demonstration
        MERGE (wr)-[:PERFORMS]->(t)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        # Create REQUIRES_KNOWLEDGE relationships between WorkRoles and Knowledge
        cypher = """
        MATCH (wr:WorkRole), (k:Knowledge)
        WHERE wr.source = 'DCWF' AND k.source = 'DCWF'
        // Create relationships based on shared DCWF context
        // This is a simplified approach - in production you'd track actual knowledge requirements
        WITH wr, k
        WHERE rand() < 0.05  // Sample 5% of possible relationships for demonstration
        MERGE (wr)-[:REQUIRES_KNOWLEDGE]->(k)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        # Create REQUIRES_SKILL relationships between WorkRoles and Skills
        cypher = """
        MATCH (wr:WorkRole), (s:Skill)
        WHERE wr.source = 'DCWF' AND s.source = 'DCWF'
        // Create relationships based on shared DCWF context
        // This is a simplified approach - in production you'd track actual skill requirements
        WITH wr, s
        WHERE rand() < 0.05  // Sample 5% of possible relationships for demonstration
        MERGE (wr)-[:REQUIRES_SKILL]->(s)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        # Create REQUIRES_ABILITY relationships between WorkRoles and Abilities
        cypher = """
        MATCH (wr:WorkRole), (a:Ability)
        WHERE wr.source = 'DCWF' AND a.source = 'DCWF'
        // Create relationships based on shared DCWF context
        // This is a simplified approach - in production you'd track actual ability requirements
        WITH wr, a
        WHERE rand() < 0.05  // Sample 5% of possible relationships for demonstration
        MERGE (wr)-[:REQUIRES_ABILITY]->(a)
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created += result.get("relationships_created", 0) if result else 0

        return relationships_created

    # -------------------------------------------------------------------------
    # Crosswalk creation: dispatch and per-type builders
    # -------------------------------------------------------------------------

    def create_crosswalk(self, crosswalk_type: str) -> CrosswalkResult:
        """Create crosswalk relationships between datasets. Dispatches by crosswalk_type (e.g. capec-attack, cve-cwe, dcwf-nice)."""
        try:
            if crosswalk_type == "capec-attack":
                result = self._create_capec_attack_crosswalk()
            elif crosswalk_type == "capec-relationships":
                result = self._create_capec_relationships()
            elif crosswalk_type == "capec-mitigations":
                result = self._create_capec_mitigations()
            elif crosswalk_type == "attack-mitigations":
                result = self._create_attack_mitigations()
            elif crosswalk_type == "workrole-attack":
                result = self._create_workrole_attack_crosswalk()
            elif crosswalk_type == "cve-attack":
                result = self._create_cve_attack_crosswalk()
            elif crosswalk_type == "cwe-mitigations":
                result = self._create_cwe_mitigations()
            elif crosswalk_type == "cve-cwe":
                result = self._create_cve_cwe_crosswalk()
            elif crosswalk_type == "cve-assets":
                result = self._create_cpe_mapping()
            elif crosswalk_type == "dcwf-nice":
                result = self._create_dcwf_nice_crosswalk()
            elif crosswalk_type == "dcwf-cross-domain":
                result = self._create_dcwf_cross_domain()
            elif crosswalk_type == "nice-cross-domain":
                result = self._create_nice_cross_domain()
            elif crosswalk_type == "workrole-capec":
                result = self._create_workrole_capec_crosswalk()
            elif crosswalk_type == "cwe-categories":
                result = self._create_cwe_categories_crosswalk()
            elif crosswalk_type == "cve-attack":
                result = self._create_cve_attack_crosswalk()
            elif crosswalk_type == "cve-capec":
                result = self._create_cve_capec_crosswalk()
            else:
                return CrosswalkResult(
                    success=False, error=f"Unknown crosswalk: {crosswalk_type}"
                )

            # Validate no duplicates were created
            if result.success:
                self._validate_crosswalk_integrity(crosswalk_type)

            return result

        except Exception as e:
            logger.error(f"Error creating {crosswalk_type} crosswalk: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _validate_crosswalk_integrity(self, crosswalk_type: str):
        """Validate that no duplicate relationships were created"""
        try:
            if crosswalk_type == "capec-attack":
                validation = self.db.execute_cypher_single(
                    """
                MATCH (ap:AttackPattern)-[r:RELATES_TO]->(t:Technique)
                WHERE r.mapping_type = 'CAPEC_TO_ATTACK'
                RETURN count(r) as total, count(DISTINCT ap.uid + '-' + t.uid) as unique
                """
                )
            elif crosswalk_type == "cve-cwe":
                validation = self.db.execute_cypher_single(
                    """
                MATCH (v:Vulnerability)-[r:HAS_WEAKNESS]->(w:Weakness)
                WHERE r.mapping_type = 'CVE_TO_CWE'
                RETURN count(r) as total, count(DISTINCT v.uid + '-' + w.uid) as unique
                """
                )
            elif crosswalk_type == "cve-assets":
                validation = self.db.execute_cypher_single(
                    """
                    MATCH (v:Vulnerability)-[r:AFFECTS]->(asset:Asset)
                    WHERE r.mapping_type = 'CVE_TO_ASSET'
                    RETURN count(r) as total, count(DISTINCT v.uid) as unique
                    """
                )
            elif crosswalk_type == "dcwf-nice":
                validation = self.db.execute_cypher_single(
                    """
                MATCH (dcwf:WorkRole)-[r:RELATES_TO]->(nice:WorkRole)
                WHERE r.mapping_type = 'DCWF_TO_NICE'
                RETURN count(r) as total, count(DISTINCT dcwf.uid + '-' + nice.uid) as unique
                """
                )
            elif crosswalk_type == "capec-relationships":
                # Check both RELATES_TO and EXPLOITS relationships
                rel_validation = self.db.execute_cypher_single(
                    """
                MATCH (ap1:AttackPattern)-[r:RELATES_TO]->(ap2:AttackPattern)
                WHERE r.source = 'CAPEC_Related_Patterns'
                RETURN count(r) as total, count(DISTINCT ap1.uid + '-' + ap2.uid) as unique
                """
                )
                exp_validation = self.db.execute_cypher_single(
                    """
                MATCH (ap:AttackPattern)-[r:EXPLOITS]->(w:Weakness)
                WHERE r.source = 'CAPEC_Related_Weaknesses'
                RETURN count(r) as total, count(DISTINCT ap.uid + '-' + w.uid) as unique
                """
                )
                validation = {
                    "total": (rel_validation["total"] if rel_validation else 0)
                    + (exp_validation["total"] if exp_validation else 0),
                    "unique": (rel_validation["unique"] if rel_validation else 0)
                    + (exp_validation["unique"] if exp_validation else 0),
                }
            elif crosswalk_type == "cwe-categories":
                validation = self.db.execute_cypher_single(
                    """
                MATCH (v:Vulnerability)-[r:IS_CWE_TYPE]->(cc:CWECategory)
                WHERE r.mapping_type = 'CVE_TO_CWE_CATEGORY'
                RETURN count(r) as total, count(DISTINCT v.uid + '-' + cc.cwe_uid) as unique
                """
                )
            elif crosswalk_type == "cve-attack":
                validation = self.db.execute_cypher_single(
                    """
                MATCH (v:Vulnerability)-[r:CAN_BE_EXPLOITED_BY]->(t:Technique)
                WHERE r.mapping_type = 'CVE_TO_ATTACK'
                RETURN count(r) as total, count(DISTINCT v.uid + '-' + t.uid) as unique
                """
                )
            elif crosswalk_type == "cve-capec":
                validation = self.db.execute_cypher_single(
                    """
                MATCH (v:Vulnerability)-[r:CAN_BE_EXPLOITED_BY]->(ap:AttackPattern)
                WHERE r.mapping_type = 'CVE_TO_CAPEC'
                RETURN count(r) as total, count(DISTINCT v.uid + '-' + ap.uid) as unique
                """
                )
            else:
                return  # Unknown crosswalk type, skip validation

            if validation and validation["total"] != validation["unique"]:
                duplicates = validation["total"] - validation["unique"]
                logger.warning(
                    f" DUPLICATES DETECTED in {crosswalk_type}: {validation['total']} total, {validation['unique']} unique, {duplicates} duplicates"
                )
            else:
                logger.info(
                    f"OK: {crosswalk_type} crosswalk integrity validated: {validation['total']} relationships, no duplicates"
                )

        except Exception as e:
            logger.error(f"Error validating {crosswalk_type} crosswalk integrity: {e}")

    def _create_capec_attack_crosswalk(self) -> CrosswalkResult:
        """Create CAPEC to ATT&CK crosswalk"""
        cypher = """
        MATCH (ap:AttackPattern)
        WHERE ap.taxonomy_mappings IS NOT NULL 
        AND ap.taxonomy_mappings CONTAINS 'ATTACK:ENTRY ID:'
        WITH ap, ap.taxonomy_mappings as mappings
        UNWIND split(mappings, 'TAXONOMY NAME:') as mapping
        WITH ap, trim(mapping) as clean_mapping
        WHERE clean_mapping STARTS WITH 'ATTACK:ENTRY ID:'
        WITH ap, clean_mapping,
             split(clean_mapping, ':')[2] as technique_id_raw
        WITH ap, technique_id_raw,
             CASE 
               WHEN technique_id_raw CONTAINS '.' THEN split(technique_id_raw, '.')[0]
               ELSE technique_id_raw
             END as technique_id
        WITH ap, 'T' + technique_id as technique_id
        WHERE technique_id STARTS WITH 'T'
        MATCH (t:Technique {uid: technique_id})
        MERGE (ap)-[r:RELATES_TO {
            mapping_type: 'CAPEC_TO_ATTACK',
            confidence: 'high',
            source: 'CAPEC_Taxonomy_Mappings'
        }]->(t)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created = result["relationships_created"] if result else 0

        return CrosswalkResult(
            success=True, relationships_created=relationships_created
        )

    def _create_cve_cwe_crosswalk(self) -> CrosswalkResult:
        """Create CVE to CWE crosswalk - FIXED for all weakness types"""
        # First, handle CVEs with actual CWE IDs
        cwe_crosswalk_cypher = """
        MATCH (v:Vulnerability)
        WHERE v.weaknesses IS NOT NULL
        WITH v, v.weaknesses as weaknessStr
        WHERE weaknessStr =~ '.*CWE-[0-9]+.*'
        WITH v, weaknessStr
        MATCH (w:Weakness)
        WHERE weaknessStr CONTAINS w.uid
        MERGE (v)-[r:HAS_WEAKNESS {
            mapping_type: 'CVE_TO_CWE',
            confidence: 'high',
            source: 'CVE_Weaknesses_Field',
            weakness_type: 'Primary'
        }]->(w)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cwe_crosswalk_cypher)
        cwe_relationships = result["relationships_created"] if result else 0

        # Second, handle CVEs with NVD-CWE-noinfo and NVD-CWE-Other
        noinfo_crosswalk_cypher = """
        MATCH (v:Vulnerability)
        WHERE v.weaknesses IS NOT NULL AND v.weaknesses =~ '.*NVD-CWE-noinfo.*'
        MERGE (w:Weakness {uid: 'NVD-CWE-noinfo'})
        SET w.name = 'No CWE Information Available',
            w.description = 'CVE has no CWE information available',
            w.status = 'Unknown',
            w.source = 'NVD_NOINFO',
            w.ingested_at = datetime()
        MERGE (v)-[r:HAS_WEAKNESS {
            mapping_type: 'CVE_TO_NOINFO',
            confidence: 'medium',
            source: 'NVD_NoInfo',
            weakness_type: 'NoInfo'
        }]->(w)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(noinfo_crosswalk_cypher)
        noinfo_relationships = result["relationships_created"] if result else 0

        # Third, handle CVEs with NVD-CWE-Other
        other_crosswalk_cypher = """
        MATCH (v:Vulnerability)
        WHERE v.weaknesses IS NOT NULL AND v.weaknesses =~ '.*NVD-CWE-Other.*'
        MERGE (w:Weakness {uid: 'NVD-CWE-Other'})
        SET w.name = 'Other CWE Type',
            w.description = 'CVE has other/unknown CWE type',
            w.status = 'Unknown',
            w.source = 'NVD_OTHER',
            w.ingested_at = datetime()
        MERGE (v)-[r:HAS_WEAKNESS {
            mapping_type: 'CVE_TO_OTHER',
            confidence: 'medium',
            source: 'NVD_Other',
            weakness_type: 'Other'
        }]->(w)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(other_crosswalk_cypher)
        other_relationships = result["relationships_created"] if result else 0

        total_relationships = (
            cwe_relationships + noinfo_relationships + other_relationships
        )

        return CrosswalkResult(success=True, relationships_created=total_relationships)

    def _create_cpe_mapping(self) -> CrosswalkResult:
        """Create CPE mapping for CVEs using APOC periodic iterate for performance"""
        try:
            print(" Analyzing CVE configurations for CPE strings...")

            # First, check how many CVEs have CPE configurations
            check_cypher = """
            MATCH (v:Vulnerability)
            WHERE v.configurations IS NOT NULL
            RETURN count(*) as cve_count
            """

            result = self.db.execute_cypher_single(check_cypher)
            cve_count = result["cve_count"] if result else 0
            print(f" Found {cve_count:,} CVEs with configurations")

            if cve_count == 0:
                print("WARNING:  No CVEs with configurations found")
                return CrosswalkResult(success=True, relationships_created=0)

            print("🔄 Processing CPE configurations with batched processing...")

            # Use APOC periodic iterate for efficient batch processing
            cypher = """
            CALL apoc.periodic.iterate(
                'MATCH (v:Vulnerability) WHERE v.configurations IS NOT NULL AND v.configurations <> "null" RETURN v',
                '
                WITH v, apoc.convert.fromJsonList(v.configurations) as configs
                UNWIND configs as config
                UNWIND config.nodes as node
                UNWIND node.cpeMatch as cpe_match
                WITH v, cpe_match.criteria as cpe_string
                WHERE cpe_string STARTS WITH "cpe:2.3:"
                WITH v, cpe_string, split(cpe_string, ":")[3] as vendor, split(cpe_string, ":")[4] as product
                MERGE (asset:Asset {uid: "cpe:" + vendor + ":" + product})
                ON CREATE SET
                    asset.name = vendor + " " + product,
                    asset.vendor = vendor,
                    asset.product = product,
                    asset.cpe_type = split(cpe_string, ":")[2],
                    asset.source = "CPE_Mapping",
                    asset.ingested_at = datetime()
                MERGE (v)-[:AFFECTS {
                    mapping_type: "CVE_TO_ASSET",
                    confidence: "high",
                    source: "CVE_CPE_Configuration",
                    cpe_string: cpe_string,
                    affected_version: split(cpe_string, ":")[5],
                    vendor: vendor,
                    product: product,
                    ingested_at: datetime()
                }]->(asset)
                ',
                {batchSize: 1000, parallel: false}
            ) YIELD batches, total, timeTaken, committedOperations, failedOperations, errorMessages
            RETURN batches, total, timeTaken, committedOperations, failedOperations, errorMessages
            """

            result = self.db.execute_cypher_single(cypher)

            if result:
                print(
                    f"OK: Processed {result['total']:,} CVEs in {result['timeTaken']:,}ms"
                )
                print(f"OK: Created {result['committedOperations']:,} relationships")
                if result["failedOperations"] > 0:
                    print(f"WARNING:  {result['failedOperations']:,} operations failed")
                    if result["errorMessages"]:
                        print(f"ERROR: Errors: {result['errorMessages']}")

                return CrosswalkResult(
                    success=True, relationships_created=result["committedOperations"]
                )
            else:
                print("ERROR: No result from periodic iterate")
                return CrosswalkResult(
                    success=False, error="No result from periodic iterate"
                )

        except Exception as e:
            print(f"ERROR: Error in CPE mapping: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_dcwf_nice_crosswalk(self) -> CrosswalkResult:
        """Create DCWF to NICE crosswalk"""
        cypher = """
        MATCH (dcwf_sa:SpecialtyArea {source: 'DCWF'})
        WITH dcwf_sa, dcwf_sa.element_code as dcwf_code
        WITH dcwf_sa, dcwf_code,
             CASE dcwf_code
               WHEN 'CS' THEN 'PD'
               WHEN 'IT' THEN 'IO'
               WHEN 'SE' THEN 'DD'
               WHEN 'DA' THEN 'DD'
               WHEN 'IN' THEN 'IN'
               WHEN 'EN' THEN 'DD'
               WHEN 'CE' THEN 'PD'
               WHEN 'OPR' THEN 'OG'
               ELSE 'UNKNOWN'
             END as nice_prefix
        WHERE nice_prefix <> 'UNKNOWN'
        MATCH (nice_sa:SpecialtyArea {source: 'NICE'})
        WHERE nice_sa.specialty_prefix = nice_prefix
        MERGE (dcwf_sa)-[r:RELATES_TO {
            mapping_type: 'DCWF_TO_NICE',
            confidence: 'high',
            source: 'Domain_Code_Mapping',
            dcwf_domain: dcwf_code,
            nice_category: nice_prefix
        }]->(nice_sa)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(*) as relationships_created
        """

        result = self.db.execute_cypher_single(cypher)
        relationships_created = result["relationships_created"] if result else 0

        return CrosswalkResult(
            success=True, relationships_created=relationships_created
        )

    def _create_dcwf_cross_domain(self) -> CrosswalkResult:
        """Create DCWF cross-domain relationships to CVE, ATT&CK, CAPEC, CWE

        DISABLED: Synthetic WORKS_WITH relationships have been removed.
        This method no longer creates heuristic-based cross-dataset relationships
        between WorkForce (DCWF) and Threat Intel (CVE, ATT&CK, CAPEC, CWE) datasets.

        These relationships were created using keyword matching heuristics (e.g.,
        matching "Cyber" or "Security" in role names with high-severity CVEs),
        which were not based on actual framework mappings.

        Returns 0 relationships to indicate the method is disabled.
        """
        # Synthetic relationships have been removed - return early
        return CrosswalkResult(success=True, relationships_created=0)

        # DISABLED CODE BELOW - kept for reference only
        # total_relationships = 0

        # 1. DCWF WorkRoles to CVEs (addressing vulnerabilities)
        cypher_cve = """
        MATCH (wr:WorkRole {source: 'DCWF'})
        WHERE wr.work_role CONTAINS 'Cyber' OR wr.work_role CONTAINS 'Security' OR wr.work_role CONTAINS 'IT'
        MATCH (v:Vulnerability)
        WHERE v.severity IN ['HIGH', 'CRITICAL'] OR v.cvss_v31 >= 7.0
        WITH wr, v,
             CASE
               WHEN wr.work_role CONTAINS 'Cyber' AND v.severity = 'CRITICAL' THEN 0.9
               WHEN wr.work_role CONTAINS 'Security' AND v.severity = 'HIGH' THEN 0.8
               WHEN wr.work_role CONTAINS 'IT' AND v.cvss_v31 >= 7.0 THEN 0.7
               ELSE 0.6
             END as confidence
        WHERE confidence >= 0.6
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'DCWF_TO_CVE',
            confidence: CASE WHEN confidence >= 0.8 THEN 'high' ELSE 'medium' END,
            source: 'Domain_Code_Mapping',
            dcwf_domain: wr.work_role,
            vulnerability_severity: v.severity
        }]->(v)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_cve = self.db.execute_cypher_single(cypher_cve)
        cve_rels = result_cve["relationships_created"] if result_cve else 0
        total_relationships += cve_rels

        # 2. DCWF WorkRoles to ATT&CK Techniques (defending against)
        cypher_attack = """
        MATCH (wr:WorkRole {source: 'DCWF'})
        WHERE wr.element CONTAINS 'Cyber' OR wr.element CONTAINS 'Security' OR wr.element CONTAINS 'Intelligence'
        MATCH (t:Technique)
        WHERE t.tactics CONTAINS 'Defense' OR t.tactics CONTAINS 'Persistence' OR t.tactics CONTAINS 'Discovery'
        WITH wr, t,
             CASE
               WHEN wr.element CONTAINS 'Cyber' AND t.tactics CONTAINS 'Defense' THEN 0.9
               WHEN wr.element CONTAINS 'Security' AND t.tactics CONTAINS 'Persistence' THEN 0.8
               WHEN wr.element CONTAINS 'Intelligence' AND t.tactics CONTAINS 'Discovery' THEN 0.8
               ELSE 0.6
             END as confidence
        WHERE confidence >= 0.6
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'DCWF_TO_ATTACK',
            confidence: CASE WHEN confidence >= 0.8 THEN 'high' ELSE 'medium' END,
            source: 'Domain_Code_Mapping',
            dcwf_domain: wr.element,
            technique_tactics: t.tactics
        }]->(t)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_attack = self.db.execute_cypher_single(cypher_attack)
        attack_rels = result_attack["relationships_created"] if result_attack else 0
        total_relationships += attack_rels

        # 3. DCWF WorkRoles to CAPEC Attack Patterns (mitigating attacks)
        cypher_capec = """
        MATCH (wr:WorkRole {source: 'DCWF'})
        WHERE wr.element CONTAINS 'Cyber' OR wr.element CONTAINS 'Security'
        MATCH (ap:AttackPattern)
        WHERE ap.abstraction_level IN ['Standard', 'Detailed'] AND ap.status = 'Stable'
        WITH wr, ap,
             CASE
               WHEN wr.element CONTAINS 'Cyber' AND ap.abstraction_level = 'Standard' THEN 0.9
               WHEN wr.element CONTAINS 'Security' AND ap.abstraction_level = 'Detailed' THEN 0.8
               ELSE 0.6
             END as confidence
        WHERE confidence >= 0.6
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'DCWF_TO_CAPEC',
            confidence: CASE WHEN confidence >= 0.8 THEN 'high' ELSE 'medium' END,
            source: 'Domain_Code_Mapping',
            dcwf_domain: wr.element,
            attack_abstraction: ap.abstraction_level
        }]->(ap)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_capec = self.db.execute_cypher_single(cypher_capec)
        capec_rels = result_capec["relationships_created"] if result_capec else 0
        total_relationships += capec_rels

        # 4. DCWF WorkRoles to CWE Weaknesses (addressing weaknesses)
        cypher_cwe = """
        MATCH (wr:WorkRole {source: 'DCWF'})
        WHERE wr.element CONTAINS 'Cyber' OR wr.element CONTAINS 'Security' OR wr.element CONTAINS 'IT'
        MATCH (w:Weakness)
        WHERE w.status = 'Stable' AND w.weakness_ordinality IN ['Primary', 'Class']
        WITH wr, w,
             CASE
               WHEN wr.element CONTAINS 'Cyber' AND w.weakness_ordinality = 'Primary' THEN 0.9
               WHEN wr.element CONTAINS 'Security' AND w.weakness_ordinality = 'Class' THEN 0.8
               WHEN wr.element CONTAINS 'IT' AND w.weakness_ordinality = 'Primary' THEN 0.7
               ELSE 0.6
             END as confidence
        WHERE confidence >= 0.6
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'DCWF_TO_CWE',
            confidence: CASE WHEN confidence >= 0.8 THEN 'high' ELSE 'medium' END,
            source: 'Domain_Code_Mapping',
            dcwf_domain: wr.element,
            weakness_ordinality: w.weakness_ordinality
        }]->(w)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_cwe = self.db.execute_cypher_single(cypher_cwe)
        cwe_rels = result_cwe["relationships_created"] if result_cwe else 0
        total_relationships += cwe_rels

        return CrosswalkResult(success=True, relationships_created=total_relationships)

    def _create_nice_cross_domain(self) -> CrosswalkResult:
        """Create NICE cross-domain relationships to CVE, ATT&CK, CAPEC, CWE

        DISABLED: Synthetic WORKS_WITH relationships have been removed.
        This method no longer creates heuristic-based cross-dataset relationships
        between WorkForce (NICE) and Threat Intel (CVE, ATT&CK, CAPEC, CWE) datasets.

        These relationships were created using keyword matching heuristics (e.g.,
        matching "Cyber" or "Security" in role titles with high-severity CVEs),
        which were not based on actual framework mappings.

        Returns 0 relationships to indicate the method is disabled.
        """
        # Synthetic relationships have been removed - return early
        return CrosswalkResult(success=True, relationships_created=0)

        # DISABLED CODE BELOW - kept for reference only
        # total_relationships = 0

        # 1. NICE WorkRoles to CVEs (addressing vulnerabilities)
        cypher_cve = """
        MATCH (wr:WorkRole {source: 'NICE'})
        WHERE wr.title CONTAINS 'Cyber' OR wr.title CONTAINS 'Security' OR wr.title CONTAINS 'Defense'
        MATCH (v:Vulnerability)
        WHERE v.severity IN ['HIGH', 'CRITICAL'] OR v.cvss_v31 >= 7.0
        WITH wr, v,
             CASE
               WHEN wr.title CONTAINS 'Cyber' AND v.severity = 'CRITICAL' THEN 0.9
               WHEN wr.title CONTAINS 'Security' AND v.severity = 'HIGH' THEN 0.8
               WHEN wr.title CONTAINS 'Defense' AND v.cvss_v31 >= 7.0 THEN 0.8
               ELSE 0.6
             END as confidence
        WHERE confidence >= 0.6
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'NICE_TO_CVE',
            confidence: CASE WHEN confidence >= 0.8 THEN 'high' ELSE 'medium' END,
            source: 'Domain_Code_Mapping',
            nice_specialty: wr.element_identifier,
            vulnerability_severity: v.severity
        }]->(v)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_cve = self.db.execute_cypher_single(cypher_cve)
        cve_rels = result_cve["relationships_created"] if result_cve else 0
        total_relationships += cve_rels

        # 2. NICE WorkRoles to ATT&CK Techniques (defending against)
        cypher_attack = """
        MATCH (wr:WorkRole {source: 'NICE'})
        WHERE wr.title CONTAINS 'Cyber' OR wr.title CONTAINS 'Security' OR wr.title CONTAINS 'Analysis'
        MATCH (t:Technique)
        WHERE t.tactics CONTAINS 'Defense' OR t.tactics CONTAINS 'Persistence' OR t.tactics CONTAINS 'Discovery'
        WITH wr, t,
             CASE
               WHEN wr.title CONTAINS 'Cyber' AND t.tactics CONTAINS 'Defense' THEN 0.9
               WHEN wr.title CONTAINS 'Security' AND t.tactics CONTAINS 'Persistence' THEN 0.8
               WHEN wr.title CONTAINS 'Analysis' AND t.tactics CONTAINS 'Discovery' THEN 0.8
               ELSE 0.6
             END as confidence
        WHERE confidence >= 0.6
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'NICE_TO_ATTACK',
            confidence: CASE WHEN confidence >= 0.8 THEN 'high' ELSE 'medium' END,
            source: 'Domain_Code_Mapping',
            nice_specialty: wr.element_identifier,
            technique_tactics: t.tactics
        }]->(t)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_attack = self.db.execute_cypher_single(cypher_attack)
        attack_rels = result_attack["relationships_created"] if result_attack else 0
        total_relationships += attack_rels

        # 3. NICE WorkRoles to CAPEC Attack Patterns (mitigating attacks)
        cypher_capec = """
        MATCH (wr:WorkRole {source: 'NICE'})
        WHERE wr.title CONTAINS 'Cyber' OR wr.title CONTAINS 'Security' OR wr.title CONTAINS 'Defense'
        MATCH (ap:AttackPattern)
        WHERE ap.abstraction_level IN ['Standard', 'Detailed'] AND ap.status = 'Stable'
        WITH wr, ap,
             CASE
               WHEN wr.title CONTAINS 'Cyber' AND ap.abstraction_level = 'Standard' THEN 0.9
               WHEN wr.title CONTAINS 'Security' AND ap.abstraction_level = 'Detailed' THEN 0.8
               WHEN wr.title CONTAINS 'Defense' AND ap.abstraction_level = 'Standard' THEN 0.8
               ELSE 0.6
             END as confidence
        WHERE confidence >= 0.6
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'NICE_TO_CAPEC',
            confidence: CASE WHEN confidence >= 0.8 THEN 'high' ELSE 'medium' END,
            source: 'Domain_Code_Mapping',
            nice_specialty: wr.element_identifier,
            attack_abstraction: ap.abstraction_level
        }]->(ap)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_capec = self.db.execute_cypher_single(cypher_capec)
        capec_rels = result_capec["relationships_created"] if result_capec else 0
        total_relationships += capec_rels

        # 4. NICE WorkRoles to CWE Weaknesses (addressing weaknesses)
        cypher_cwe = """
        MATCH (wr:WorkRole {source: 'NICE'})
        WHERE wr.title CONTAINS 'Cyber' OR wr.title CONTAINS 'Security' OR wr.title CONTAINS 'Development'
        MATCH (w:Weakness)
        WHERE w.status = 'Stable' AND w.weakness_ordinality IN ['Primary', 'Class']
        WITH wr, w,
             CASE
               WHEN wr.title CONTAINS 'Cyber' AND w.weakness_ordinality = 'Primary' THEN 0.9
               WHEN wr.title CONTAINS 'Security' AND w.weakness_ordinality = 'Class' THEN 0.8
               WHEN wr.title CONTAINS 'Development' AND w.weakness_ordinality = 'Primary' THEN 0.7
               ELSE 0.6
             END as confidence
        WHERE confidence >= 0.6
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'NICE_TO_CWE',
            confidence: CASE WHEN confidence >= 0.8 THEN 'high' ELSE 'medium' END,
            source: 'Domain_Code_Mapping',
            nice_specialty: wr.element_identifier,
            weakness_ordinality: w.weakness_ordinality
        }]->(w)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_cwe = self.db.execute_cypher_single(cypher_cwe)
        cwe_rels = result_cwe["relationships_created"] if result_cwe else 0
        total_relationships += cwe_rels

        return CrosswalkResult(success=True, relationships_created=total_relationships)

    def _create_workrole_capec_crosswalk(self) -> CrosswalkResult:
        """Create WorkRole-CAPEC cross-domain relationships using semantic matching

        DISABLED: Synthetic WORKS_WITH relationships have been removed.
        This method no longer creates heuristic-based cross-dataset relationships
        between WorkRole and CAPEC AttackPattern nodes.

        These relationships were created using keyword matching between WorkRole
        titles and CAPEC attack pattern names, which were not based on actual
        framework mappings.

        Returns 0 relationships to indicate the method is disabled.
        """
        # Synthetic relationships have been removed - return early
        return CrosswalkResult(success=True, relationships_created=0)

        # DISABLED CODE BELOW - kept for reference only
        # Clear existing WorkRole-CAPEC relationships
        # clear_cypher = """
        # MATCH (wr:WorkRole)-[r:WORKS_WITH]->(ap:AttackPattern)
        # DELETE r
        # """

        # Create WorkRole-CAPEC relationships using semantic matching
        cypher = """
        MATCH (wr:WorkRole), (ap:AttackPattern)
        WHERE (
            // Match based on WorkRole title keywords and CAPEC attack patterns
            (toLower(wr.title) CONTAINS 'security' AND toLower(ap.name) CONTAINS 'security') OR
            (toLower(wr.title) CONTAINS 'threat' AND toLower(ap.name) CONTAINS 'threat') OR
            (toLower(wr.title) CONTAINS 'incident' AND toLower(ap.name) CONTAINS 'incident') OR
            (toLower(wr.title) CONTAINS 'vulnerability' AND toLower(ap.name) CONTAINS 'vulnerability') OR
            (toLower(wr.title) CONTAINS 'malware' AND toLower(ap.name) CONTAINS 'malware') OR
            (toLower(wr.title) CONTAINS 'network' AND toLower(ap.name) CONTAINS 'network') OR
            (toLower(wr.title) CONTAINS 'forensic' AND toLower(ap.name) CONTAINS 'forensic') OR
            (toLower(wr.title) CONTAINS 'analyst' AND toLower(ap.name) CONTAINS 'analysis') OR
            (toLower(wr.title) CONTAINS 'defense' AND toLower(ap.name) CONTAINS 'defense') OR
            (toLower(wr.title) CONTAINS 'monitor' AND toLower(ap.name) CONTAINS 'monitor') OR
            (toLower(wr.title) CONTAINS 'detect' AND toLower(ap.name) CONTAINS 'detect') OR
            (toLower(wr.title) CONTAINS 'respond' AND toLower(ap.name) CONTAINS 'respond') OR
            (toLower(wr.title) CONTAINS 'investigate' AND toLower(ap.name) CONTAINS 'investigate') OR
            (toLower(wr.title) CONTAINS 'penetration' AND toLower(ap.name) CONTAINS 'penetration') OR
            (toLower(wr.title) CONTAINS 'red team' AND toLower(ap.name) CONTAINS 'red team') OR
            (toLower(wr.title) CONTAINS 'blue team' AND toLower(ap.name) CONTAINS 'blue team') OR
            (toLower(wr.title) CONTAINS 'soc' AND toLower(ap.name) CONTAINS 'soc') OR
            (toLower(wr.title) CONTAINS 'siem' AND toLower(ap.name) CONTAINS 'siem') OR
            (toLower(wr.title) CONTAINS 'firewall' AND toLower(ap.name) CONTAINS 'firewall') OR
            (toLower(wr.title) CONTAINS 'endpoint' AND toLower(ap.name) CONTAINS 'endpoint') OR
            (toLower(wr.title) CONTAINS 'cloud' AND toLower(ap.name) CONTAINS 'cloud') OR
            (toLower(wr.title) CONTAINS 'identity' AND toLower(ap.name) CONTAINS 'identity') OR
            (toLower(wr.title) CONTAINS 'privilege' AND toLower(ap.name) CONTAINS 'privilege') OR
            (toLower(wr.title) CONTAINS 'credential' AND toLower(ap.name) CONTAINS 'credential') OR
            (toLower(wr.title) CONTAINS 'persistence' AND toLower(ap.name) CONTAINS 'persistence') OR
            (toLower(wr.title) CONTAINS 'lateral' AND toLower(ap.name) CONTAINS 'lateral') OR
            (toLower(wr.title) CONTAINS 'exfiltration' AND toLower(ap.name) CONTAINS 'exfiltration') OR
            (toLower(wr.title) CONTAINS 'command' AND toLower(ap.name) CONTAINS 'command') OR
            (toLower(wr.title) CONTAINS 'execution' AND toLower(ap.name) CONTAINS 'execution') OR
            (toLower(wr.title) CONTAINS 'collection' AND toLower(ap.name) CONTAINS 'collection') OR
            (toLower(wr.title) CONTAINS 'impact' AND toLower(ap.name) CONTAINS 'impact') OR
            (toLower(wr.title) CONTAINS 'initial' AND toLower(ap.name) CONTAINS 'initial') OR
            (toLower(wr.title) CONTAINS 'discovery' AND toLower(ap.name) CONTAINS 'discovery') OR
            (toLower(wr.title) CONTAINS 'defense' AND toLower(ap.name) CONTAINS 'defense') OR
            (toLower(wr.title) CONTAINS 'evasion' AND toLower(ap.name) CONTAINS 'evasion') OR
            (toLower(wr.title) CONTAINS 'credential' AND toLower(ap.name) CONTAINS 'credential') OR
            (toLower(wr.title) CONTAINS 'access' AND toLower(ap.name) CONTAINS 'access') OR
            (toLower(wr.title) CONTAINS 'privilege' AND toLower(ap.name) CONTAINS 'privilege') OR
            (toLower(wr.title) CONTAINS 'persistence' AND toLower(ap.name) CONTAINS 'persistence') OR
            (toLower(wr.title) CONTAINS 'lateral' AND toLower(ap.name) CONTAINS 'lateral') OR
            (toLower(wr.title) CONTAINS 'collection' AND toLower(ap.name) CONTAINS 'collection') OR
            (toLower(wr.title) CONTAINS 'exfiltration' AND toLower(ap.name) CONTAINS 'exfiltration') OR
            (toLower(wr.title) CONTAINS 'impact' AND toLower(ap.name) CONTAINS 'impact')
        )
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'WORKROLE_TO_CAPEC',
            confidence: 'medium',
            source: 'Semantic_Matching',
            relationship_type: 'works_with'
        }]->(ap)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        try:
            # Clear existing relationships
            self.db.execute_cypher_single(clear_cypher)

            # Create new relationships
            result = self.db.execute_cypher_single(cypher)
            relationships_created = (
                result.get("relationships_created", 0) if result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating WorkRole-CAPEC crosswalk: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_capec_relationships(self) -> CrosswalkResult:
        """Create CAPEC internal relationships (related attack patterns and weaknesses)"""
        # Create related attack pattern relationships
        cypher_attack_patterns = """
        MATCH (ap:AttackPattern)
        WHERE ap.related_attack_patterns IS NOT NULL AND ap.related_attack_patterns <> ''
        WITH ap, ap.related_attack_patterns as related
        UNWIND split(related, '::') as pattern_ref
        WITH ap, trim(pattern_ref) as clean_ref
        WHERE clean_ref <> '' AND clean_ref CONTAINS 'CAPEC ID:'
        WITH ap, clean_ref,
             CASE 
               WHEN clean_ref CONTAINS 'Child Of' THEN 'CHILD_OF'
               WHEN clean_ref CONTAINS 'Can Precede' THEN 'CAN_PRECEDE'
               WHEN clean_ref CONTAINS 'Can Also Be' THEN 'CAN_ALSO_BE'
               ELSE 'RELATES_TO'
             END as rel_type
        WITH ap, clean_ref, rel_type,
             CASE 
               WHEN clean_ref CONTAINS 'CAPEC ID:' THEN 
                 'CAPEC-' + split(clean_ref, 'CAPEC ID:')[1]
               ELSE null
             END as target_uid
        WHERE target_uid IS NOT NULL
        MATCH (target:AttackPattern {uid: target_uid})
        MERGE (ap)-[r:RELATES_TO {
            mapping_type: 'CAPEC_TO_CAPEC',
            relationship_type: rel_type,
            source: 'CAPEC_Related_Patterns'
        }]->(target)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_attack_patterns = self.db.execute_cypher_single(cypher_attack_patterns)
        attack_pattern_rels = (
            result_attack_patterns["relationships_created"]
            if result_attack_patterns
            else 0
        )

        # Create related weakness relationships
        cypher_weaknesses = """
        MATCH (ap:AttackPattern)
        WHERE ap.related_weaknesses IS NOT NULL AND ap.related_weaknesses <> ''
        WITH ap, ap.related_weaknesses as related
        UNWIND split(related, '::') as weakness_ref
        WITH ap, trim(weakness_ref) as clean_ref
        WHERE clean_ref <> '' AND clean_ref =~ '^[0-9]+$'
        WITH ap, clean_ref,
             'CWE-' + clean_ref as target_uid
        WHERE target_uid IS NOT NULL
        MATCH (target:Weakness {uid: target_uid})
        MERGE (ap)-[r:EXPLOITS {
            mapping_type: 'CAPEC_TO_CWE',
            source: 'CAPEC_Related_Weaknesses'
        }]->(target)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        result_weaknesses = self.db.execute_cypher_single(cypher_weaknesses)
        weakness_rels = (
            result_weaknesses["relationships_created"] if result_weaknesses else 0
        )

        total_relationships = attack_pattern_rels + weakness_rels

        return CrosswalkResult(success=True, relationships_created=total_relationships)

    def _create_capec_mitigations(self) -> CrosswalkResult:
        """Create CAPEC mitigations from AttackPattern mitigations data"""
        # First, clear existing CAPEC mitigations
        clear_cypher = """
        MATCH (m:Mitigation {source: 'CAPEC'})
        DETACH DELETE m
        """

        # Main mitigations extraction
        cypher = """
        MATCH (ap:AttackPattern)
        WHERE ap.mitigations IS NOT NULL
        UNWIND ap.mitigations as mitigation_string
        WITH ap, mitigation_string
        WHERE mitigation_string <> '' AND mitigation_string <> ':'
        WITH ap, split(mitigation_string, '::') as mitigation_parts
        UNWIND mitigation_parts as mitigation_text
        WITH ap, trim(mitigation_text) as clean_mitigation
        WHERE clean_mitigation <> ''
        MERGE (m:Mitigation {uid: ap.uid + '_mitigation_' + toString(rand())})
        SET m.name = clean_mitigation,
            m.description = clean_mitigation,
            m.source = 'CAPEC',
            m.source_id = ap.uid,
            m.ingested_at = datetime(),
            m.from_source = true
        MERGE (m)-[r:MITIGATES {
            mapping_type: 'CAPEC_MITIGATION',
            confidence: 'high',
            source: 'CAPEC_Mitigations_Field'
        }]->(ap)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(m) as nodes_created, count(r) as relationships_created
        """

        try:
            # Clear existing mitigations
            self.db.execute_cypher_single(clear_cypher)

            # Create mitigations
            result = self.db.execute_cypher_single(cypher)

            relationships_created = (
                result.get("relationships_created", 0) if result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating CAPEC mitigations: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_cwe_mitigations(self) -> CrosswalkResult:
        """Create CWE mitigations from Weakness potential_mitigations data"""
        # First, clear existing CWE mitigations
        clear_cypher = """
        MATCH (m:Mitigation {source: 'CWE'})
        DETACH DELETE m
        """

        # Main mitigations extraction
        cypher = """
        MATCH (w:Weakness)
        WHERE w.potential_mitigations IS NOT NULL AND w.potential_mitigations <> ''
        UNWIND w.potential_mitigations as mitigation_item
        WITH w, mitigation_item
        WHERE mitigation_item <> '' AND mitigation_item <> 'Phase: Implementation; Description:'
        WITH w, trim(mitigation_item) as clean_mitigation
        WHERE clean_mitigation <> '' AND clean_mitigation <> 'Phase: Implementation; Description:'
        MERGE (m:Mitigation {uid: w.uid + '_mitigation_' + toString(rand())})
        SET m.name = clean_mitigation,
            m.description = clean_mitigation,
            m.source = 'CWE',
            m.source_id = w.uid,
            m.ingested_at = datetime(),
            m.from_source = true
        MERGE (m)-[r:MITIGATES {
            mapping_type: 'CWE_MITIGATION',
            confidence: 'high',
            source: 'CWE_Potential_Mitigations_Field'
        }]->(w)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(m) as nodes_created, count(r) as relationships_created
        """

        try:
            # Clear existing CWE mitigations
            self.db.execute_cypher_single(clear_cypher)

            # Create mitigations
            result = self.db.execute_cypher_single(cypher)

            relationships_created = (
                result.get("relationships_created", 0) if result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating CWE mitigations: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_attack_mitigations(self) -> CrosswalkResult:
        """Create ATTACK mitigations from course-of-action objects"""
        # First, clear existing ATTACK mitigations
        clear_cypher = """
        MATCH (m:Mitigation {source: 'ATT&CK'})
        DETACH DELETE m
        """

        # Main mitigations extraction
        cypher = """
        CALL apoc.load.json('/import/attack/enterprise-attack.jsonl') YIELD value
        WHERE value.type = 'course-of-action'
        WITH value, value.external_references[0].external_id as external_id
        WHERE external_id IS NOT NULL
        MERGE (m:Mitigation {uid: external_id})
        SET m.name = value.name,
            m.description = value.description,
            m.source = 'ATT&CK',
            m.source_id = external_id,
            m.ingested_at = datetime(),
            m.from_source = true
        RETURN count(m) as nodes_created
        """

        # Create relationships to Techniques
        relationships_cypher = """
        MATCH (m:Mitigation {source: 'ATT&CK'})
        MATCH (t:Technique {source: 'ATT&CK'})
        WHERE m.source_id = t.uid OR m.name CONTAINS t.name
        MERGE (m)-[r:MITIGATES {
            mapping_type: 'ATTACK_MITIGATION',
            confidence: 'high',
            source: 'ATTACK_Course_of_Action'
        }]->(t)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        try:
            # Clear existing ATTACK mitigations
            self.db.execute_cypher_single(clear_cypher)

            # Create mitigations
            result = self.db.execute_cypher_single(cypher)
            nodes_created = result.get("nodes_created", 0) if result else 0

            # Create relationships
            rel_result = self.db.execute_cypher_single(relationships_cypher)
            relationships_created = (
                rel_result.get("relationships_created", 0) if rel_result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating ATTACK mitigations: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_workrole_attack_crosswalk(self) -> CrosswalkResult:
        """Create WorkRole-ATTACK cross-domain relationships using semantic matching

        DISABLED: Synthetic WORKS_WITH relationships have been removed.
        This method no longer creates heuristic-based cross-dataset relationships
        between WorkRole and ATT&CK Technique nodes.

        These relationships were created using keyword matching between WorkRole
        titles and ATT&CK technique names, which were not based on actual
        framework mappings.

        Returns 0 relationships to indicate the method is disabled.
        """
        # Synthetic relationships have been removed - return early
        return CrosswalkResult(success=True, relationships_created=0)

        # DISABLED CODE BELOW - kept for reference only
        # Clear existing WorkRole-ATTACK relationships
        # clear_cypher = """
        # MATCH (wr:WorkRole)-[r:WORKS_WITH]->(t:Technique {source: 'ATT&CK'})
        # DELETE r
        # """

        # Create WorkRole-ATTACK relationships using semantic matching
        cypher = """
        MATCH (wr:WorkRole), (t:Technique {source: 'ATT&CK'})
        WHERE (
            // Match based on WorkRole title keywords
            (toLower(wr.title) CONTAINS 'security' AND toLower(t.name) CONTAINS 'security') OR
            (toLower(wr.title) CONTAINS 'threat' AND toLower(t.name) CONTAINS 'threat') OR
            (toLower(wr.title) CONTAINS 'incident' AND toLower(t.name) CONTAINS 'incident') OR
            (toLower(wr.title) CONTAINS 'vulnerability' AND toLower(t.name) CONTAINS 'vulnerability') OR
            (toLower(wr.title) CONTAINS 'malware' AND toLower(t.name) CONTAINS 'malware') OR
            (toLower(wr.title) CONTAINS 'network' AND toLower(t.name) CONTAINS 'network') OR
            (toLower(wr.title) CONTAINS 'forensic' AND toLower(t.name) CONTAINS 'forensic') OR
            (toLower(wr.title) CONTAINS 'analyst' AND toLower(t.name) CONTAINS 'analysis') OR
            (toLower(wr.title) CONTAINS 'defense' AND toLower(t.name) CONTAINS 'defense') OR
            (toLower(wr.title) CONTAINS 'monitor' AND toLower(t.name) CONTAINS 'monitor') OR
            (toLower(wr.title) CONTAINS 'detect' AND toLower(t.name) CONTAINS 'detect') OR
            (toLower(wr.title) CONTAINS 'respond' AND toLower(t.name) CONTAINS 'respond') OR
            (toLower(wr.title) CONTAINS 'investigate' AND toLower(t.name) CONTAINS 'investigate') OR
            (toLower(wr.title) CONTAINS 'penetration' AND toLower(t.name) CONTAINS 'penetration') OR
            (toLower(wr.title) CONTAINS 'red team' AND toLower(t.name) CONTAINS 'red team') OR
            (toLower(wr.title) CONTAINS 'blue team' AND toLower(t.name) CONTAINS 'blue team') OR
            (toLower(wr.title) CONTAINS 'soc' AND toLower(t.name) CONTAINS 'soc') OR
            (toLower(wr.title) CONTAINS 'siem' AND toLower(t.name) CONTAINS 'siem') OR
            (toLower(wr.title) CONTAINS 'firewall' AND toLower(t.name) CONTAINS 'firewall') OR
            (toLower(wr.title) CONTAINS 'endpoint' AND toLower(t.name) CONTAINS 'endpoint') OR
            (toLower(wr.title) CONTAINS 'cloud' AND toLower(t.name) CONTAINS 'cloud') OR
            (toLower(wr.title) CONTAINS 'identity' AND toLower(t.name) CONTAINS 'identity') OR
            (toLower(wr.title) CONTAINS 'privilege' AND toLower(t.name) CONTAINS 'privilege') OR
            (toLower(wr.title) CONTAINS 'credential' AND toLower(t.name) CONTAINS 'credential') OR
            (toLower(wr.title) CONTAINS 'persistence' AND toLower(t.name) CONTAINS 'persistence') OR
            (toLower(wr.title) CONTAINS 'lateral' AND toLower(t.name) CONTAINS 'lateral') OR
            (toLower(wr.title) CONTAINS 'exfiltration' AND toLower(t.name) CONTAINS 'exfiltration') OR
            (toLower(wr.title) CONTAINS 'command' AND toLower(t.name) CONTAINS 'command') OR
            (toLower(wr.title) CONTAINS 'execution' AND toLower(t.name) CONTAINS 'execution') OR
            (toLower(wr.title) CONTAINS 'collection' AND toLower(t.name) CONTAINS 'collection') OR
            (toLower(wr.title) CONTAINS 'impact' AND toLower(t.name) CONTAINS 'impact') OR
            (toLower(wr.title) CONTAINS 'initial' AND toLower(t.name) CONTAINS 'initial') OR
            (toLower(wr.title) CONTAINS 'discovery' AND toLower(t.name) CONTAINS 'discovery') OR
            (toLower(wr.title) CONTAINS 'defense' AND toLower(t.name) CONTAINS 'defense') OR
            (toLower(wr.title) CONTAINS 'evasion' AND toLower(t.name) CONTAINS 'evasion') OR
            (toLower(wr.title) CONTAINS 'credential' AND toLower(t.name) CONTAINS 'credential') OR
            (toLower(wr.title) CONTAINS 'access' AND toLower(t.name) CONTAINS 'access') OR
            (toLower(wr.title) CONTAINS 'privilege' AND toLower(t.name) CONTAINS 'privilege') OR
            (toLower(wr.title) CONTAINS 'persistence' AND toLower(t.name) CONTAINS 'persistence') OR
            (toLower(wr.title) CONTAINS 'lateral' AND toLower(t.name) CONTAINS 'lateral') OR
            (toLower(wr.title) CONTAINS 'collection' AND toLower(t.name) CONTAINS 'collection') OR
            (toLower(wr.title) CONTAINS 'exfiltration' AND toLower(t.name) CONTAINS 'exfiltration') OR
            (toLower(wr.title) CONTAINS 'impact' AND toLower(t.name) CONTAINS 'impact')
        )
        MERGE (wr)-[r:WORKS_WITH {
            mapping_type: 'WORKROLE_TO_ATTACK',
            confidence: 'medium',
            source: 'Semantic_Matching',
            relationship_type: 'can_analyze'
        }]->(t)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        try:
            # Clear existing relationships
            self.db.execute_cypher_single(clear_cypher)

            # Create new relationships
            result = self.db.execute_cypher_single(cypher)
            relationships_created = (
                result.get("relationships_created", 0) if result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating WorkRole-ATTACK crosswalk: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_cve_attack_crosswalk(self) -> CrosswalkResult:
        """Create CVE-ATTACK cross-domain relationships using simple semantic matching"""
        # Clear existing CVE-ATTACK relationships
        clear_cypher = """
        MATCH (v:Vulnerability)-[r:CAN_BE_EXPLOITED_BY]->(t:Technique {source: 'ATT&CK'})
        DELETE r
        """

        # Create CVE-ATTACK relationships using minimal semantic matching
        cypher = """
        MATCH (v:Vulnerability), (t:Technique {source: 'ATT&CK'})
        WHERE (
            // Core vulnerability types only
            (toLower(v.description) CONTAINS 'buffer overflow' AND toLower(t.name) CONTAINS 'buffer') OR
            (toLower(v.description) CONTAINS 'sql injection' AND toLower(t.name) CONTAINS 'sql') OR
            (toLower(v.description) CONTAINS 'xss' AND toLower(t.name) CONTAINS 'xss') OR
            (toLower(v.description) CONTAINS 'csrf' AND toLower(t.name) CONTAINS 'csrf') OR
            (toLower(v.description) CONTAINS 'privilege escalation' AND toLower(t.name) CONTAINS 'privilege') OR
            (toLower(v.description) CONTAINS 'remote code execution' AND toLower(t.name) CONTAINS 'execution') OR
            (toLower(v.description) CONTAINS 'local code execution' AND toLower(t.name) CONTAINS 'execution') OR
            (toLower(v.description) CONTAINS 'denial of service' AND toLower(t.name) CONTAINS 'denial') OR
            (toLower(v.description) CONTAINS 'dos' AND toLower(t.name) CONTAINS 'denial') OR
            (toLower(v.description) CONTAINS 'information disclosure' AND toLower(t.name) CONTAINS 'disclosure') OR
            (toLower(v.description) CONTAINS 'memory corruption' AND toLower(t.name) CONTAINS 'memory') OR
            (toLower(v.description) CONTAINS 'heap overflow' AND toLower(t.name) CONTAINS 'heap') OR
            (toLower(v.description) CONTAINS 'stack overflow' AND toLower(t.name) CONTAINS 'stack') OR
            (toLower(v.description) CONTAINS 'integer overflow' AND toLower(t.name) CONTAINS 'integer') OR
            (toLower(v.description) CONTAINS 'format string' AND toLower(t.name) CONTAINS 'format') OR
            (toLower(v.description) CONTAINS 'use after free' AND toLower(t.name) CONTAINS 'use') OR
            (toLower(v.description) CONTAINS 'double free' AND toLower(t.name) CONTAINS 'free') OR
            (toLower(v.description) CONTAINS 'null pointer' AND toLower(t.name) CONTAINS 'null') OR
            (toLower(v.description) CONTAINS 'race condition' AND toLower(t.name) CONTAINS 'race') OR
            (toLower(v.description) CONTAINS 'path traversal' AND toLower(t.name) CONTAINS 'path') OR
            (toLower(v.description) CONTAINS 'directory traversal' AND toLower(t.name) CONTAINS 'directory') OR
            (toLower(v.description) CONTAINS 'command injection' AND toLower(t.name) CONTAINS 'command') OR
            (toLower(v.description) CONTAINS 'code injection' AND toLower(t.name) CONTAINS 'injection') OR
            (toLower(v.description) CONTAINS 'ldap injection' AND toLower(t.name) CONTAINS 'ldap') OR
            (toLower(v.description) CONTAINS 'xml injection' AND toLower(t.name) CONTAINS 'xml') OR
            (toLower(v.description) CONTAINS 'xxe' AND toLower(t.name) CONTAINS 'xml') OR
            (toLower(v.description) CONTAINS 'deserialization' AND toLower(t.name) CONTAINS 'deserialization') OR
            (toLower(v.description) CONTAINS 'ssrf' AND toLower(t.name) CONTAINS 'server') OR
            (toLower(v.description) CONTAINS 'server side request forgery' AND toLower(t.name) CONTAINS 'server') OR
            (toLower(v.description) CONTAINS 'open redirect' AND toLower(t.name) CONTAINS 'redirect') OR
            (toLower(v.description) CONTAINS 'file inclusion' AND toLower(t.name) CONTAINS 'file') OR
            (toLower(v.description) CONTAINS 'authentication bypass' AND toLower(t.name) CONTAINS 'authentication') OR
            (toLower(v.description) CONTAINS 'authorization bypass' AND toLower(t.name) CONTAINS 'authorization') OR
            (toLower(v.description) CONTAINS 'session fixation' AND toLower(t.name) CONTAINS 'session') OR
            (toLower(v.description) CONTAINS 'session hijacking' AND toLower(t.name) CONTAINS 'session') OR
            (toLower(v.description) CONTAINS 'credential theft' AND toLower(t.name) CONTAINS 'credential') OR
            (toLower(v.description) CONTAINS 'password cracking' AND toLower(t.name) CONTAINS 'password') OR
            (toLower(v.description) CONTAINS 'brute force' AND toLower(t.name) CONTAINS 'brute') OR
            (toLower(v.description) CONTAINS 'side channel' AND toLower(t.name) CONTAINS 'side') OR
            (toLower(v.description) CONTAINS 'timing attack' AND toLower(t.name) CONTAINS 'timing') OR
            (toLower(v.description) CONTAINS 'spectre' AND toLower(t.name) CONTAINS 'spectre') OR
            (toLower(v.description) CONTAINS 'meltdown' AND toLower(t.name) CONTAINS 'meltdown') OR
            (toLower(v.description) CONTAINS 'rowhammer' AND toLower(t.name) CONTAINS 'rowhammer') OR
            (toLower(v.description) CONTAINS 'firmware' AND toLower(t.name) CONTAINS 'firmware') OR
            (toLower(v.description) CONTAINS 'bios' AND toLower(t.name) CONTAINS 'bios') OR
            (toLower(v.description) CONTAINS 'hypervisor' AND toLower(t.name) CONTAINS 'hypervisor') OR
            (toLower(v.description) CONTAINS 'virtualization' AND toLower(t.name) CONTAINS 'virtual') OR
            (toLower(v.description) CONTAINS 'container' AND toLower(t.name) CONTAINS 'container') OR
            (toLower(v.description) CONTAINS 'docker' AND toLower(t.name) CONTAINS 'docker') OR
            (toLower(v.description) CONTAINS 'cloud' AND toLower(t.name) CONTAINS 'cloud') OR
            (toLower(v.description) CONTAINS 'network' AND toLower(t.name) CONTAINS 'network') OR
            (toLower(v.description) CONTAINS 'wireless' AND toLower(t.name) CONTAINS 'wireless') OR
            (toLower(v.description) CONTAINS 'bluetooth' AND toLower(t.name) CONTAINS 'bluetooth') OR
            (toLower(v.description) CONTAINS 'wifi' AND toLower(t.name) CONTAINS 'wifi') OR
            (toLower(v.description) CONTAINS 'iot' AND toLower(t.name) CONTAINS 'iot') OR
            (toLower(v.description) CONTAINS 'embedded' AND toLower(t.name) CONTAINS 'embedded') OR
            (toLower(v.description) CONTAINS 'scada' AND toLower(t.name) CONTAINS 'scada') OR
            (toLower(v.description) CONTAINS 'industrial' AND toLower(t.name) CONTAINS 'industrial') OR
            (toLower(v.description) CONTAINS 'medical' AND toLower(t.name) CONTAINS 'medical') OR
            (toLower(v.description) CONTAINS 'financial' AND toLower(t.name) CONTAINS 'financial') OR
            (toLower(v.description) CONTAINS 'cryptocurrency' AND toLower(t.name) CONTAINS 'crypto') OR
            (toLower(v.description) CONTAINS 'blockchain' AND toLower(t.name) CONTAINS 'blockchain') OR
            (toLower(v.description) CONTAINS 'ai' AND toLower(t.name) CONTAINS 'ai') OR
            (toLower(v.description) CONTAINS 'machine learning' AND toLower(t.name) CONTAINS 'machine') OR
            (toLower(v.description) CONTAINS 'backdoor' AND toLower(t.name) CONTAINS 'backdoor') OR
            (toLower(v.description) CONTAINS 'trojan' AND toLower(t.name) CONTAINS 'trojan') OR
            (toLower(v.description) CONTAINS 'virus' AND toLower(t.name) CONTAINS 'virus') OR
            (toLower(v.description) CONTAINS 'worm' AND toLower(t.name) CONTAINS 'worm') OR
            (toLower(v.description) CONTAINS 'rootkit' AND toLower(t.name) CONTAINS 'rootkit') OR
            (toLower(v.description) CONTAINS 'keylogger' AND toLower(t.name) CONTAINS 'keylogger') OR
            (toLower(v.description) CONTAINS 'spyware' AND toLower(t.name) CONTAINS 'spyware') OR
            (toLower(v.description) CONTAINS 'ransomware' AND toLower(t.name) CONTAINS 'ransomware') OR
            (toLower(v.description) CONTAINS 'botnet' AND toLower(t.name) CONTAINS 'botnet') OR
            (toLower(v.description) CONTAINS 'ddos' AND toLower(t.name) CONTAINS 'ddos') OR
            (toLower(v.description) CONTAINS 'spoofing' AND toLower(t.name) CONTAINS 'spoofing') OR
            (toLower(v.description) CONTAINS 'man in the middle' AND toLower(t.name) CONTAINS 'middle') OR
            (toLower(v.description) CONTAINS 'mitm' AND toLower(t.name) CONTAINS 'middle') OR
            (toLower(v.description) CONTAINS 'eavesdropping' AND toLower(t.name) CONTAINS 'eavesdrop') OR
            (toLower(v.description) CONTAINS 'sniffing' AND toLower(t.name) CONTAINS 'sniff') OR
            (toLower(v.description) CONTAINS 'packet capture' AND toLower(t.name) CONTAINS 'packet') OR
            (toLower(v.description) CONTAINS 'network scanning' AND toLower(t.name) CONTAINS 'scan') OR
            (toLower(v.description) CONTAINS 'port scanning' AND toLower(t.name) CONTAINS 'port') OR
            (toLower(v.description) CONTAINS 'vulnerability scanning' AND toLower(t.name) CONTAINS 'vulnerability') OR
            (toLower(v.description) CONTAINS 'penetration testing' AND toLower(t.name) CONTAINS 'penetration') OR
            (toLower(v.description) CONTAINS 'threat hunting' AND toLower(t.name) CONTAINS 'threat') OR
            (toLower(v.description) CONTAINS 'incident response' AND toLower(t.name) CONTAINS 'incident') OR
            (toLower(v.description) CONTAINS 'forensics' AND toLower(t.name) CONTAINS 'forensic') OR
            (toLower(v.description) CONTAINS 'malware analysis' AND toLower(t.name) CONTAINS 'malware') OR
            (toLower(v.description) CONTAINS 'reverse engineering' AND toLower(t.name) CONTAINS 'reverse') OR
            (toLower(v.description) CONTAINS 'fuzzing' AND toLower(t.name) CONTAINS 'fuzz') OR
            (toLower(v.description) CONTAINS 'exploit development' AND toLower(t.name) CONTAINS 'exploit') OR
            (toLower(v.description) CONTAINS 'payload' AND toLower(t.name) CONTAINS 'payload') OR
            (toLower(v.description) CONTAINS 'shellcode' AND toLower(t.name) CONTAINS 'shellcode') OR
            (toLower(v.description) CONTAINS 'aslr' AND toLower(t.name) CONTAINS 'aslr') OR
            (toLower(v.description) CONTAINS 'dep' AND toLower(t.name) CONTAINS 'dep') OR
            (toLower(v.description) CONTAINS 'stack canary' AND toLower(t.name) CONTAINS 'canary') OR
            (toLower(v.description) CONTAINS 'control flow' AND toLower(t.name) CONTAINS 'control') OR
            (toLower(v.description) CONTAINS 'return address' AND toLower(t.name) CONTAINS 'return') OR
            (toLower(v.description) CONTAINS 'function pointer' AND toLower(t.name) CONTAINS 'function') OR
            (toLower(v.description) CONTAINS 'code signing' AND toLower(t.name) CONTAINS 'signing') OR
            (toLower(v.description) CONTAINS 'integrity' AND toLower(t.name) CONTAINS 'integrity') OR
            (toLower(v.description) CONTAINS 'tamper' AND toLower(t.name) CONTAINS 'tamper') OR
            (toLower(v.description) CONTAINS 'anti-debug' AND toLower(t.name) CONTAINS 'debug') OR
            (toLower(v.description) CONTAINS 'anti-vm' AND toLower(t.name) CONTAINS 'vm') OR
            (toLower(v.description) CONTAINS 'sandbox' AND toLower(t.name) CONTAINS 'sandbox') OR
            (toLower(v.description) CONTAINS 'emulation' AND toLower(t.name) CONTAINS 'emulation') OR
            (toLower(v.description) CONTAINS 'isolation' AND toLower(t.name) CONTAINS 'isolation') OR
            (toLower(v.description) CONTAINS 'privilege separation' AND toLower(t.name) CONTAINS 'privilege') OR
            (toLower(v.description) CONTAINS 'access control' AND toLower(t.name) CONTAINS 'access') OR
            (toLower(v.description) CONTAINS 'authentication' AND toLower(t.name) CONTAINS 'authentication') OR
            (toLower(v.description) CONTAINS 'authorization' AND toLower(t.name) CONTAINS 'authorization') OR
            (toLower(v.description) CONTAINS 'multi factor' AND toLower(t.name) CONTAINS 'multi') OR
            (toLower(v.description) CONTAINS 'mfa' AND toLower(t.name) CONTAINS 'mfa') OR
            (toLower(v.description) CONTAINS 'biometric' AND toLower(t.name) CONTAINS 'biometric') OR
            (toLower(v.description) CONTAINS 'fingerprint' AND toLower(t.name) CONTAINS 'fingerprint') OR
            (toLower(v.description) CONTAINS 'face recognition' AND toLower(t.name) CONTAINS 'face') OR
            (toLower(v.description) CONTAINS 'voice recognition' AND toLower(t.name) CONTAINS 'voice') OR
            (toLower(v.description) CONTAINS 'behavioral' AND toLower(t.name) CONTAINS 'behavioral') OR
            (toLower(v.description) CONTAINS 'risk based' AND toLower(t.name) CONTAINS 'risk') OR
            (toLower(v.description) CONTAINS 'adaptive' AND toLower(t.name) CONTAINS 'adaptive') OR
            (toLower(v.description) CONTAINS 'contextual' AND toLower(t.name) CONTAINS 'contextual') OR
            (toLower(v.description) CONTAINS 'location based' AND toLower(t.name) CONTAINS 'location') OR
            (toLower(v.description) CONTAINS 'device based' AND toLower(t.name) CONTAINS 'device') OR
            (toLower(v.description) CONTAINS 'network based' AND toLower(t.name) CONTAINS 'network') OR
            (toLower(v.description) CONTAINS 'geolocation' AND toLower(t.name) CONTAINS 'geo') OR
            (toLower(v.description) CONTAINS 'gps' AND toLower(t.name) CONTAINS 'gps') OR
            (toLower(v.description) CONTAINS 'cellular' AND toLower(t.name) CONTAINS 'cellular') OR
            (toLower(v.description) CONTAINS 'nfc' AND toLower(t.name) CONTAINS 'nfc') OR
            (toLower(v.description) CONTAINS 'rfid' AND toLower(t.name) CONTAINS 'rfid') OR
            (toLower(v.description) CONTAINS 'beacon' AND toLower(t.name) CONTAINS 'beacon') OR
            (toLower(v.description) CONTAINS 'proximity' AND toLower(t.name) CONTAINS 'proximity') OR
            (toLower(v.description) CONTAINS 'tracking' AND toLower(t.name) CONTAINS 'tracking') OR
            (toLower(v.description) CONTAINS 'monitoring' AND toLower(t.name) CONTAINS 'monitoring') OR
            (toLower(v.description) CONTAINS 'surveillance' AND toLower(t.name) CONTAINS 'surveillance') OR
            (toLower(v.description) CONTAINS 'privacy' AND toLower(t.name) CONTAINS 'privacy') OR
            (toLower(v.description) CONTAINS 'gdpr' AND toLower(t.name) CONTAINS 'gdpr') OR
            (toLower(v.description) CONTAINS 'hipaa' AND toLower(t.name) CONTAINS 'hipaa') OR
            (toLower(v.description) CONTAINS 'pci' AND toLower(t.name) CONTAINS 'pci') OR
            (toLower(v.description) CONTAINS 'nist' AND toLower(t.name) CONTAINS 'nist') OR
            (toLower(v.description) CONTAINS 'compliance' AND toLower(t.name) CONTAINS 'compliance') OR
            (toLower(v.description) CONTAINS 'governance' AND toLower(t.name) CONTAINS 'governance') OR
            (toLower(v.description) CONTAINS 'risk management' AND toLower(t.name) CONTAINS 'risk') OR
            (toLower(v.description) CONTAINS 'threat modeling' AND toLower(t.name) CONTAINS 'threat') OR
            (toLower(v.description) CONTAINS 'attack surface' AND toLower(t.name) CONTAINS 'attack') OR
            (toLower(v.description) CONTAINS 'security architecture' AND toLower(t.name) CONTAINS 'architecture') OR
            (toLower(v.description) CONTAINS 'security testing' AND toLower(t.name) CONTAINS 'testing') OR
            (toLower(v.description) CONTAINS 'security validation' AND toLower(t.name) CONTAINS 'validation') OR
            (toLower(v.description) CONTAINS 'security audit' AND toLower(t.name) CONTAINS 'audit') OR
            (toLower(v.description) CONTAINS 'security policy' AND toLower(t.name) CONTAINS 'policy') OR
            (toLower(v.description) CONTAINS 'security standard' AND toLower(t.name) CONTAINS 'standard') OR
            (toLower(v.description) CONTAINS 'security framework' AND toLower(t.name) CONTAINS 'framework') OR
            (toLower(v.description) CONTAINS 'security process' AND toLower(t.name) CONTAINS 'process') OR
            (toLower(v.description) CONTAINS 'security lifecycle' AND toLower(t.name) CONTAINS 'lifecycle') OR
            (toLower(v.description) CONTAINS 'security maturity' AND toLower(t.name) CONTAINS 'maturity') OR
            (toLower(v.description) CONTAINS 'security capability' AND toLower(t.name) CONTAINS 'capability') OR
            (toLower(v.description) CONTAINS 'security resilience' AND toLower(t.name) CONTAINS 'resilience') OR
            (toLower(v.description) CONTAINS 'security recovery' AND toLower(t.name) CONTAINS 'recovery') OR
            (toLower(v.description) CONTAINS 'security continuity' AND toLower(t.name) CONTAINS 'continuity') OR
            (toLower(v.description) CONTAINS 'security availability' AND toLower(t.name) CONTAINS 'availability') OR
            (toLower(v.description) CONTAINS 'security reliability' AND toLower(t.name) CONTAINS 'reliability') OR
            (toLower(v.description) CONTAINS 'security performance' AND toLower(t.name) CONTAINS 'performance') OR
            (toLower(v.description) CONTAINS 'security quality' AND toLower(t.name) CONTAINS 'quality') OR
            (toLower(v.description) CONTAINS 'security assurance' AND toLower(t.name) CONTAINS 'assurance') OR
            (toLower(v.description) CONTAINS 'security confidence' AND toLower(t.name) CONTAINS 'confidence') OR
            (toLower(v.description) CONTAINS 'security trust' AND toLower(t.name) CONTAINS 'trust') OR
            (toLower(v.description) CONTAINS 'security awareness' AND toLower(t.name) CONTAINS 'awareness') OR
            (toLower(v.description) CONTAINS 'security education' AND toLower(t.name) CONTAINS 'education') OR
            (toLower(v.description) CONTAINS 'security training' AND toLower(t.name) CONTAINS 'training') OR
            (toLower(v.description) CONTAINS 'security culture' AND toLower(t.name) CONTAINS 'culture') OR
            (toLower(v.description) CONTAINS 'security behavior' AND toLower(t.name) CONTAINS 'behavior') OR
            (toLower(v.description) CONTAINS 'security mindset' AND toLower(t.name) CONTAINS 'mindset') OR
            (toLower(v.description) CONTAINS 'security approach' AND toLower(t.name) CONTAINS 'approach') OR
            (toLower(v.description) CONTAINS 'security strategy' AND toLower(t.name) CONTAINS 'strategy') OR
            (toLower(v.description) CONTAINS 'security technique' AND toLower(t.name) CONTAINS 'technique') OR
            (toLower(v.description) CONTAINS 'security method' AND toLower(t.name) CONTAINS 'method') OR
            (toLower(v.description) CONTAINS 'security tool' AND toLower(t.name) CONTAINS 'tool') OR
            (toLower(v.description) CONTAINS 'security technology' AND toLower(t.name) CONTAINS 'technology') OR
            (toLower(v.description) CONTAINS 'security solution' AND toLower(t.name) CONTAINS 'solution') OR
            (toLower(v.description) CONTAINS 'security product' AND toLower(t.name) CONTAINS 'product') OR
            (toLower(v.description) CONTAINS 'security service' AND toLower(t.name) CONTAINS 'service') OR
            (toLower(v.description) CONTAINS 'security system' AND toLower(t.name) CONTAINS 'system') OR
            (toLower(v.description) CONTAINS 'security platform' AND toLower(t.name) CONTAINS 'platform') OR
            (toLower(v.description) CONTAINS 'security infrastructure' AND toLower(t.name) CONTAINS 'infrastructure') OR
            (toLower(v.description) CONTAINS 'security environment' AND toLower(t.name) CONTAINS 'environment') OR
            (toLower(v.description) CONTAINS 'security domain' AND toLower(t.name) CONTAINS 'domain') OR
            (toLower(v.description) CONTAINS 'security area' AND toLower(t.name) CONTAINS 'area') OR
            (toLower(v.description) CONTAINS 'security field' AND toLower(t.name) CONTAINS 'field') OR
            (toLower(v.description) CONTAINS 'security discipline' AND toLower(t.name) CONTAINS 'discipline') OR
            (toLower(v.description) CONTAINS 'security practice' AND toLower(t.name) CONTAINS 'practice') OR
            (toLower(v.description) CONTAINS 'security profession' AND toLower(t.name) CONTAINS 'profession') OR
            (toLower(v.description) CONTAINS 'security career' AND toLower(t.name) CONTAINS 'career') OR
            (toLower(v.description) CONTAINS 'security role' AND toLower(t.name) CONTAINS 'role') OR
            (toLower(v.description) CONTAINS 'security responsibility' AND toLower(t.name) CONTAINS 'responsibility') OR
            (toLower(v.description) CONTAINS 'security task' AND toLower(t.name) CONTAINS 'task') OR
            (toLower(v.description) CONTAINS 'security activity' AND toLower(t.name) CONTAINS 'activity') OR
            (toLower(v.description) CONTAINS 'security operation' AND toLower(t.name) CONTAINS 'operation') OR
            (toLower(v.description) CONTAINS 'security skill' AND toLower(t.name) CONTAINS 'skill') OR
            (toLower(v.description) CONTAINS 'security knowledge' AND toLower(t.name) CONTAINS 'knowledge') OR
            (toLower(v.description) CONTAINS 'security expertise' AND toLower(t.name) CONTAINS 'expertise') OR
            (toLower(v.description) CONTAINS 'security experience' AND toLower(t.name) CONTAINS 'experience') OR
            (toLower(v.description) CONTAINS 'security certification' AND toLower(t.name) CONTAINS 'certification') OR
            (toLower(v.description) CONTAINS 'security credential' AND toLower(t.name) CONTAINS 'credential') OR
            (toLower(v.description) CONTAINS 'security authorization' AND toLower(t.name) CONTAINS 'authorization') OR
            (toLower(v.description) CONTAINS 'security access' AND toLower(t.name) CONTAINS 'access') OR
            (toLower(v.description) CONTAINS 'security permission' AND toLower(t.name) CONTAINS 'permission') OR
            (toLower(v.description) CONTAINS 'security privilege' AND toLower(t.name) CONTAINS 'privilege') OR
            (toLower(v.description) CONTAINS 'security value' AND toLower(t.name) CONTAINS 'value') OR
            (toLower(v.description) CONTAINS 'security importance' AND toLower(t.name) CONTAINS 'importance') OR
            (toLower(v.description) CONTAINS 'security significance' AND toLower(t.name) CONTAINS 'significance') OR
            (toLower(v.description) CONTAINS 'security relevance' AND toLower(t.name) CONTAINS 'relevance') OR
            (toLower(v.description) CONTAINS 'security applicability' AND toLower(t.name) CONTAINS 'applicability') OR
            (toLower(v.description) CONTAINS 'security suitability' AND toLower(t.name) CONTAINS 'suitability') OR
            (toLower(v.description) CONTAINS 'security adequacy' AND toLower(t.name) CONTAINS 'adequacy') OR
            (toLower(v.description) CONTAINS 'security completeness' AND toLower(t.name) CONTAINS 'completeness') OR
            (toLower(v.description) CONTAINS 'security thoroughness' AND toLower(t.name) CONTAINS 'thoroughness')
        )
        MERGE (v)-[r:CAN_BE_EXPLOITED_BY {
            mapping_type: 'CVE_TO_ATTACK',
            confidence: 'medium',
            source: 'Semantic_Matching',
            relationship_type: 'can_be_exploited_by'
        }]->(t)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(r) as relationships_created
        """

        try:
            # Clear existing relationships
            self.db.execute_cypher_single(clear_cypher)

            # Create new relationships
            result = self.db.execute_cypher_single(cypher)
            relationships_created = (
                result.get("relationships_created", 0) if result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating CVE-ATTACK crosswalk: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_cwe_categories_crosswalk(self) -> CrosswalkResult:
        """Create CWE categories crosswalk - data-driven approach for all CWEs"""
        cypher = """
        // Create CWE_CATEGORY nodes for each CWE
        MATCH (w:Weakness)
        MERGE (cc:CWECategory {uid: w.uid + '_CATEGORY'})
        SET cc.name = w.name,
            cc.description = w.description,
            cc.cwe_uid = w.uid,
            cc.type = 'CWE_CATEGORY',
            cc.source = 'CROSSWALK',
            cc.ingested_at = datetime()
        WITH cc

        // Link CVEs to their CWE categories
        MATCH (v:Vulnerability)-[r:HAS_WEAKNESS]->(w:Weakness)
        MATCH (cc2:CWECategory {cwe_uid: w.uid})
        MERGE (v)-[r2:IS_CWE_TYPE {
            mapping_type: 'CVE_TO_CWE_CATEGORY',
            confidence: 'high',
            source: 'CWE_MAPPING',
            cwe_uid: w.uid,
            weakness_type: 'Primary'
        }]->(cc2)
        ON CREATE SET r2.ingested_at = datetime()

        RETURN count(*) as relationships_created
        """

        try:
            result = self.db.execute_cypher_single(cypher)
            relationships_created = (
                result.get("relationships_created", 0) if result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating CWE categories crosswalk: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_cve_attack_crosswalk(self) -> CrosswalkResult:
        """Create CVE to ATT&CK crosswalk via CWE and CAPEC"""
        cypher = """
        // CVE → CWE → CAPEC → ATT&CK path
        MATCH (v:Vulnerability)-[:HAS_WEAKNESS]->(w:Weakness)-[:EXPLOITS]-(ap:AttackPattern)-[:RELATES_TO]->(t:Technique)
        MERGE (v)-[r:CAN_BE_EXPLOITED_BY {
            mapping_type: 'CVE_TO_ATTACK',
            confidence: 'high',
            source: 'CVE_CWE_CAPEC_ATTACK',
            cwe_uid: w.uid,
            capec_uid: ap.uid,
            attack_uid: t.uid,
            path: 'CVE→CWE→CAPEC→ATT&CK'
        }]->(t)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(DISTINCT v.uid + '-' + t.uid) as relationships_created
        """

        try:
            result = self.db.execute_cypher_single(cypher)
            relationships_created = (
                result.get("relationships_created", 0) if result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating CVE-ATTACK crosswalk: {e}")
            return CrosswalkResult(success=False, error=str(e))

    def _create_cve_capec_crosswalk(self) -> CrosswalkResult:
        """Create CVE to CAPEC crosswalk via CWE"""
        cypher = """
        // CVE → CWE → CAPEC path
        MATCH (v:Vulnerability)-[:HAS_WEAKNESS]->(w:Weakness)-[:EXPLOITS]-(ap:AttackPattern)
        MERGE (v)-[r:CAN_BE_EXPLOITED_BY {
            mapping_type: 'CVE_TO_CAPEC',
            confidence: 'high',
            source: 'CVE_CWE_CAPEC',
            cwe_uid: w.uid,
            capec_uid: ap.uid,
            path: 'CVE→CWE→CAPEC'
        }]->(ap)
        ON CREATE SET r.ingested_at = datetime()
        RETURN count(DISTINCT v.uid + '-' + ap.uid) as relationships_created
        """

        try:
            result = self.db.execute_cypher_single(cypher)
            relationships_created = (
                result.get("relationships_created", 0) if result else 0
            )

            return CrosswalkResult(
                success=True, relationships_created=relationships_created
            )

        except Exception as e:
            logger.error(f"Error creating CVE-CAPEC crosswalk: {e}")
            return CrosswalkResult(success=False, error=str(e))

    # -------------------------------------------------------------------------
    # Embedding generation (OpenAI or sentence-transformers)
    # -------------------------------------------------------------------------

    def generate_embeddings(
        self,
        dataset: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ) -> EmbeddingResult:
        """Generate dense embeddings for nodes using OpenAI embeddings API.

        Supports:
        - CVE (Vulnerability): Regenerates as dense embeddings (replaces sparse TF-IDF)
        - CAPEC (AttackPattern): Generates new dense embeddings
        - ATT&CK (Technique): Generates new dense embeddings
        - CWE (Weakness): Generates new dense embeddings
        - None/All: Generates for all datasets

        Args:
            dataset: Dataset name ('cve', 'capec', 'attack', 'cwe') or None for all
            model: OpenAI embedding model name (default: text-embedding-3-small)

        Returns:
            EmbeddingResult with success status and nodes processed count
        """
        try:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv
            from tqdm import tqdm

            load_dotenv()

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return EmbeddingResult(
                    success=False,
                    error="OPENAI_API_KEY not set. Required for embedding generation.",
                )

            openai_client = OpenAI(api_key=api_key)
            embedding_model_name = (
                model if model.startswith("text-") else "text-embedding-3-small"
            )
            logger.info(
                f"Using OpenAI embeddings API with model: {embedding_model_name}"
            )

            # Define datasets to process
            datasets_to_process = []
            if dataset is None or dataset.lower() == "all":
                datasets_to_process = ["cve", "capec", "attack", "cwe"]
            else:
                datasets_to_process = [dataset.lower()]

            total_processed = 0

            for dataset_name in datasets_to_process:
                logger.info(f"Processing embeddings for {dataset_name}...")

                # Define queries and labels for each dataset
                if dataset_name == "cve":
                    query = """
                    MATCH (n:Vulnerability)
                    WHERE n.uid IS NOT NULL
                    RETURN n.uid as uid, 
                           COALESCE(n.name, '') as name,
                           COALESCE(n.descriptions, '') as description
                    """
                    label = "Vulnerability"
                elif dataset_name == "capec":
                    query = """
                    MATCH (n:AttackPattern)
                    WHERE n.uid IS NOT NULL
                    RETURN n.uid as uid,
                           COALESCE(n.name, '') as name,
                           COALESCE(n.description, '') as description
                    """
                    label = "AttackPattern"
                elif dataset_name == "attack":
                    query = """
                    MATCH (n:Technique)
                    WHERE n.uid IS NOT NULL
                    RETURN n.uid as uid,
                           COALESCE(n.name, '') as name,
                           COALESCE(n.description, '') as description
                    """
                    label = "Technique"
                elif dataset_name == "cwe":
                    query = """
                    MATCH (n:Weakness)
                    WHERE n.uid IS NOT NULL
                    RETURN n.uid as uid,
                           COALESCE(n.name, '') as name,
                           COALESCE(n.description, '') as description
                    """
                    label = "Weakness"
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}, skipping")
                    continue

                # Fetch nodes
                nodes = self.db.execute_cypher(query)
                if not nodes:
                    logger.warning(f"No nodes found for {dataset_name}")
                    continue

                logger.info(f"Found {len(nodes)} nodes for {dataset_name}")

                # Prepare texts for embedding
                texts = []
                uids = []
                for node in nodes:
                    # Combine name + description for embedding
                    name = node.get("name", "") or ""
                    description = node.get("description", "") or ""
                    text = f"{name} {description}".strip()
                    if text:  # Only add non-empty texts
                        texts.append(text)
                        uids.append(node.get("uid"))

                if not texts:
                    logger.warning(f"No text content found for {dataset_name}")
                    continue

                # Generate embeddings in batches using OpenAI API
                logger.info(
                    f"Generating embeddings for {len(texts)} {dataset_name} nodes..."
                )
                batch_size = 100  # OpenAI can handle up to 2048 inputs per request
                embeddings = []

                for i in tqdm(
                    range(0, len(texts), batch_size), desc=f"Embedding {dataset_name}"
                ):
                    batch_texts = texts[i : i + batch_size]

                    # Use OpenAI embeddings API
                    response = openai_client.embeddings.create(
                        model=embedding_model_name,
                        input=batch_texts,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)

                # Store embeddings in Neo4j in batches
                logger.info(f"Storing {len(embeddings)} embeddings in Neo4j...")
                stored_count = 0

                for i in tqdm(
                    range(0, len(embeddings), self.batch_size),
                    desc=f"Storing {dataset_name}",
                ):
                    batch_uids = uids[i : i + self.batch_size]
                    batch_embeddings = embeddings[i : i + self.batch_size]

                    # Create batch update query
                    batch_params = []
                    for uid, embedding in zip(batch_uids, batch_embeddings):
                        # OpenAI embeddings are already lists, but handle numpy arrays if present
                        if hasattr(embedding, "tolist"):
                            embedding_list = embedding.tolist()
                        else:
                            embedding_list = embedding
                        batch_params.append(
                            {
                                "uid": uid,
                                "embedding": embedding_list,
                            }
                        )

                    # Update nodes with embeddings
                    update_query = f"""
                    UNWIND $batch as item
                    MATCH (n:{label} {{uid: item.uid}})
                    SET n.embedding = item.embedding
                    RETURN count(n) as updated
                    """

                    result = self.db.execute_cypher_single(
                        update_query, {"batch": batch_params}
                    )
                    if result:
                        stored_count += result.get("updated", 0)

                logger.info(f"Stored {stored_count} embeddings for {dataset_name}")
                total_processed += stored_count

            logger.info(
                f"Embedding generation complete. Total nodes processed: {total_processed}"
            )
            return EmbeddingResult(success=True, nodes_processed=total_processed)

        except ImportError as e:
            error_msg = f"sentence-transformers not installed: {e}. Install with: uv add sentence-transformers"
            logger.error(error_msg)
            return EmbeddingResult(success=False, error=error_msg)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return EmbeddingResult(success=False, error=str(e))
