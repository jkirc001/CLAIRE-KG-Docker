"""
Curated Schema Builder for CLAIRE-KG.

Builds filtered Neo4j schema text (node labels, properties, relationship types)
based on the question classifier's detected datasets and crosswalk groups. Used
by the Cypher generator to keep LLM prompts small (~80–85% token reduction vs
full schema) while still providing the relevant nodes and relationships.

Constants:
  - SINGLE_DATASET_SCHEMAS: One schema blob per dataset (CVE, CWE, CAPEC,
    ATT&CK, NICE, DCWF, ASSET, MITIGATION). Each includes Nodes, Relationships,
    and Notes (property names, relationship direction, query hints).
  - JOINT_SCHEMAS: Schema blobs for crosswalks (e.g. CVE_CWE, CVE_ATTACK,
    NICE_CVE_ATTACK). Used when the question spans multiple datasets.

Entry point: build_curated_schema(datasets, crosswalks, user_query) returns a
single concatenated schema string or None. Joint schemas are applied first (by
priority); remaining datasets are filled from SINGLE_DATASET_SCHEMAS.

Reference: CLAIRE_KG_schema_packs_CORRECTED.txt.
"""

from typing import List, Dict, Optional

# -----------------------------------------------------------------------------
# Single-dataset schemas (one blob per primary dataset)
# -----------------------------------------------------------------------------

SINGLE_DATASET_SCHEMAS: Dict[str, str] = {
    "CVE": """
Nodes:
  (:Vulnerability {uid, name, descriptions, cvss_v31, cvss_score, published, year, severity, configurations, source})
Relationships:
  (:Vulnerability)-[:HAS_WEAKNESS]->(:Weakness)
  (:Vulnerability)-[:AFFECTS]->(:Asset)
  (:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(:AttackPattern)
Notes:
  - Use cvss_v31 for score, published for dates, year for year filters
  - Vendor and product data come from (:Asset {vendor, product})
  - CAN_BE_EXPLOITED_BY connects CVE → AttackPattern (324,336 instances); to reach Technique use: (v:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(ap:AttackPattern)-[:RELATES_TO]->(t:Technique)
  - Use 'descriptions' (plural), NOT 'description' for CVE text
""",
    "CWE": """
Nodes:
  (:Weakness {uid, name, description, notes, abstraction, alternate_terms, common_consequences, modes_of_introduction, weakness_ordinalities, source})
Relationships:
  (:AttackPattern)-[:EXPLOITS]->(:Weakness)
  (:Vulnerability)-[:HAS_WEAKNESS]->(:Weakness)
  (:Mitigation)-[:MITIGATES]->(:Weakness)
Notes:
  - CWEs referenced by uid (e.g., CWE-79)
  - CRITICAL: For specific CWE IDs (e.g., "CWE-79"), use w.uid = 'CWE-79' in MATCH clause: MATCH (w:Weakness {uid: 'CWE-79'})
  - DO NOT use toLower(w.name) CONTAINS 'cwe-79' for specific CWE IDs - use uid instead
  - Use toLower(w.name) CONTAINS only for semantic searches (e.g., "sql injection", "buffer overflow")
  - CWE weaknesses do NOT directly relate to ATT&CK techniques (only via CAPEC)
  - Use 'description' (singular), NOT 'descriptions'
  - WARNING: CRITICAL: MITIGATES direction is (m:Mitigation)-[:MITIGATES]->(w:Weakness), NOT (w)-[:MITIGATES]->(m)
  - For mitigation queries: MATCH (w:Weakness {uid: 'CWE-XX'})<-[:MITIGATES]-(m:Mitigation) or MATCH (m:Mitigation)-[:MITIGATES]->(w:Weakness {uid: 'CWE-XX'})
  - Direct CWE-to-Mitigation: MATCH (w:Weakness {uid: 'CWE-79'})<-[:MITIGATES]-(m:Mitigation) (no need to go through Vulnerabilities)
""",
    "CAPEC": """
Nodes:
  (:AttackPattern {uid, name, description, severity, execution_flow, notes, resources_required, abstraction, alternate_terms, related_attack_patterns, source, prerequisites, consequences})
Relationships:
  (:AttackPattern)-[:EXPLOITS]->(:Weakness)
  (:AttackPattern)-[:RELATES_TO]->(:Technique)
  (:Category)-[:HAS_MEMBER]->(:AttackPattern)
Notes:
  - CAPEC to ATT&CK: Use (ap:AttackPattern)-[:RELATES_TO]->(t:Technique)
  - Relationship direction is AP -> Technique (NOT Technique -> AP)
  - AttackPattern uses uid and name (NOT element_code or element_name). Category members: MATCH (c:Category {{name: 'X'}})-[:HAS_MEMBER]->(ap:AttackPattern) RETURN ap.uid, ap.name, ap.description
  - For "prerequisites" or "consequences" questions: MATCH (ap:AttackPattern {{uid: 'CAPEC-XX'}}) RETURN ap.uid, ap.name, ap.prerequisites (or ap.consequences) - use the property directly, no RELATES_TO needed
""",
    "ATTACK": """
Nodes:
  (:Technique {uid, name, description, level, x_mitre_domains, x_mitre_platforms, source})
  (:SubTechnique {uid, name, description, level, x_mitre_domains, x_mitre_platforms, source})
  (:Tactic {uid, name, description, level, x_mitre_domains, source})
Relationships:
  (:SubTechnique)-[:IS_PART_OF]->(:Technique)
  (:Technique)-[:USES_TACTIC]->(:Tactic)
Notes:
  - For sub-techniques: (t:Technique)<-[:IS_PART_OF]-(st:SubTechnique)
  - Technique -> Tactic uses :USES_TACTIC relationship
  - WARNING: CRITICAL: ONLY Technique nodes have USES_TACTIC relationships
  - WARNING: DO NOT use USES_TACTIC with Vulnerability nodes - they use CAN_BE_EXPLOITED_BY instead
""",
    "NICE": """
Nodes:
  (:WorkRole {uid, work_role, dcwf_code, title, text, source, doc_identifier, element_identifier, definition, element, ncwf_id})
  (:Task {uid, title, text, source, doc_identifier, element_identifier})
  (:Knowledge {uid, title, text, source, doc_identifier, element_identifier})
  (:Skill {uid, title, text, source, doc_identifier, element_identifier})
  (:Ability {uid, description, text, dcwf_number, nice_mapping, source})
  (:SpecialtyArea {specialty_prefix, element_name, element_code, source})
Relationships:
  (:WorkRole)-[:PERFORMS]->(:Task)
  (:WorkRole)-[:REQUIRES_KNOWLEDGE]->(:Knowledge)
  (:WorkRole)-[:REQUIRES_SKILL]->(:Skill)
  (:WorkRole)-[:REQUIRES_ABILITY]->(:Ability)
  (:WorkRole)-[:IN_SPECIALTY_AREA]->(:SpecialtyArea)
  (:WorkRole)-[:WORKS_WITH]->(:Technique)
  (:WorkRole)-[:WORKS_WITH]->(:Vulnerability)
  (:WorkRole)-[:WORKS_WITH]->(:AttackPattern)
Notes:
  - WorkRole properties vary by source:
    * DCWF-aligned roles (74 nodes): Have work_role, dcwf_code, definition, element, ncwf_id
    * NICE-only roles (41 nodes): Have title, text, doc_identifier, element_identifier
  - Recommendation: Use COALESCE(wr.work_role, wr.title) for role name queries
  - For Task/Knowledge/Skill: Use 'title' or 'text' properties (use coalesce for robustness); Task may use dcwf_number or element_identifier as uid
  - For Ability: Use 'description' or 'text' properties
  - SpecialtyArea: element_name may have abbreviations (e.g. Software Engineering (SE)). Use a single keyword in CONTAINS: for \"Secure Software Development\" use toLower(sa.element_name) CONTAINS 'software' (DB has \"Software Engineering (SE)\"), not the full phrase
  - For \"tasks under [specialty]\" or \"tasks fall under X\": use (wr)-[:PERFORMS]->(t:Task) and RETURN t.uid, t.title, t.text (do NOT return Ability)
""",
    "DCWF": """
Nodes:
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, element, ncwf_id})
  (:Ability {uid, description, text, dcwf_number, nice_mapping, source})
  (:SpecialtyArea {specialty_prefix, element_name, element_code, source})
Relationships:
  (:WorkRole)-[:REQUIRES_ABILITY]->(:Ability)
  (:WorkRole)-[:IN_SPECIALTY_AREA]->(:SpecialtyArea)
Notes:
  - DCWF integration through:
    * Ability.dcwf_number: Available on all Ability nodes
    * WorkRole.dcwf_code: Available on DCWF-aligned WorkRole nodes (74 out of 115)
    * WorkRole.work_role: Available on DCWF-aligned WorkRole nodes (74 out of 115)
  - Use WHERE clauses to filter for DCWF-aligned nodes if needed
  - Use COALESCE(wr.work_role, wr.title) for role name queries
""",
    "ASSET": """
Nodes:
  (:Asset {uid, name, product, vendor, cpe_type, source})
Relationships:
  (:Vulnerability)-[:AFFECTS]->(:Asset)
Notes:
  - WARNING: CRITICAL: Use 'vendor' property (NOT 'product') for vendor filtering
  - Example: WHERE toLower(a.vendor) = 'microsoft' (NOT a.product = 'Microsoft')
""",
    "MITIGATION": """
Nodes:
  (:Mitigation {uid, name, description, from_source, source, source_id})
Relationships:
  (:Mitigation)-[:MITIGATES]->(:Weakness)
Notes:
  - Security mitigations that address weaknesses
  - WARNING: CRITICAL: Relationship direction is (m:Mitigation)-[:MITIGATES]->(w:Weakness)
  - For queries: MATCH (w:Weakness)<-[:MITIGATES]-(m:Mitigation) or MATCH (m:Mitigation)-[:MITIGATES]->(w:Weakness)
  - DO NOT use: (w:Weakness)-[:MITIGATES]->(m:Mitigation) (wrong direction)
  - Use toLower(w.name) CONTAINS 'keyword' for partial name matching (not exact match)
  - RETURN m.uid AS uid, m.name AS title, m.description AS text (NOT v.uid or w.uid)
""",
}

# -----------------------------------------------------------------------------
# Joint (crosswalk) schemas: two or more datasets and their relationships
# -----------------------------------------------------------------------------

JOINT_SCHEMAS: Dict[str, str] = {
    "CVE_CWE": """
Nodes:
  (:Vulnerability {uid, name, descriptions, cvss_v31, cvss_score, published, year, severity, source})
  (:Weakness {uid, name, description, abstraction, source})
Relationships:
  (:Vulnerability)-[:HAS_WEAKNESS]->(:Weakness)
Notes:
  - Crosswalk: CVE to CWE (86,724 instances)
  - Use 'descriptions' (plural) for CVE, 'description' (singular) for CWE
""",
    "CVE_ASSET": """
Nodes:
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
  (:Asset {uid, name, product, vendor, cpe_type, source})
Relationships:
  (:Vulnerability)-[:AFFECTS]->(:Asset)
Notes:
  - Crosswalk: CVE to Asset (178,773 instances)
  - Use 'vendor' property (NOT 'product') for vendor filtering
""",
    "CVE_ATTACK": """
Nodes:
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
  (:AttackPattern {uid, name, description, severity, source})
  (:Technique {uid, name, description, level, x_mitre_domains, source})
  (:Tactic {uid, name, description, level, x_mitre_domains, source})
Relationships:
  (:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(:AttackPattern)
  (:AttackPattern)-[:RELATES_TO]->(:Technique)
  (:Technique)-[:USES_TACTIC]->(:Tactic)
Notes:
  - Crosswalk: CVE to ATT&CK via AttackPattern (324,336 CAN_BE_EXPLOITED_BY + 238 RELATES_TO instances)
  - CRITICAL: CAN_BE_EXPLOITED_BY connects Vulnerability → AttackPattern (NOT directly to Technique)
  - To get techniques for a CVE: MATCH (v:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(ap:AttackPattern)-[:RELATES_TO]->(t:Technique)
  - For reverse queries (Technique to CVE): MATCH (v:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(ap:AttackPattern)-[:RELATES_TO]->(t:Technique {uid: 'TXXXX'})
  - To include affected systems: Add MATCH (v)-[:AFFECTS]->(a:Asset)
""",
    "CAPEC_ATTACK": """
Nodes:
  (:AttackPattern {uid, name, description, severity, source})
  (:Technique {uid, name, description, level, x_mitre_domains, source})
  (:Tactic {uid, name, description, level, x_mitre_domains, source})
Relationships:
  (:AttackPattern)-[:RELATES_TO]->(:Technique)
  (:Technique)-[:USES_TACTIC]->(:Tactic)
Notes:
  - Crosswalk: CAPEC to ATT&CK (238 instances)
  - Relationship direction is AP -> Technique (NOT Technique -> AP)
""",
    "CAPEC_CWE": """
Nodes:
  (:AttackPattern {uid, name, description, severity, source})
  (:Weakness {uid, name, description, abstraction, source})
Relationships:
  (:AttackPattern)-[:EXPLOITS]->(:Weakness)
Notes:
  - Crosswalk: CAPEC to CWE (1,212 instances)
""",
    "NICE_ATTACK": """
Nodes:
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, source})
  (:Technique {uid, name, description, level, x_mitre_domains, source})
  (:Tactic {uid, name, description, level, x_mitre_domains, source})
Relationships:
  (:WorkRole)-[:WORKS_WITH]->(:Technique)
  (:Technique)-[:USES_TACTIC]->(:Tactic)
Notes:
  - Crosswalk: NICE to ATT&CK (68 instances)
  - Use COALESCE(wr.work_role, wr.title) for role name queries
""",
    "NICE_DCWF": """
Nodes:
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, element, ncwf_id})
  (:Ability {uid, description, text, dcwf_number, nice_mapping, source})
  (:SpecialtyArea {specialty_prefix, element_name, element_code, source})
Relationships:
  (:WorkRole)-[:REQUIRES_ABILITY]->(:Ability)
  (:WorkRole)-[:IN_SPECIALTY_AREA]->(:SpecialtyArea)
Notes:
  - Crosswalk: NICE to DCWF (embedded within NICE domain)
  - DCWF properties: Ability.dcwf_number (all nodes), WorkRole.dcwf_code (74/115 nodes)
  - Use COALESCE(wr.work_role, wr.title) for role name queries
""",
    "CVE_CWE_MITIGATION": """
Nodes:
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
  (:Weakness {uid, name, description, abstraction, source})
  (:Mitigation {uid, name, description, from_source, source, source_id})
Relationships:
  (:Vulnerability)-[:HAS_WEAKNESS]->(:Weakness)
  (:Mitigation)-[:MITIGATES]->(:Weakness)
Notes:
  - Crosswalk: CVE -> CWE -> Mitigation
  - Use 'descriptions' (plural) for CVE, 'description' (singular) for CWE and Mitigation
  - WARNING: CRITICAL: MITIGATES direction is (m:Mitigation)-[:MITIGATES]->(w:Weakness)
  - For mitigation queries: MATCH (v:Vulnerability)-[:HAS_WEAKNESS]->(w:Weakness)<-[:MITIGATES]-(m:Mitigation)
  - DO NOT use: (w:Weakness)-[:MITIGATES]->(m:Mitigation) (wrong direction)
  - CRITICAL: For specific CWE IDs (e.g., "CWE-79"), use w.uid = 'CWE-79' in MATCH: MATCH (w:Weakness {uid: 'CWE-79'})<-[:MITIGATES]-(m:Mitigation)
  - DO NOT use toLower(w.name) CONTAINS 'cwe-79' for specific CWE IDs - use uid instead
  - Use toLower(w.name) CONTAINS only for semantic searches (e.g., "sql injection", "buffer overflow")
  - Direct CWE-to-Mitigation (simpler): MATCH (w:Weakness {uid: 'CWE-79'})<-[:MITIGATES]-(m:Mitigation) (no need to go through Vulnerabilities)
  - RETURN m.uid AS uid, m.name AS title, m.description AS text (NOT v.uid or w.uid)
""",
    "NICE_CVE": """
Nodes:
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, source})
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
Relationships:
  (:WorkRole)-[:WORKS_WITH]->(:Vulnerability)
Notes:
  - Crosswalk: NICE to CVE (609,921 instances)
  - Use COALESCE(wr.work_role, wr.title) for role name queries
  - Use 'descriptions' (plural), NOT 'description' for CVE text
""",
    "NICE_CVE_ATTACK": """
Nodes:
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, source})
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
  (:AttackPattern {uid, name, description, severity, source})
  (:Technique {uid, name, description, level, x_mitre_domains, source})
  (:Tactic {uid, name, description, level, x_mitre_domains, source})
Relationships:
  (:WorkRole)-[:WORKS_WITH]->(:Vulnerability)
  (:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(:AttackPattern)
  (:AttackPattern)-[:RELATES_TO]->(:Technique)
  (:Technique)-[:USES_TACTIC]->(:Tactic)
Notes:
  - Crosswalk: NICE → CVE → AttackPattern → ATT&CK
  - Path: WorkRole → WORKS_WITH → Vulnerability → CAN_BE_EXPLOITED_BY → AttackPattern → RELATES_TO → Technique → USES_TACTIC → Tactic
  - CRITICAL: CAN_BE_EXPLOITED_BY connects Vulnerability → AttackPattern (NOT directly to Technique)
  - Use COALESCE(wr.work_role, wr.title) for role name queries
  - Use 'descriptions' (plural), NOT 'description' for CVE text
  - For tactic filtering: MATCH (wr:WorkRole)-[:WORKS_WITH]->(v:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(ap:AttackPattern)-[:RELATES_TO]->(t:Technique)-[:USES_TACTIC]->(ta:Tactic {name: 'TacticName'})
  - Tactic names are case-sensitive (e.g., 'Persistence', 'Execution', 'Initial Access')
""",
    "NICE_SKILL_CVE_ATTACK": """
Nodes:
  (:Skill {uid, title, text, source, doc_identifier, element_identifier})
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, source})
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
  (:AttackPattern {uid, name, description, severity, source})
  (:Technique {uid, name, description, level, x_mitre_domains, source})
Relationships:
  (:WorkRole)-[:REQUIRES_SKILL]->(:Skill)
  (:WorkRole)-[:WORKS_WITH]->(:Vulnerability)
  (:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(:AttackPattern)
  (:AttackPattern)-[:RELATES_TO]->(:Technique)
Notes:
  - Crosswalk: NICE Skills → WorkRole → CVE → AttackPattern → ATT&CK
  - Path: Skill <-[:REQUIRES_SKILL]- WorkRole -[:WORKS_WITH]-> Vulnerability -[:CAN_BE_EXPLOITED_BY]-> AttackPattern -[:RELATES_TO]-> Technique
  - For "top N skills" queries: Use WITH s, COUNT(DISTINCT wr) AS workrole_count ORDER BY workrole_count DESC RETURN s.uid, COALESCE(s.title, s.text) AS title, COALESCE(s.text) AS text, workrole_count
  - CRITICAL: Always include workrole_count in RETURN clause for "top N" queries so the ranking is visible
  - Use COALESCE(s.title, s.text) for skill names
  - Use COALESCE(wr.work_role, wr.title) for role name queries
  - Use 'descriptions' (plural), NOT 'description' for CVE text
  - CRITICAL: For aggregation queries, group by Skill and count distinct WorkRoles
""",
    "NICE_ABILITY_CVE_ATTACK": """
Nodes:
  (:Ability {uid, description, text, dcwf_number, nice_mapping, source})
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, source})
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
  (:AttackPattern {uid, name, description, severity, source})
  (:Technique {uid, name, description, level, x_mitre_domains, source})
Relationships:
  (:WorkRole)-[:REQUIRES_ABILITY]->(:Ability)
  (:WorkRole)-[:WORKS_WITH]->(:Vulnerability)
  (:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(:AttackPattern)
  (:AttackPattern)-[:RELATES_TO]->(:Technique)
Notes:
  - Crosswalk: NICE Abilities → WorkRole → CVE → AttackPattern → ATT&CK
  - Path: Ability <-[:REQUIRES_ABILITY]- WorkRole -[:WORKS_WITH]-> Vulnerability -[:CAN_BE_EXPLOITED_BY]-> AttackPattern -[:RELATES_TO]-> Technique
  - For "top N abilities" queries: Use WITH a, COUNT(DISTINCT wr) AS workrole_count ORDER BY workrole_count DESC RETURN a.uid, COALESCE(a.description, a.text) AS title, COALESCE(a.text) AS text, workrole_count
  - CRITICAL: Always include workrole_count in RETURN clause for "top N" queries so the ranking is visible
  - Use COALESCE(a.description, a.text) for ability names (Abilities use 'description' not 'title')
  - Use COALESCE(wr.work_role, wr.title) for role name queries
  - Use 'descriptions' (plural), NOT 'description' for CVE text
  - CRITICAL: For aggregation queries, group by Ability and count distinct WorkRoles
""",
    "NICE_KNOWLEDGE_CVE": """
Nodes:
  (:Knowledge {uid, text, source, doc_identifier, element_identifier})
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, source})
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
Relationships:
  (:WorkRole)-[:REQUIRES_KNOWLEDGE]->(:Knowledge)
  (:WorkRole)-[:WORKS_WITH]->(:Vulnerability)
Notes:
  - Crosswalk: NICE Knowledge → WorkRole → CVE (3-hop path)
  - Path: Knowledge <-[:REQUIRES_KNOWLEDGE]- WorkRole -[:WORKS_WITH]-> Vulnerability
  - For queries combining knowledge requirements with CVE: MATCH (k:Knowledge)<-[:REQUIRES_KNOWLEDGE]-(wr:WorkRole)-[:WORKS_WITH]->(v:Vulnerability) WHERE toLower(k.text) CONTAINS 'knowledge_term' RETURN wr.uid, COALESCE(wr.work_role, wr.title) AS title, COALESCE(wr.definition, wr.text) AS text
  - Use toLower(k.text) CONTAINS for semantic knowledge searches (e.g., "buffer overflow", "sql injection")
  - Use COALESCE(wr.work_role, wr.title) for role name queries
  - Use 'descriptions' (plural), NOT 'description' for CVE text
  - CRITICAL: For AND conditions (knowledge AND CVE), use the same WorkRole variable (wr) in both relationships
  - Example: "Which roles need to understand buffer overflow AND work with CVE vulnerabilities?" → MATCH (k:Knowledge)<-[:REQUIRES_KNOWLEDGE]-(wr:WorkRole)-[:WORKS_WITH]->(v:Vulnerability) WHERE toLower(k.text) CONTAINS 'buffer overflow' RETURN wr.uid, COALESCE(wr.work_role, wr.title) AS title, COALESCE(wr.definition, wr.text) AS text
""",
    "NICE_KNOWLEDGE_CVE_ATTACK": """
Nodes:
  (:Knowledge {uid, text, source, doc_identifier, element_identifier})
  (:WorkRole {uid, work_role, dcwf_code, title, text, definition, source})
  (:Vulnerability {uid, name, descriptions, cvss_v31, published, year, severity, source})
  (:AttackPattern {uid, name, description, severity, source})
  (:Technique {uid, name, description, level, x_mitre_domains, source})
Relationships:
  (:WorkRole)-[:REQUIRES_KNOWLEDGE]->(:Knowledge)
  (:WorkRole)-[:WORKS_WITH]->(:Vulnerability)
  (:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(:AttackPattern)
  (:AttackPattern)-[:RELATES_TO]->(:Technique)
Notes:
  - Crosswalk: NICE Knowledge → WorkRole → CVE → AttackPattern → ATT&CK
  - Path: Knowledge <-[:REQUIRES_KNOWLEDGE]- WorkRole -[:WORKS_WITH]-> Vulnerability -[:CAN_BE_EXPLOITED_BY]-> AttackPattern -[:RELATES_TO]-> Technique
  - For queries combining knowledge requirements with CVE/ATT&CK: MATCH (k:Knowledge)<-[:REQUIRES_KNOWLEDGE]-(wr:WorkRole)-[:WORKS_WITH]->(v:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(ap:AttackPattern)-[:RELATES_TO]->(t:Technique) WHERE toLower(k.text) CONTAINS 'knowledge_term' RETURN wr.uid, COALESCE(wr.work_role, wr.title) AS title, COALESCE(wr.definition, wr.text) AS text
  - Use toLower(k.text) CONTAINS for semantic knowledge searches (e.g., "buffer overflow", "sql injection")
  - Use COALESCE(wr.work_role, wr.title) for role name queries
  - Use 'descriptions' (plural), NOT 'description' for CVE text
  - CRITICAL: For AND conditions (knowledge AND CVE/ATT&CK), use the same WorkRole variable (wr) in both relationships
  - Example: "Which roles need to understand buffer overflow AND work with vulnerabilities that can be exploited by ATT&CK techniques?" → MATCH (k:Knowledge)<-[:REQUIRES_KNOWLEDGE]-(wr:WorkRole)-[:WORKS_WITH]->(v:Vulnerability)-[:CAN_BE_EXPLOITED_BY]->(ap:AttackPattern)-[:RELATES_TO]->(t:Technique) WHERE toLower(k.text) CONTAINS 'buffer overflow' RETURN wr.uid, COALESCE(wr.work_role, wr.title) AS title, COALESCE(wr.definition, wr.text) AS text
""",
}


# -----------------------------------------------------------------------------
# Schema assembly: combine joint + single schemas by priority
# -----------------------------------------------------------------------------


def build_curated_schema(
    datasets: List[str],
    crosswalks: Optional[List[str]] = None,
    user_query: Optional[str] = None,
) -> Optional[str]:
    """
    Build a curated schema string based on detected datasets and crosswalks.

    Algorithm: (1) Normalize dataset/crosswalk names (ATT&CK -> ATTACK, etc.).
    (2) Apply joint schemas in priority order (e.g. CVE_CWE_MITIGATION before
    CVE_CWE; NICE_ABILITY_CVE_ATTACK before NICE_SKILL_CVE_ATTACK before
    NICE_CVE_ATTACK) and remove consumed datasets from the set. (3) Add
    SINGLE_DATASET_SCHEMAS for any remaining datasets. (4) Append any
    explicitly requested crosswalks not already included. (5) Join parts with
    "---" and return, or None if nothing was added.

    Args:
        datasets: List of detected primary datasets (e.g., ["CVE", "CWE"]).
        crosswalks: Optional list of crosswalk groups (e.g., ["CVE_CWE"]).
        user_query: Optional question text; used to choose between overlapping
            joint schemas (e.g. ability vs skill vs knowledge for NICE+CVE+ATTACK).

    Returns:
        Curated schema string (sections separated by "---"), or None if no
        valid schemas found.
    """
    if not datasets:
        return None

    # Normalize dataset names: uppercase, ATT&CK -> ATTACK for dict keys
    datasets_upper = []
    for d in datasets:
        normalized = d.upper()
        # Handle ATT&CK -> ATTACK (preserve the K)
        normalized = normalized.replace("ATT&CK", "ATTACK").replace("&", "")
        datasets_upper.append(normalized)
    # Normalize crosswalk names: "↔" -> "_", ATT&CK -> ATTACK, uppercase
    crosswalks_upper = []
    for c in crosswalks or []:
        normalized = c.upper().replace("↔", "_")
        normalized = normalized.replace("ATT&CK", "ATTACK").replace("&", "")
        crosswalks_upper.append(normalized)

    # Build schema by priority: joint (crosswalk) schemas first, then single-dataset
    schema_parts = []

    datasets_set = set(datasets_upper)

    # --- Joint schema priority order (order matters: first match consumes datasets) ---
    # CVE_CWE_MITIGATION: CVE + CWE + Mitigation (mitigation queries)
    if (
        "CVE" in datasets_set
        and "CWE" in datasets_set
        and "MITIGATION" in datasets_set
        and "CVE_CWE_MITIGATION" not in crosswalks_upper
    ):
        if "CVE_CWE_MITIGATION" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["CVE_CWE_MITIGATION"])
            datasets_set.discard("CVE")
            datasets_set.discard("CWE")
            datasets_set.discard("MITIGATION")

    # CVE_CWE: CVE ↔ CWE (vulnerability–weakness)
    if (
        "CVE" in datasets_set
        and "CWE" in datasets_set
        and "CVE_CWE" not in crosswalks_upper
    ):
        if "CVE_CWE" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["CVE_CWE"])
            datasets_set.discard("CVE")
            datasets_set.discard("CWE")

    # NICE_ABILITY_CVE_ATTACK: 4-hop Ability → WorkRole → CVE → AP → Technique (check first for "ability"/"abilities")
    user_query_lower = (user_query or "").lower()
    if (
        ("NICE" in datasets_set or "DCWF" in datasets_set)
        and "CVE" in datasets_set
        and "ATTACK" in datasets_set
        and any(
            keyword in user_query_lower
            for keyword in ["ability", "abilities", "top", "most", "important"]
        )
        and "skill" not in user_query_lower
    ):
        if "NICE_ABILITY_CVE_ATTACK" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["NICE_ABILITY_CVE_ATTACK"])
            datasets_set.discard("NICE")
            datasets_set.discard("DCWF")
            datasets_set.discard("CVE")
            datasets_set.discard("ATTACK")

    # NICE_SKILL_CVE_ATTACK: 4-hop Skill → WorkRole → CVE → AP → Technique (after abilities)
    if (
        ("NICE" in datasets_set or "DCWF" in datasets_set)
        and "CVE" in datasets_set
        and "ATTACK" in datasets_set
        and any(
            keyword in user_query_lower
            for keyword in ["skill", "skills", "top", "most", "important"]
        )
    ):
        if "NICE_SKILL_CVE_ATTACK" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["NICE_SKILL_CVE_ATTACK"])
            datasets_set.discard("NICE")
            datasets_set.discard("DCWF")
            datasets_set.discard("CVE")
            datasets_set.discard("ATTACK")

    # NICE_KNOWLEDGE_CVE_ATTACK: 4-hop Knowledge → WorkRole → CVE → AP → Technique (e.g. "understand buffer overflow")
    if (
        ("NICE" in datasets_set or "DCWF" in datasets_set)
        and "CVE" in datasets_set
        and "ATTACK" in datasets_set
        and any(
            keyword in user_query_lower
            for keyword in [
                "knowledge",
                "understand",
                "know",
                "needs to understand",
                "need to understand",
            ]
        )
    ):
        if "NICE_KNOWLEDGE_CVE_ATTACK" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["NICE_KNOWLEDGE_CVE_ATTACK"])
            datasets_set.discard("NICE")
            datasets_set.discard("DCWF")
            datasets_set.discard("CVE")
            datasets_set.discard("ATTACK")

    # NICE_CVE_ATTACK: 3-hop WorkRole → CVE → AP → Technique (after ability/skill/knowledge variants)
    if (
        ("NICE" in datasets_set or "DCWF" in datasets_set)
        and "CVE" in datasets_set
        and "ATTACK" in datasets_set
    ):
        if "NICE_CVE_ATTACK" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["NICE_CVE_ATTACK"])
            datasets_set.discard("NICE")
            datasets_set.discard("DCWF")
            datasets_set.discard("CVE")
            datasets_set.discard("ATTACK")

    # CVE_ATTACK: CVE → AP → Technique (before CAPEC_ATTACK so both can be included for "full attack chain")
    if "CVE" in datasets_set and "ATTACK" in datasets_set:
        if "CVE_ATTACK" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["CVE_ATTACK"])
            datasets_set.discard("CVE")
            # Keep ATTACK so CAPEC_ATTACK can be added next if CAPEC is present

    # CAPEC_ATTACK: CAPEC → ATT&CK (after CVE_ATTACK so "full attack chain" gets both)
    if "CAPEC" in datasets_set and "ATTACK" in datasets_set:
        if "CAPEC_ATTACK" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["CAPEC_ATTACK"])
            datasets_set.discard("CAPEC")
            datasets_set.discard("ATTACK")

    # NICE_ATTACK: WorkRole → Technique
    if ("NICE" in datasets_set or "DCWF" in datasets_set) and "ATTACK" in datasets_set:
        if "NICE_ATTACK" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["NICE_ATTACK"])
            datasets_set.discard("NICE")
            datasets_set.discard("DCWF")
            datasets_set.discard("ATTACK")

    # NICE_KNOWLEDGE_CVE: Knowledge → WorkRole → CVE (before NICE_CVE for "knowledge + CVE" questions)
    if (
        ("NICE" in datasets_set or "DCWF" in datasets_set)
        and "CVE" in datasets_set
        and any(
            keyword in user_query_lower
            for keyword in [
                "knowledge",
                "understand",
                "know",
                "needs to understand",
                "need to understand",
            ]
        )
    ):
        if "NICE_KNOWLEDGE_CVE" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["NICE_KNOWLEDGE_CVE"])
            datasets_set.discard("NICE")
            datasets_set.discard("DCWF")
            datasets_set.discard("CVE")

    # NICE_CVE: WorkRole → Vulnerability
    if ("NICE" in datasets_set or "DCWF" in datasets_set) and "CVE" in datasets_set:
        if "NICE_CVE" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["NICE_CVE"])
            datasets_set.discard("NICE")
            datasets_set.discard("DCWF")
            datasets_set.discard("CVE")

    # NICE_DCWF: WorkRole + Ability + SpecialtyArea; skip if question asks for tasks/KSA (use full NICE schema instead)
    user_query_lower_ksa = (user_query or "").lower()
    ksa_question = "ksa" in user_query_lower_ksa or "ksas" in user_query_lower_ksa
    tasks_question = (
        "task" in user_query_lower_ksa
    )  # "tasks fall under", "tasks belong to", etc.
    if "NICE" in datasets_set and "DCWF" in datasets_set:
        if ksa_question or tasks_question:
            # Prefer full NICE (Task, PERFORMS, KSA) over minimal NICE_DCWF
            datasets_set.discard("DCWF")
        elif "NICE_DCWF" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["NICE_DCWF"])
            datasets_set.discard("NICE")
            datasets_set.discard("DCWF")

    # CAPEC_CWE: AttackPattern → Weakness
    if "CAPEC" in datasets_set and "CWE" in datasets_set:
        if "CAPEC_CWE" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["CAPEC_CWE"])
            datasets_set.discard("CAPEC")
            datasets_set.discard("CWE")

    # CVE_ASSET: Vulnerability → Asset (vendor/product, CPE)
    if "CVE" in datasets_set and "ASSET" in datasets_set:
        if "CVE_ASSET" in JOINT_SCHEMAS:
            schema_parts.append(JOINT_SCHEMAS["CVE_ASSET"])
            datasets_set.discard("CVE")
            datasets_set.discard("ASSET")

    # Any datasets not consumed by a joint schema get their single-dataset schema
    for dataset in datasets_set:
        if dataset in SINGLE_DATASET_SCHEMAS:
            schema_parts.append(SINGLE_DATASET_SCHEMAS[dataset])
        elif dataset == "ATT&CK":
            # Normalized key is ATTACK; alias may still appear in input
            if "ATTACK" in SINGLE_DATASET_SCHEMAS:
                schema_parts.append(SINGLE_DATASET_SCHEMAS["ATTACK"])

    # Append any crosswalks explicitly requested by the classifier (avoid duplicates)
    for crosswalk in crosswalks_upper:
        if crosswalk in JOINT_SCHEMAS and JOINT_SCHEMAS[crosswalk] not in schema_parts:
            schema_parts.append(JOINT_SCHEMAS[crosswalk])

    if not schema_parts:
        return None

    # Sections separated by "---" for readability in the prompt
    return "\n\n---\n\n".join(schema_parts)
