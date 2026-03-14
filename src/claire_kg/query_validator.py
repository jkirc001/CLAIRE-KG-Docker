"""
Schema-aware query validation: check Cypher results match question intent.

Compares expected entity types (from the natural language question) with
actual types returned by the query. Used by CypherGenerator and LLMOrchestrator
for advisory validation (e.g. before/after execution) and by evaluators to
detect wrong-entity-type failures.

Flow: extract_expected_types(question) → extract_actual_types_from_query(cypher)
and infer_types_from_results(results) → combine actual types → validate()
returns ValidationResult (is_valid, expected/actual sets, mismatch_reason, confidence).

Entry point: QueryValidator.validate(question, cypher_query, results).
"""

import re
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# ValidationResult and QueryValidator
# -----------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of schema-aware validation: expected vs actual entity types."""

    is_valid: bool
    expected_types: Set[str]
    actual_types: Set[str]
    mismatch_reason: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0


class QueryValidator:
    """
    Validates that Cypher queries return the correct entity types for the question.

    Uses ENTITY_TYPE_PATTERNS (question keywords → labels) and UID_PATTERNS
    (UID format → labels) to infer expected vs actual types; combines query
    RETURN analysis with result UID inspection for actual types.
    """

    # Question keywords/phrases → expected node labels (word-boundary matched)
    ENTITY_TYPE_PATTERNS = {
        # NICE Framework
        "knowledge statement": {"Knowledge"},
        "knowledge statements": {"Knowledge"},
        "required knowledge": {"Knowledge"},
        "skills": {"Skill"},
        "skill": {"Skill"},
        "work role": {"WorkRole"},
        "work roles": {"WorkRole"},
        "tasks": {"Task"},
        "task": {"Task"},
        "specialty area": {"SpecialtyArea"},
        "specialty areas": {"SpecialtyArea"},
        "abilities": {"Ability"},
        "ability": {"Ability"},
        # CVE/Vulnerabilities
        "vulnerabilit": {"Vulnerability"},
        "cve": {"Vulnerability"},
        "cves": {"Vulnerability"},
        # CWE/Weaknesses
        "weakness": {"Weakness"},
        "weaknesses": {"Weakness"},
        "cwe": {"Weakness"},
        "cwes": {"Weakness"},
        # CAPEC/Attack Patterns
        "attack pattern": {"AttackPattern"},
        "attack patterns": {"AttackPattern"},
        "capec": {"AttackPattern"},
        # ATT&CK
        "technique": {"Technique"},
        "techniques": {"Technique"},
        "tactic": {"Tactic"},
        "tactics": {"Tactic"},
        "sub-technique": {"SubTechnique"},
        "sub-techniques": {"SubTechnique"},
        # Mitigations
        "mitigation": {"Mitigation"},
        "mitigations": {"Mitigation"},
        # DCWF
        "dcwf": {"WorkRole", "Task"},  # Could be either
        # Assets
        "asset": {"Asset"},
        "assets": {"Asset"},
        "cpe": {"Asset"},
    }

    # UID regex → node label (order matters: more specific first, e.g. CWE mitigation before CWE)
    UID_PATTERNS = {
        # More specific patterns first (to avoid false matches)
        r"^CWE-\d+_mitigation_": "Mitigation",  # Mitigation UIDs: CWE-564_mitigation_0.xxx
        r"^CAPEC-\d+_mitigation_": "Mitigation",  # CAPEC mitigation UIDs
        r"^M\d+": "Mitigation",  # Generic Mitigation pattern
        r"^K\d+$": "Knowledge",
        r"^S\d+$": "Skill",
        # ATT&CK Technique (T + 4 digits) before Task (T + digits) so T1027 -> Technique not Task
        r"^T\d{4}$": "Technique",
        r"^T\d+$": "Task",
        r"^WRL-\d+$": "WorkRole",
        r"^WRL\d+$": "WorkRole",
        r"^IO-WRL-": "WorkRole",
        r"^CVE-": "Vulnerability",
        r"^CWE-": "Weakness",  # Check CWE after mitigation patterns
        r"^CAPEC-": "AttackPattern",
        r"^T\d+": "Technique",  # ATT&CK techniques (fallback for T12345-style)
        r"^TA\d+": "Tactic",
    }

    def __init__(self):
        """Initialize the query validator."""
        pass

    def extract_expected_types(self, question: str) -> Set[str]:
        """
        Extract expected entity types from a natural language question.

        Args:
            question: Natural language question

        Returns:
            Set of expected node label names
        """
        question_lower = question.lower()
        expected_types: Set[str] = set()

        # Word-boundary match so e.g. "abilities" doesn't match inside "vulnerabilities"
        for pattern, types in self.ENTITY_TYPE_PATTERNS.items():
            # Use word boundary matching to avoid substring false positives
            # Pattern like "abilities" should match "abilities" but not "vulnerabilities"
            pattern_re = r"\b" + re.escape(pattern) + r"\b"
            if re.search(pattern_re, question_lower, re.IGNORECASE):
                expected_types.update(types)

        # Check for explicit UIDs in question
        for uid_pattern, node_type in self.UID_PATTERNS.items():
            if re.search(uid_pattern, question, re.IGNORECASE):
                expected_types.add(node_type)

        return expected_types

    def extract_actual_types_from_query(self, cypher_query: str) -> Set[str]:
        """
        Extract actual node types being returned from a Cypher query.

        Args:
            cypher_query: Generated Cypher query

        Returns:
            Set of node label names that are being returned
        """
        actual_types: Set[str] = set()

        # Find all RETURN clauses (handles UNION); lookahead for LIMIT to avoid clipping "title" etc.
        return_clauses = re.findall(
            r"RETURN\s+(.+?)(?=\s+LIMIT\s+\d+|$)",
            cypher_query,
            re.IGNORECASE | re.DOTALL,
        )

        if not return_clauses:
            return actual_types

        # For each RETURN clause, find what's being returned
        for return_clause in return_clauses:
            # Look for variable references (e.g., k.uid, wr.title)
            # Pattern: identifier.property AS alias
            variable_refs = re.findall(
                r"(\w+)\.(?:uid|title|name|description|text|work_role|definition)",
                return_clause,
                re.IGNORECASE,
            )

            # Also look for explicit node type matching in MATCH clauses
            # Find all MATCH clauses to map variables to labels
            match_clauses = re.findall(
                r"MATCH\s+\((\w+)(?::(\w+))?\)", cypher_query, re.IGNORECASE
            )

            # MATCH (var:Label) → var_to_label; unlabeled nodes get "Unknown"
            var_to_label: Dict[str, str] = {}
            for var, label in match_clauses:
                if label:
                    var_to_label[var] = label
                else:
                    var_to_label[var] = "Unknown"

            # Map RETURN variables to labels; fallback heuristics for Unknown (k→Knowledge, wr→WorkRole, etc.)
            for var in variable_refs:
                if var in var_to_label:
                    label = var_to_label[var]
                    # Map common variable names to likely node types
                    if label != "Unknown":
                        actual_types.add(label)
                    elif var.startswith("k") or var == "knowledge":
                        actual_types.add("Knowledge")
                    elif var.startswith("wr") or "workrole" in var.lower():
                        actual_types.add("WorkRole")
                    elif var.startswith("v") or "vuln" in var.lower():
                        actual_types.add("Vulnerability")
                    elif var.startswith("w") and not var.startswith("wr"):
                        actual_types.add("Weakness")
                    elif var.startswith("ap") or "attackpattern" in var.lower():
                        actual_types.add("AttackPattern")
                    elif var.startswith("t") and "tech" in var.lower():
                        actual_types.add("Technique")

        return actual_types

    def infer_types_from_results(self, results: List[Dict[str, Any]]) -> Set[str]:
        """
        Infer entity types from actual query results by examining UID patterns.

        Args:
            results: Query results

        Returns:
            Set of inferred node types based on UID patterns
        """
        if not results:
            return set()

        inferred_types: Set[str] = set()

        for result in results:
            uid = str(result.get("uid", ""))
            if not uid:
                continue

            # Match UID against patterns
            for pattern, node_type in self.UID_PATTERNS.items():
                if re.match(pattern, uid, re.IGNORECASE):
                    inferred_types.add(node_type)
                    break

        return inferred_types

    def validate(
        self,
        question: str,
        cypher_query: str,
        results: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate that query results match expected entity types from question.

        Args:
            question: Original natural language question
            cypher_query: Generated Cypher query
            results: Query execution results

        Returns:
            ValidationResult with validation status and details
        """
        # Extract expected types from question
        expected_types = self.extract_expected_types(question)

        # If we can't determine expected types, skip validation
        if not expected_types:
            return ValidationResult(
                is_valid=True,
                expected_types=expected_types,
                actual_types=set(),
                confidence=0.5,  # Low confidence - couldn't determine expectation
            )

        # Extract actual types from query
        actual_types_from_query = self.extract_actual_types_from_query(cypher_query)

        # Infer types from results (most reliable)
        actual_types_from_results = self.infer_types_from_results(results)

        # Combine actual types: prefer result UIDs; if results disagree with query but query matches expected, trust query
        if actual_types_from_results and actual_types_from_query:
            if expected_types:
                results_match_expected = bool(
                    expected_types & actual_types_from_results
                )
                query_match_expected = bool(expected_types & actual_types_from_query)
                if not results_match_expected and query_match_expected:
                    actual_types = actual_types_from_query
                else:
                    actual_types = actual_types_from_results
            else:
                actual_types = actual_types_from_results
        else:
            actual_types = actual_types_from_results or actual_types_from_query

        if not actual_types:
            # Can't determine actual types - mark as uncertain
            return ValidationResult(
                is_valid=True,  # Don't fail on uncertainty
                expected_types=expected_types,
                actual_types=actual_types,
                confidence=0.5,
            )

        overlap = expected_types & actual_types

        # Primary expected type = highest priority among expected (used for pass/fail and confidence)
        type_priority = {
            "Knowledge": 10,
            "Skill": 9,
            "Task": 8,
            "WorkRole": 7,
            "Ability": 6,
            "AttackPattern": 5,
            "Technique": 4,
            "Vulnerability": 3,
            "Weakness": 2,
        }

        # Only determine primary expected type if we have expected types
        if expected_types:
            primary_expected = max(
                expected_types, key=lambda t: type_priority.get(t, 0)
            )
            has_primary_type = primary_expected in actual_types
        else:
            # No expected types detected - can't validate, consider it valid
            has_primary_type = True  # Don't fail if we can't determine expected types

        if not expected_types:
            # Can't determine expected types from question - assume valid but with low confidence
            return ValidationResult(
                is_valid=True,
                expected_types=expected_types,
                actual_types=actual_types,
                confidence=0.5,  # Low confidence when we can't determine what's expected
            )
        elif has_primary_type:
            # Primary expected type is present - valid
            return ValidationResult(
                is_valid=True,
                expected_types=expected_types,
                actual_types=actual_types,
                confidence=1.0 if primary_expected in actual_types else 0.9,
            )
        elif overlap:
            # Partial match: some expected types present but not primary — still valid, lower confidence
            return ValidationResult(
                is_valid=True,
                expected_types=expected_types,
                actual_types=actual_types,
                confidence=0.6,  # Lower confidence for partial matches
            )

        # No overlap: question asks for X, query returns Y — invalid
        expected_str = ", ".join(sorted(expected_types))
        actual_str = ", ".join(sorted(actual_types))

        return ValidationResult(
            is_valid=False,
            expected_types=expected_types,
            actual_types=actual_types,
            mismatch_reason=(
                f"Expected {expected_str} but query returns {actual_str}. "
                f"Query appears to return wrong entity types."
            ),
            confidence=1.0,
        )
