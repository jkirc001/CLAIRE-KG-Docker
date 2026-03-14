"""
Vocabulary and alias mapping system for CLAIRE-KG.

This module provides functionality to extract, manage, and resolve
cybersecurity terminology aliases to canonical graph entities.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from neo4j import GraphDatabase


@dataclass
class AliasMapping:
    """Represents an alias mapping to a canonical entity."""

    alias: str
    canonical_id: str
    canonical_name: str
    node_type: str
    confidence: float
    category: str
    source: str


class VocabularyExtractor:
    """Extracts vocabulary and aliases from the knowledge graph."""

    def __init__(self, db_uri: str, username: str, password: str):
        """Initialize with database connection."""
        self.driver = GraphDatabase.driver(db_uri, auth=(username, password))

    def extract_cwe_aliases(self) -> List[AliasMapping]:
        """Extract CWE aliases from alternate_terms field."""
        aliases = []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (w:Weakness) 
                WHERE w.alternate_terms IS NOT NULL 
                AND w.alternate_terms <> '' 
                AND size(w.alternate_terms) > 0
                RETURN w.uid, w.name, w.alternate_terms
            """
            )

            for record in result:
                uid = record["w.uid"]
                name = record["w.name"]
                alternate_terms = record["w.alternate_terms"]

                # Parse alternate terms array
                for term_data in alternate_terms:
                    if isinstance(term_data, str) and "Term:" in term_data:
                        # Extract term from "Term: X; Description: Y" format
                        term_match = re.search(r"Term:\s*([^;]+)", term_data)
                        if term_match:
                            alias = term_match.group(1).strip()
                            aliases.append(
                                AliasMapping(
                                    alias=alias,
                                    canonical_id=uid,
                                    canonical_name=name,
                                    node_type="Weakness",
                                    confidence=0.95,
                                    category="cwe",
                                    source="alternate_terms",
                                )
                            )

        return aliases

    def extract_capec_aliases(self) -> List[AliasMapping]:
        """Extract CAPEC aliases from alternate_terms and example_instances fields."""
        aliases = []

        with self.driver.session() as session:
            # Extract from alternate_terms
            result = session.run(
                """
                MATCH (a:AttackPattern) 
                WHERE a.alternate_terms IS NOT NULL 
                AND a.alternate_terms <> ''
                RETURN a.uid, a.name, a.alternate_terms
            """
            )

            for record in result:
                uid = record["a.uid"]
                name = record["a.name"]
                alternate_terms = record["a.alternate_terms"]

                # Parse structured string format: "::TERM:X:DESCRIPTION:::"
                term_matches = re.findall(r"TERM:([^:]+):", alternate_terms)
                for term in term_matches:
                    aliases.append(
                        AliasMapping(
                            alias=term.strip(),
                            canonical_id=uid,
                            canonical_name=name,
                            node_type="AttackPattern",
                            confidence=0.95,
                            category="capec",
                            source="alternate_terms",
                        )
                    )

            # Extract from example_instances
            result = session.run(
                """
                MATCH (a:AttackPattern) 
                WHERE a.example_instances IS NOT NULL 
                AND a.example_instances <> ''
                RETURN a.uid, a.name, a.example_instances
            """
            )

            for record in result:
                uid = record["a.uid"]
                name = record["a.name"]
                example_instances = record["a.example_instances"]

                # Extract terms from example instances
                # Look for common attack pattern names
                common_terms = [
                    "SQL Injection",
                    "Cross-site Scripting",
                    "Buffer Overflow",
                    "Directory Traversal",
                    "Path Traversal",
                    "XML Bomb",
                ]

                for term in common_terms:
                    if term.lower() in example_instances.lower():
                        aliases.append(
                            AliasMapping(
                                alias=term,
                                canonical_id=uid,
                                canonical_name=name,
                                node_type="AttackPattern",
                                confidence=0.85,
                                category="capec",
                                source="example_instances",
                            )
                        )

        return aliases

    def extract_cve_patterns(self) -> List[AliasMapping]:
        """Extract CVE vulnerability type patterns and variations."""
        aliases = []

        # Vulnerability type patterns with all their variations
        vulnerability_patterns = {
            "Cross-site Scripting": {
                "aliases": [
                    "XSS",
                    "cross-site scripting",
                    "cross site scripting",
                    "xss",
                ],
                "category": "web_security",
            },
            "SQL Injection": {
                "aliases": ["SQLi", "sql injection", "SQL injection", "sqli"],
                "category": "database_security",
            },
            "Buffer Overflow": {
                "aliases": ["BOF", "bof", "buffer overflow", "buffer overrun"],
                "category": "memory_corruption",
            },
            "Remote Code Execution": {
                "aliases": ["RCE", "rce", "remote code execution", "code execution"],
                "category": "code_execution",
            },
            "Local File Inclusion": {
                "aliases": ["LFI", "lfi", "local file inclusion", "file inclusion"],
                "category": "file_system",
            },
            "Remote File Inclusion": {
                "aliases": ["RFI", "rfi", "remote file inclusion"],
                "category": "file_system",
            },
            "Cross-Site Request Forgery": {
                "aliases": ["CSRF", "csrf", "cross-site request forgery"],
                "category": "web_security",
            },
            "XML External Entity": {
                "aliases": ["XXE", "xxe", "xml external entity"],
                "category": "web_security",
            },
            "Server-Side Request Forgery": {
                "aliases": ["SSRF", "ssrf", "server-side request forgery"],
                "category": "web_security",
            },
            "Deserialization": {
                "aliases": [
                    "deserialization",
                    "unsafe deserialization",
                    "insecure deserialization",
                ],
                "category": "application_security",
            },
            "Path Traversal": {
                "aliases": ["path traversal", "directory traversal", "file traversal"],
                "category": "file_system",
            },
            "Clickjacking": {
                "aliases": [
                    "clickjacking",
                    "UI Redress",
                    "ui redress",
                    "UI redressing",
                ],
                "category": "web_security",
            },
            "LDAP Injection": {
                "aliases": ["LDAP Injection", "ldap injection", "LDAP", "ldap"],
                "category": "database_security",
            },
            "XML Injection": {
                "aliases": ["XML Injection", "xml injection", "XML", "xml"],
                "category": "web_security",
            },
            "Command Injection": {
                "aliases": [
                    "Command Injection",
                    "command injection",
                    "OS Command Injection",
                    "os command injection",
                ],
                "category": "code_execution",
            },
            "Privilege Escalation": {
                "aliases": [
                    "Privilege Escalation",
                    "privilege escalation",
                    "Priv-Esc",
                    "priv-esc",
                ],
                "category": "access_control",
            },
            "Information Disclosure": {
                "aliases": [
                    "Information Disclosure",
                    "information disclosure",
                    "Info Disclosure",
                    "info disclosure",
                    "data leak",
                ],
                "category": "information_leakage",
            },
            "Authentication Bypass": {
                "aliases": [
                    "auth bypass",
                    "authentication bypass",
                    "auth bypass",
                    "bypass authentication",
                ],
                "category": "access_control",
            },
            "Session Fixation": {
                "aliases": ["session fixation", "session fix", "session hijacking"],
                "category": "session_management",
            },
            "Race Condition": {
                "aliases": [
                    "race condition",
                    "race",
                    "TOCTOU",
                    "time-of-check time-of-use",
                ],
                "category": "concurrency",
            },
        }

        # Create aliases for each vulnerability type and its variations
        for vuln_type, data in vulnerability_patterns.items():
            aliases_list = data["aliases"]
            category = data["category"]

            for alias in aliases_list:
                aliases.append(
                    AliasMapping(
                        alias=alias,
                        canonical_id=f"CVE-PATTERN-{vuln_type.replace(' ', '-').upper()}",
                        canonical_name=vuln_type,
                        node_type="VulnerabilityPattern",
                        confidence=0.95,
                        category=category,
                        source="vulnerability_pattern",
                    )
                )

        return aliases

    def extract_all_aliases(self) -> Dict[str, List[AliasMapping]]:
        """Extract all aliases from all datasets."""
        return {
            "cwe": self.extract_cwe_aliases(),
            "capec": self.extract_capec_aliases(),
            "cve": self.extract_cve_patterns(),
        }

    def save_aliases(self, aliases: Dict[str, List[AliasMapping]], output_dir: str):
        """Save aliases to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for dataset, alias_list in aliases.items():
            # Convert to dictionary format
            aliases_dict = {}
            for alias in alias_list:
                aliases_dict[alias.alias] = {
                    "canonical_id": alias.canonical_id,
                    "canonical_name": alias.canonical_name,
                    "node_type": alias.node_type,
                    "confidence": alias.confidence,
                    "category": alias.category,
                    "source": alias.source,
                }

            # Save to file
            output_file = output_path / f"extracted_aliases_{dataset}.json"
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "dataset": dataset.upper(),
                        "description": f"Extracted aliases from {dataset.upper()} dataset",
                        "version": "1.0",
                        "created": "2025-01-27",
                        "aliases": aliases_dict,
                    },
                    f,
                    indent=2,
                )

            print(f"Saved {len(alias_list)} aliases to {output_file}")


class AliasResolver:
    """Resolves aliases to canonical entities."""

    def __init__(self, alias_files: List[str]):
        """Initialize with alias dictionary files."""
        self.aliases = {}
        self.load_aliases(alias_files)

    def load_aliases(self, alias_files: List[str]):
        """Load aliases from JSON files."""
        for file_path in alias_files:
            with open(file_path, "r") as f:
                data = json.load(f)
                if "aliases" in data:
                    self.aliases.update(data["aliases"])

    def resolve_alias(self, alias: str) -> List[Dict[str, Any]]:
        """Resolve an alias to canonical entities."""
        alias_lower = alias.lower()
        results = []

        # Exact match
        if alias_lower in self.aliases:
            results.append(self.aliases[alias_lower])

        # Fuzzy match
        for key, value in self.aliases.items():
            if alias_lower in key.lower() or key.lower() in alias_lower:
                if key not in [r.get("alias", "") for r in results]:
                    results.append(value)

        return results

    def enhance_query(self, query: str) -> str:
        """Enhance a query by replacing aliases with canonical terms."""
        enhanced_query = query

        # Find aliases in the query
        words = query.split()
        for word in words:
            word_clean = re.sub(r"[^\w-]", "", word.lower())
            if word_clean in self.aliases:
                alias_data = self.aliases[word_clean]
                # Replace with canonical name
                enhanced_query = enhanced_query.replace(
                    word, alias_data.get("canonical_name", word)
                )

        return enhanced_query


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="CLAIRE-KG Vocabulary System")
    parser.add_argument(
        "--extract", action="store_true", help="Extract aliases from database"
    )
    parser.add_argument("--resolve", type=str, help="Resolve an alias")
    parser.add_argument("--enhance", type=str, help="Enhance a query")
    parser.add_argument(
        "--output-dir",
        default="docs/vocab",
        help="Output directory for extracted aliases",
    )

    args = parser.parse_args()

    if args.extract:
        # Extract aliases from database
        extractor = VocabularyExtractor(
            db_uri="bolt://localhost:7687",
            username="neo4j",
            password="graphrag-password",
        )

        aliases = extractor.extract_all_aliases()
        extractor.save_aliases(aliases, args.output_dir)

        # Print summary
        total_aliases = sum(len(alias_list) for alias_list in aliases.values())
        print(f"\nExtracted {total_aliases} total aliases:")
        for dataset, alias_list in aliases.items():
            print(f"  {dataset.upper()}: {len(alias_list)} aliases")

    elif args.resolve:
        # Resolve an alias
        resolver = AliasResolver(
            [
                "docs/vocab/aliases-cwe.json",
                "docs/vocab/aliases-capec.json",
                "docs/vocab/aliases-cve.json",
            ]
        )

        results = resolver.resolve_alias(args.resolve)
        print(f"Alias '{args.resolve}' resolves to:")
        for result in results:
            print(f"  {result.get('canonical_id')}: {result.get('canonical_name')}")

    elif args.enhance:
        # Enhance a query
        resolver = AliasResolver(
            [
                "docs/vocab/aliases-cwe.json",
                "docs/vocab/aliases-capec.json",
                "docs/vocab/aliases-cve.json",
            ]
        )

        enhanced = resolver.enhance_query(args.enhance)
        print(f"Original: {args.enhance}")
        print(f"Enhanced: {enhanced}")


if __name__ == "__main__":
    main()
