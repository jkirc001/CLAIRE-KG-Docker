#!/usr/bin/env python3
"""
Query orchestrator: generate Cypher from natural language and execute on Neo4j.

Thin wrapper around CypherGenerator and Neo4jConnection for a single pipeline:
schema selection (QuestionClassifier) → generate Cypher → validate → execute →
optional relaxed retry on zero results → display results. Uses OpenAI SDK inside
CypherGenerator for LLM calls.

This module is an alternative to the full LLMOrchestrator (llm_orchestrator.py),
which adds Phase 2 (answer enhancement with citations) and Phase 3 (evaluation).
Use QueryOrchestrator when you only need Cypher generation + execution + rich
display, without answer enhancement or DeepEval.

Entry point: QueryOrchestrator.process_query(query, limit, verbose, debug).
"""

import re
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .database import Neo4jConnection
from .cypher_generator import CypherGenerator, CypherQueryResult
from .question_classifier import QuestionClassifier

# -----------------------------------------------------------------------------
# QueryOrchestrator: Cypher generation + execution + display
# -----------------------------------------------------------------------------


class QueryOrchestrator:
    """Orchestrator for Cypher generation and execution (no answer enhancement).

    Uses CypherGenerator (OpenAI SDK) for schema selection and Cypher generation,
    then executes on Neo4j and displays results. Supports relaxed retry on zero
    results and helpful messages for known patterns (e.g. "both" mitigation).
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """Initialize Neo4j connection, CypherGenerator, and schema selector."""
        self.console = Console()
        self.neo4j_connection = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        self.cypher_generator = CypherGenerator(neo4j_uri)
        # Schema selection (QuestionClassifier) for metadata used by generate_cypher
        try:
            self.classifier = QuestionClassifier(use_metadata=True)
        except Exception:
            self.classifier = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            self.neo4j_connection._connect()
            self.connected = True
            return True
        except Exception as e:
            self.console.print(f"[red]ERROR: Failed to connect to Neo4j: {e}[/red]")
            return False

    def close(self):
        """Close the database connection."""
        if self.connected:
            self.neo4j_connection.close()
            self.connected = False

    def process_query(
        self, query: str, limit: int = 10, verbose: bool = False, debug: bool = False
    ) -> Dict[str, Any]:
        """Process a natural language query using LLM-generated Cypher (OpenAI SDK directly)."""
        start_time = time.time()

        if verbose or debug:
            self.console.print(f"[bold blue]🧠 LLM Query Processing[/bold blue]")
            self.console.print(f"[dim blue]Input: {query}[/dim blue]")

        try:
            # Step 1: Schema selection + Cypher generation (LLM)
            if debug:
                self.console.print(
                    f"[bold yellow] DEBUG - Generating Cypher Query:[/bold yellow]"
                )
                self.console.print(f"[dim yellow]  • Query: '{query}'[/dim yellow]")
                self.console.print(f"[dim yellow]  • Limit: {limit}[/dim yellow]")

            # Schema selection: get metadata for CypherGenerator (primary_datasets, intent, crosswalk)
            classification_metadata = None
            if self.classifier:
                try:
                    classification_result = self.classifier.classify(query)
                    # Dict format expected by generate_cypher
                    classification_metadata = {
                        "primary_datasets": classification_result.primary_datasets
                        or [],
                        "intent_types": classification_result.intent_types or [],
                        "crosswalk_groups": classification_result.crosswalk_groups
                        or [],
                    }
                except Exception:
                    # If schema selection fails, continue without metadata
                    pass

            cypher_result = self.cypher_generator.generate_cypher(
                query, limit, classification_metadata=classification_metadata
            )

            if debug:
                self.console.print(
                    f"[bold yellow] DEBUG - Generated Cypher:[/bold yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Query: {cypher_result.query}[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Parameters: {cypher_result.parameters}[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Reasoning: {cypher_result.reasoning}[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Cost: ${cypher_result.cost:.6f}[/dim yellow]"
                )

            # Step 1.5: Validate query structure (advisory; execution continues either way)
            is_valid, issues, suggestions = (
                self.cypher_generator._validate_query_structure(
                    cypher_result.query, query
                )
            )
            if not is_valid and debug:
                self.console.print(
                    f"[bold yellow]⚠️  Query validation issues detected:[/bold yellow]"
                )
                for issue, suggestion in zip(issues, suggestions):
                    self.console.print(f"[dim yellow]  • {issue}[/dim yellow]")
                    self.console.print(f"[dim yellow]    → {suggestion}[/dim yellow]")
                # _preflight_fix_cypher is applied during generation; we still run the query

            # Step 2: Execute Cypher on Neo4j
            if debug:
                self.console.print(
                    f"[bold yellow] DEBUG - Executing Query:[/bold yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Executing Cypher query...[/dim yellow]"
                )

            # Execute query with error handling for automatic fixes
            try:
                results = self.neo4j_connection.execute_cypher(
                    cypher_result.query, cypher_result.parameters
                )
            except Exception as e:
                error_msg = str(e)
                # Auto-fix: "Variable X not defined" — retry with preflight-fixed query
                if "Variable" in error_msg and "not defined" in error_msg:
                    import re

                    var_match = re.search(r"Variable `(\w+)` not defined", error_msg)
                    if var_match:
                        undefined_var = var_match.group(1)
                        # Retry with preflight-fixed query (variable scope/typos)
                        fixed_query = self.cypher_generator._preflight_fix_cypher(
                            cypher_result.query, query
                        )
                        if fixed_query != cypher_result.query:
                            # Retry with fixed query
                            if debug:
                                self.console.print(
                                    f"[bold yellow]🔄 Retrying with auto-fixed query...[/bold yellow]"
                                )
                            results = self.neo4j_connection.execute_cypher(
                                fixed_query, cypher_result.parameters
                            )
                            cypher_result.query = fixed_query
                        else:
                            # Couldn't auto-fix, re-raise
                            raise
                    else:
                        raise
                else:
                    raise

            # Ensure results is a list (handle None or other types)
            if results is None:
                results = []

            # Step 2.5: On zero results, optionally retry with relaxed query (broader search)
            if len(results) == 0:
                if debug:
                    self.console.print(
                        f"[bold yellow]🔄 No results found, retrying with relaxed query...[/bold yellow]"
                    )

                relaxed_result = self._retry_with_relaxed_query(
                    query, limit, cypher_result
                )
                if relaxed_result and len(relaxed_result.get("results", [])) > 0:
                    results = relaxed_result["results"]
                    cypher_result.query = relaxed_result["cypher_query"]
                    cypher_result.reasoning = relaxed_result.get(
                        "reasoning", cypher_result.reasoning
                    )
                    if debug:
                        self.console.print(
                            f"[bold green]OK: Relaxed query found {len(results)} results[/bold green]"
                        )

            execution_time = time.time() - start_time

            if debug:
                self.console.print(
                    f"[bold yellow] DEBUG - Query Results:[/bold yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Results: {len(results)} records[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Execution Time: {execution_time:.3f}s[/dim yellow]"
                )

            # Step 3: Display results
            if verbose or debug:
                self.console.print(
                    f"[bold green]OK: Query processed successfully[/bold green]"
                )
                self.console.print(
                    f"[green]Processing time: {execution_time:.3f}s[/green]"
                )

            # Helpful message for "both" mitigation questions: KG has no single node with both CWE and CAPEC links
            error_message = None
            if len(results) == 0:
                if query and "both" in query.lower() and "mitigation" in query.lower():
                    # Detect dual MATCH (CWE + CAPEC) or marker in generated Cypher
                    if "BOTH_MITIGATION_QUERY" in cypher_result.query or re.search(
                        r"MATCH.*Mitigation.*MITIGATES.*Weakness.*MATCH.*Mitigation.*MITIGATES.*AttackPattern",
                        cypher_result.query,
                        re.IGNORECASE | re.DOTALL,
                    ):
                        error_message = (
                            "⚠️  No mitigations found that address both entities.\n\n"
                            "**Why?** The knowledge graph stores mitigations separately by source:\n"
                            "• CWE mitigations are linked only to Weaknesses\n"
                            "• CAPEC mitigations are linked only to AttackPatterns\n"
                            "• No single mitigation node has both relationships\n\n"
                            "**Suggestion**: Try asking for mitigations that address either entity:\n"
                            '• "What mitigations address CWE-XXX or CAPEC-YYY?"\n'
                            '• Or ask separately: "What mitigations address CWE-XXX?" and "What mitigations address CAPEC-YYY?"'
                        )

            # Build result for caller and for _display_results
            result_summary = {
                "query": query,
                "cypher_query": cypher_result.query,
                "parameters": cypher_result.parameters,
                "results": results,
                "total_count": len(results),
                "execution_time": execution_time,
                "reasoning": cypher_result.reasoning,
                "cost": cypher_result.cost,
                "confidence": cypher_result.confidence,
                "success": True,
                "error_message": error_message,
            }

            # Display results
            self._display_results(result_summary, verbose)

            return result_summary

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error processing query: {e}"

            if debug:
                self.console.print(f"[bold red] DEBUG - Error:[/bold red]")
                self.console.print(f"[dim red]  • Error: {error_msg}[/dim red]")
                self.console.print(
                    f"[dim red]  • Execution Time: {execution_time:.3f}s[/dim red]"
                )

            return {
                "query": query,
                "cypher_query": "",
                "parameters": {},
                "results": [],
                "total_count": 0,
                "execution_time": execution_time,
                "reasoning": "Error occurred",
                "cost": 0.0,
                "confidence": 0.0,
                "success": False,
                "error_message": error_msg,
            }

    def _display_results(self, result: Dict[str, Any], verbose: bool = False):
        """Display query results: Panel for analysis, then table or verbose panels per row."""
        if not result["results"]:
            # Check if there's a helpful error message (e.g., for "both" mitigation queries)
            if result.get("error_message"):
                self.console.print(
                    Panel(
                        result["error_message"],
                        title="⚠️  No Results Found",
                        border_style="yellow",
                    )
                )
            else:
                self.console.print("[yellow]No results found[/yellow]")
            return

        # Show query analysis
        analysis_panel = Panel(
            f"[bold]Generated Cypher:[/bold] {result['cypher_query']}\n"
            f"[bold]Reasoning:[/bold] {result['reasoning']}\n"
            f"[bold]Confidence:[/bold] {result['confidence']:.2f}\n"
            f"[bold]Cost:[/bold] ${result['cost']:.6f}\n"
            f"[bold]Execution Time:[/bold] {result['execution_time']:.3f}s",
            title="Query Analysis",
            border_style="blue",
        )
        self.console.print(analysis_panel)

        if verbose:
            # Verbose mode: Show detailed information for each result
            self.console.print(
                f"[bold green] Detailed Results ({result['total_count']} found)[/bold green]"
            )

            for i, res in enumerate(result["results"][:10], 1):
                # CWE count queries return w.uid + vuln_count; others use uid/title/definition
                if "w.uid" in res and "vuln_count" in res:
                    result_panel = Panel(
                        f"[bold cyan]CWE ID:[/bold cyan] {res.get('w.uid', 'N/A')}\n"
                        f"[bold cyan]Name:[/bold cyan] {res.get('w.name', 'N/A')}\n"
                        f"[bold cyan]Vulnerability Count:[/bold cyan] {res.get('vuln_count', 'N/A')}",
                        title=f"CWE {i}",
                        border_style="green",
                    )
                else:
                    # Standard result
                    result_panel = Panel(
                        f"[bold cyan]ID:[/bold cyan] {res.get('uid', res.get('wr.uid', res.get('n.uid', 'N/A')))}\n"
                        f"[bold cyan]Title:[/bold cyan] {res.get('work_role', res.get('wr.work_role', res.get('title', res.get('n.title', 'N/A'))))}\n"
                        f"[bold cyan]Name:[/bold cyan] {res.get('name', res.get('n.name', res.get('title', res.get('n.title', 'N/A'))))}\n"
                        f"[bold cyan]Description:[/bold cyan] {res.get('definition', res.get('wr.definition', res.get('text', res.get('n.text', res.get('description', 'N/A')))))}",
                        title=f"Result {i}",
                        border_style="green",
                    )
                self.console.print(result_panel)
        else:
            # Non-verbose: single table (top 10)
            table = Table(title=f"LLM Query Results ({result['total_count']} found)")
            table.add_column("Rank", style="cyan", no_wrap=True)
            table.add_column("ID", style="green", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Description", style="yellow")

            for i, res in enumerate(result["results"][:10], 1):
                # CWE count vs standard result (same key detection as verbose branch)
                if "w.uid" in res and "vuln_count" in res:
                    name = res.get("w.name") or "N/A"
                    if name and len(name) > 60:
                        name = name[:57] + "..."
                    table.add_row(
                        str(i),
                        res.get("w.uid", "N/A"),
                        name,
                        f"Count: {res.get('vuln_count', 'N/A')}",
                    )
                else:
                    # Standard result
                    description = (
                        res.get(
                            "definition",
                            res.get(
                                "wr.definition",
                                res.get(
                                    "text",
                                    res.get(
                                        "n.text",
                                        res.get(
                                            "description", "No description available"
                                        ),
                                    ),
                                ),
                            ),
                        )
                        or "No description available"
                    )
                    if description and len(description) > 60:
                        description = description[:57] + "..."

                    table.add_row(
                        str(i),
                        res.get("uid", res.get("wr.uid", res.get("n.uid", "N/A"))),
                        res.get(
                            "work_role",
                            res.get(
                                "wr.work_role",
                                res.get("title", res.get("n.title", "N/A")),
                            ),
                        ),
                        description,
                    )

            self.console.print(table)

    def _retry_with_relaxed_query(
        self, query: str, limit: int, original_result
    ) -> Dict[str, Any]:
        """Retry query with a more relaxed/broader approach when original returns no results.

        Strategies:
        - Simplify query by removing restrictive WHERE clauses
        - Use keyword search instead of exact property matches
        - Remove filters that might be too specific
        - Broaden the search scope
        """
        import re

        try:
            # Generate a relaxed query by modifying the original query text
            # Add keywords that trigger broader search patterns
            relaxed_query = query

            # If query is very specific, try a simpler version
            # Remove count/number requirements that might be too restrictive
            relaxed_query = re.sub(
                r"\b\d+\s+(specialty areas?|work roles?|tasks?|vulnerabilities?)\b",
                r"\1",
                relaxed_query,
                flags=re.IGNORECASE,
            )

            # Generate new Cypher with relaxed query
            relaxed_cypher_result = self.cypher_generator.generate_cypher(
                relaxed_query, limit * 2
            )  # Higher limit for broader results

            # Execute relaxed query
            relaxed_results = self.neo4j_connection.execute_cypher(
                relaxed_cypher_result.query, relaxed_cypher_result.parameters
            )

            # Ensure relaxed_results is a list (handle None)
            if relaxed_results is None:
                relaxed_results = []

            if len(relaxed_results) > 0:
                return {
                    "results": relaxed_results[
                        :limit
                    ],  # Limit to original requested amount
                    "cypher_query": relaxed_cypher_result.query,
                    "reasoning": f"Relaxed query (original returned 0 results): {relaxed_cypher_result.reasoning}",
                }
        except Exception as e:
            # If retry fails, return None (will use original empty results)
            pass

        return None

    def get_cost_stats(self) -> Dict[str, Any]:
        """Return CypherGenerator cost and performance statistics."""
        return self.cypher_generator.get_cost_stats()
