"""
Main Orchestrator for CLAIRE-KG

Coordinates intent classification, entity linking, and graph retrieval
to provide intelligent query processing.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .intent_classifier import IntentClassifier, QueryIntent, IntentResult
from .llm_intent_classifier import LLMIntentClassifier, ClarificationResult
from .entity_linker import EntityLinker, LinkedEntity
from .graph_retrieval import GraphRetrievalSystem, RetrievalResult, ContextBundle


@dataclass
class OrchestrationResult:
    """Complete result of orchestration process."""

    query: str
    intent_result: IntentResult
    linked_entities: List[LinkedEntity]
    retrieval_result: RetrievalResult
    context_bundle: ContextBundle
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class CLAIREOrchestrator:
    """Main orchestrator for CLAIRE-KG intelligent query processing."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """Initialize the orchestrator with all components."""
        self.console = Console()

        # Initialize components
        self.intent_classifier = IntentClassifier()
        self.llm_intent_classifier = LLMIntentClassifier()
        self.entity_linker = EntityLinker()
        self.graph_retrieval = GraphRetrievalSystem(
            neo4j_uri, neo4j_user, neo4j_password
        )

        # Connection status
        self.connected = False

    def connect(self) -> bool:
        """Connect to Neo4j database."""
        self.connected = self.graph_retrieval.connect()
        if self.connected:
            self.console.print("[green]OK: Connected to Neo4j database[/green]")
        else:
            self.console.print("[red]ERROR: Failed to connect to Neo4j database[/red]")
        return self.connected

    def close(self):
        """Close database connection."""
        if self.connected:
            self.graph_retrieval.close()
            self.connected = False

    def process_query(
        self, query: str, limit: int = 10, verbose: bool = False, debug: bool = False
    ) -> OrchestrationResult:
        """
        Process a user query through the complete pipeline.

        Args:
            query: User query string
            limit: Maximum number of results
            verbose: Enable verbose output

        Returns:
            OrchestrationResult with complete processing information
        """
        start_time = datetime.now()

        if not self.connected:
            return OrchestrationResult(
                query=query,
                intent_result=IntentResult(
                    QueryIntent.UNKNOWN, 0.0, [], {}, "Not connected"
                ),
                linked_entities=[],
                retrieval_result=RetrievalResult(
                    "unknown", "none", [], 0, 0.0, {}, {"error": "Not connected"}
                ),
                context_bundle=ContextBundle(
                    [], [], {}, [], datetime.now().isoformat(), "error"
                ),
                processing_time=0.0,
                success=False,
                error_message="Not connected to Neo4j database",
            )

        try:
            # Step 1: Intent Classification
            if verbose or debug:
                self.console.print(
                    f"[bold blue]🧠 Step 1: Intent Classification[/bold blue]"
                )
                self.console.print(f"[dim blue]Input: {query}[/dim blue]")

            # Use LLM intent classifier with clarification support
            intent_result = self.llm_intent_classifier.classify(query)

            # Handle clarification case
            if isinstance(intent_result, ClarificationResult):
                return self._handle_clarification(
                    intent_result, query, limit, verbose, debug
                )

            # Handle multiple intent case
            if intent_result.intent == "multiple":
                # Extract intents from the clarification template
                clarification = (
                    intent_result.reasoning.split(": ")[-1]
                    if ":" in intent_result.reasoning
                    else ""
                )
                # For now, we'll need to get the intents from the clarification context
                # This is a simplified approach - in practice, we'd store the intents in the result
                return self._handle_multiple_intent(
                    query, ["cwe_search", "mitigation_search"], limit, verbose, debug
                )

            if verbose or debug:
                self.console.print(
                    f"[bold green]OK: Intent: {intent_result.intent} (confidence: {intent_result.confidence:.2f})[/bold green]"
                )
                self.console.print(
                    f"[green]💭 Reasoning: {intent_result.reasoning}[/green]"
                )
                self.console.print(f"[green]💰 Cost: ${intent_result.cost:.6f}[/green]")

                if debug:
                    self.console.print(
                        f"[bold yellow] DEBUG - Intent Classification Details:[/bold yellow]"
                    )
                    self.console.print(f"[dim yellow]  • Query: '{query}'[/dim yellow]")
                    self.console.print(
                        f"[dim yellow]  • Intent: {intent_result.intent}[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Confidence: {intent_result.confidence:.3f}[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Reasoning: {intent_result.reasoning}[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Cost: ${intent_result.cost:.6f}[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Classification Reasoning: {intent_result.reasoning}[/dim yellow]"
                    )

            # Step 2: Entity Linking
            if verbose or debug:
                self.console.print(f"[bold blue]🔗 Step 2: Entity Linking[/bold blue]")

            linked_entities = self.entity_linker.link_entities(query)

            if verbose or debug:
                self.console.print(
                    f"[bold green]OK: Linked {len(linked_entities)} entities[/bold green]"
                )
                for entity in linked_entities:
                    self.console.print(
                        f"[green]  • {entity.text} → {entity.canonical_id} ({entity.entity_type.value})[/green]"
                    )

                if debug:
                    self.console.print(
                        f"[bold yellow] DEBUG - Entity Linking Details:[/bold yellow]"
                    )
                    for i, entity in enumerate(linked_entities, 1):
                        self.console.print(
                            f"[dim yellow]  {i}. Text: '{entity.text}'[/dim yellow]"
                        )
                        self.console.print(
                            f"[dim yellow]     Canonical ID: {entity.canonical_id}[/dim yellow]"
                        )
                        self.console.print(
                            f"[dim yellow]     Entity Type: {entity.entity_type.value}[/dim yellow]"
                        )
                        self.console.print(
                            f"[dim yellow]     Confidence: {entity.confidence:.3f}[/dim yellow]"
                        )
                        self.console.print(
                            f"[dim yellow]     Source: {entity.source}[/dim yellow]"
                        )
                        self.console.print(
                            f"[dim yellow]     Metadata: {entity.metadata}[/dim yellow]"
                        )
                        self.console.print()

            # Step 3: Graph Retrieval
            if verbose or debug:
                self.console.print(f"[bold blue] Step 3: Graph Retrieval[/bold blue]")

            retrieval_result = self.graph_retrieval.retrieve(
                query,
                intent_result.intent,
                linked_entities,
                limit,
                verbose,
                debug,
            )

            if verbose or debug:
                self.console.print(
                    f"[bold green]OK: Retrieved {retrieval_result.total_count} results in {retrieval_result.execution_time:.3f}s[/bold green]"
                )

                if debug:
                    self.console.print(
                        f"[bold yellow] DEBUG - Graph Retrieval Details:[/bold yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Query Intent: {retrieval_result.query_intent}[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Template Used: {retrieval_result.template_used}[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Parameters: {retrieval_result.parameters}[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Execution Time: {retrieval_result.execution_time:.3f}s[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Total Results: {retrieval_result.total_count}[/dim yellow]"
                    )
                    self.console.print(
                        f"[dim yellow]  • Metadata: {retrieval_result.metadata}[/dim yellow]"
                    )

                    if retrieval_result.results:
                        self.console.print(
                            f"[dim yellow]  • Sample Results:[/dim yellow]"
                        )
                        for i, result in enumerate(retrieval_result.results[:3], 1):
                            self.console.print(
                                f"[dim yellow]    {i}. {result}[/dim yellow]"
                            )
                    else:
                        self.console.print(
                            f"[dim yellow]  • No results found[/dim yellow]"
                        )

            # Step 4: Context Bundling
            if verbose or debug:
                self.console.print(
                    f"[bold blue]📦 Step 4: Context Bundling[/bold blue]"
                )

            context_bundle = self.graph_retrieval.create_context_bundle(
                retrieval_result, query
            )

            if debug:
                self.console.print(
                    f"[bold yellow] DEBUG - Context Bundle Details:[/bold yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Facts: {len(context_bundle.facts)} facts[/dim yellow]"
                )
                for i, fact in enumerate(context_bundle.facts[:3], 1):
                    self.console.print(f"[dim yellow]    {i}. {fact}[/dim yellow]")
                self.console.print(
                    f"[dim yellow]  • Citations: {len(context_bundle.citations)} citations[/dim yellow]"
                )
                for i, citation in enumerate(context_bundle.citations[:3], 1):
                    self.console.print(f"[dim yellow]    {i}. {citation}[/dim yellow]")
                self.console.print(
                    f"[dim yellow]  • Limits: {context_bundle.limits}[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Trace: {len(context_bundle.trace)} trace entries[/dim yellow]"
                )
                for i, trace_entry in enumerate(context_bundle.trace, 1):
                    self.console.print(
                        f"[dim yellow]    {i}. {trace_entry}[/dim yellow]"
                    )
                self.console.print(
                    f"[dim yellow]  • Query ID: {context_bundle.query_id}[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Timestamp: {context_bundle.timestamp}[/dim yellow]"
                )

            processing_time = (datetime.now() - start_time).total_seconds()

            if debug:
                self.console.print(
                    f"[bold yellow] DEBUG - Final Processing Summary:[/bold yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Total Processing Time: {processing_time:.3f}s[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Intent Classification: {intent_result.intent}[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Entities Linked: {len(linked_entities)}[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Results Retrieved: {retrieval_result.total_count}[/dim yellow]"
                )
                self.console.print(
                    f"[dim yellow]  • Context Facts: {len(context_bundle.facts)}[/dim yellow]"
                )
                self.console.print(f"[dim yellow]  • Success: True[/dim yellow]")

            return OrchestrationResult(
                query=query,
                intent_result=intent_result,
                linked_entities=linked_entities,
                retrieval_result=retrieval_result,
                context_bundle=context_bundle,
                processing_time=processing_time,
                success=True,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return OrchestrationResult(
                query=query,
                intent_result=IntentResult(QueryIntent.UNKNOWN, 0.0, [], {}, "Error"),
                linked_entities=[],
                retrieval_result=RetrievalResult(
                    "error", "none", [], 0, 0.0, {}, {"error": str(e)}
                ),
                context_bundle=ContextBundle(
                    [], [], {}, [], datetime.now().isoformat(), "error"
                ),
                processing_time=processing_time,
                success=False,
                error_message=str(e),
            )

    def _handle_clarification(
        self,
        clarification: ClarificationResult,
        query: str,
        limit: int,
        verbose: bool,
        debug: bool,
    ) -> OrchestrationResult:
        """Handle clarification requests from the user."""
        if verbose or debug:
            self.console.print(f"[bold yellow]🤔 Clarification Needed[/bold yellow]")
            self.console.print(f"[yellow]Query: {clarification.query}[/yellow]")
            self.console.print(f"[yellow]Question: {clarification.question}[/yellow]")
            self.console.print(f"[yellow]Options:[/yellow]")
            for i, option in enumerate(clarification.options, 1):
                self.console.print(f"[yellow]  {i}. {option}[/yellow]")

        # For now, return a clarification result that the CLI can handle
        return OrchestrationResult(
            query=query,
            intent_result=clarification.original_result,
            linked_entities=[],
            retrieval_result=RetrievalResult(
                query_intent="clarification_needed",
                template_used="none",
                results=[],
                total_count=0,
                execution_time=0.0,
                parameters={},
                metadata={"clarification": clarification, "clarification_needed": True},
            ),
            context_bundle=ContextBundle(
                [], [], {}, [], datetime.now().isoformat(), "clarification_needed"
            ),
            processing_time=0.0,
            success=True,
            error_message=None,
        )

    def _handle_multiple_intent(
        self, query: str, intents: List[str], limit: int, verbose: bool, debug: bool
    ) -> OrchestrationResult:
        """Handle multiple intents by running each one and combining results."""
        if verbose or debug:
            self.console.print(
                f"[bold blue]🔄 Processing Multiple Intents: {', '.join(intents)}[/bold blue]"
            )

        all_results = []
        total_count = 0

        for intent in intents:
            if verbose or debug:
                self.console.print(f"[dim blue]  • Processing {intent}...[/dim blue]")

            # Create a mock intent result for this specific intent
            mock_intent_result = IntentResult(
                intent=intent,
                confidence=1.0,
                reasoning=f"Multiple intent processing for {intent}",
                cost=0.0,
                tokens_used=0,
            )

            # Link entities for this intent
            linked_entities = self.entity_linker.link_entities(query)

            # Retrieve results for this intent
            retrieval_result = self.graph_retrieval.retrieve(
                query, intent, linked_entities, limit, verbose, debug
            )

            all_results.extend(retrieval_result.results)
            total_count += retrieval_result.total_count

        # Create combined result
        combined_retrieval = RetrievalResult(
            query_intent="multiple",
            template_used="combined",
            results=all_results[:limit],  # Limit combined results
            total_count=total_count,
            execution_time=0.0,
            parameters={},
            metadata={"intents_processed": intents, "multiple_intent": True},
        )

        return OrchestrationResult(
            query=query,
            intent_result=IntentResult(
                intent="multiple",
                confidence=1.0,
                reasoning=f"Processed multiple intents: {', '.join(intents)}",
                cost=0.0,
                tokens_used=0,
            ),
            linked_entities=[],
            retrieval_result=combined_retrieval,
            context_bundle=ContextBundle(
                [], [], {}, [], datetime.now().isoformat(), "multiple_intent"
            ),
            processing_time=0.0,
            success=True,
            error_message=None,
        )

    def display_result(self, result: OrchestrationResult, verbose: bool = False):
        """Display orchestration result in a rich format."""
        if not result.success:
            self.console.print(f"[red]ERROR: Error: {result.error_message}[/red]")
            return

        # Display intent and entities
        intent_panel = Panel(
            f"[bold]Intent:[/bold] {result.intent_result.intent}\n"
            f"[bold]Confidence:[/bold] {result.intent_result.confidence:.2f}\n"
            f"[bold]Entities:[/bold] {len(result.linked_entities)}\n"
            f"[bold]Processing Time:[/bold] {result.processing_time:.3f}s",
            title="Query Analysis",
            border_style="blue",
        )
        self.console.print(intent_panel)

        # Display linked entities
        if result.linked_entities:
            entity_table = Table(title="Linked Entities")
            entity_table.add_column("Text", style="cyan")
            entity_table.add_column("Canonical ID", style="green")
            entity_table.add_column("Type", style="yellow")
            entity_table.add_column("Confidence", style="magenta")

            for entity in result.linked_entities:
                entity_table.add_row(
                    entity.text,
                    entity.canonical_id,
                    entity.entity_type.value,
                    f"{entity.confidence:.2f}",
                )

            self.console.print(entity_table)

        # Display retrieval results
        self.graph_retrieval.display_results(
            result.retrieval_result, result.context_bundle, verbose
        )

        # Removed redundant Key Facts section - information is already in the results table

    def get_supported_intents(self) -> List[str]:
        """Get list of supported query intents."""
        return [intent.value for intent in QueryIntent if intent != QueryIntent.UNKNOWN]

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities."""
        return {
            "connected": self.connected,
            "supported_intents": self.get_supported_intents(),
            "neo4j_uri": self.graph_retrieval.neo4j_uri,
            "components": {
                "intent_classifier": "active",
                "entity_linker": "active",
                "graph_retrieval": "active" if self.connected else "inactive",
            },
        }


def test_orchestrator():
    """Test the orchestrator with sample queries."""
    orchestrator = CLAIREOrchestrator()

    if not orchestrator.connect():
        print("ERROR: Failed to connect to Neo4j")
        return

    print("🎯 CLAIRE Orchestrator Test:")
    print("=" * 50)

    test_queries = [
        "Show me recent XSS vulnerabilities",
        "Find CVEs for CWE-79",
        "What are the critical CVEs from 2024?",
        "Explain CWE-79",
        "Find ATT&CK technique T1059",
    ]

    for query in test_queries:
        print(f"\n Testing: {query}")
        result = orchestrator.process_query(query, limit=5, verbose=True)

        if result.success:
            print(f"OK: Success: {result.retrieval_result.total_count} results")
            print(f"Time: {result.processing_time:.3f}s")
        else:
            print(f"ERROR: Error: {result.error_message}")

    orchestrator.close()


if __name__ == "__main__":
    test_orchestrator()
