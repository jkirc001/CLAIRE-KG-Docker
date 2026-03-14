#!/usr/bin/env python3
"""
CLAIRE-KG CLI - Command-line interface for the CLAIRE Knowledge Graph.

Provides:
  - Query commands: ask, query (natural language questions), evaluate (DeepEval)
  - Setup commands: ingest, crosswalk, embeddings, clean, status
  - Test commands: debug, debug_help

Entry point is typically `claire-kg` or `python -m claire_kg`. Use --help on any
command for options. Example questions: use `ask --examples list` then
`ask --examples CVE` or `ask --select` for interactive choice.
"""

import typer
import json
import re
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional, List
import os
from pathlib import Path
from datetime import datetime

from .database import Neo4jConnection
from .query_orchestrator import QueryOrchestrator
from .llm_orchestrator import LLMOrchestrator
from .ingest import DatasetIngester
from .cypher_generator import invalidate_all_schema_caches

# from .rag_system import RAGSystem  # Archived
# from .orchestrator import CLAIREOrchestrator  # Archived
# from .langchain_rag_orchestrator import LangChainRAGOrchestrator  # Archived

# -----------------------------------------------------------------------------
# Event loop / process exit handling (Python 3.13 + DeepEval/httpx)
# -----------------------------------------------------------------------------


def _suppress_event_loop_closed_on_exit():
    """Suppress 'Event loop is closed' from async httpx cleanup at process exit (Python 3.13 + DeepEval)."""
    _orig = getattr(sys, "unraisablehook", None)

    def hook(unraisable):
        """Suppress 'Event loop is closed' unraisable; otherwise forward to original hook."""
        if (
            unraisable.exc_type is RuntimeError
            and unraisable.exc_value
            and str(unraisable.exc_value) == "Event loop is closed"
        ):
            # Suppress: async httpx/httpcore client cleanup at process exit (Python 3.13 + DeepEval)
            return
        if _orig is not None:
            _orig(unraisable)

    if hasattr(sys, "unraisablehook"):
        sys.unraisablehook = hook


# Run once at import so ask/evaluate commands are covered
_suppress_event_loop_closed_on_exit()


def _install_asyncio_event_loop_closed_handler():
    """Suppress 'Task exception was never retrieved' for Event loop is closed (Python 3.13 + httpx/DeepEval).
    DeepEval runs metrics in threads that create their own event loops; when Faithfulness times out,
    those loops close and httpx AsyncClient.aclose() raises. Patch the default exception handler
    so any loop (including in worker threads) suppresses this specific exception.
    """
    try:
        import asyncio
    except ImportError:
        return
    try:
        base_loop = asyncio.BaseEventLoop
    except AttributeError:
        return
    _orig_default = base_loop.default_exception_handler

    def _patched_default_exception_handler(loop, context):
        """Suppress 'Event loop is closed' exception; otherwise call original handler."""
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and str(exc) == "Event loop is closed":
            return
        _orig_default(loop, context)

    base_loop.default_exception_handler = _patched_default_exception_handler

    # Also set handler on the main thread's default loop for consistency
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(_patched_default_exception_handler)


_install_asyncio_event_loop_closed_handler()

# -----------------------------------------------------------------------------
# Typer app and command groups
# -----------------------------------------------------------------------------

# Main app
app = typer.Typer(
    name="claire-kg",
    help="CLAIRE Knowledge Graph - Cybersecurity Data Ingestion and Analysis\n\nUse 'ask --help' to see how to ask questions and view example questions with --examples.",
    add_completion=False,
)

# Subcommand groups
setup_app = typer.Typer(help="Setup and initialize the knowledge graph")
query_app = typer.Typer(help="Query and search the knowledge graph")
test_app = typer.Typer(help="Testing, validation, and development tools")

# Add subcommand groups to main app
app.add_typer(setup_app, name="setup", help="Setup and initialize the knowledge graph")
app.add_typer(query_app, name="queries", help="Query and search the knowledge graph")
app.add_typer(test_app, name="test", help="Testing, validation, and development tools")

# -----------------------------------------------------------------------------
# Main query commands: query (simple), ask (full), evaluate (DeepEval)
# -----------------------------------------------------------------------------


# Convenience command: simple ask with table width/verbose options (no --eval, --compare, etc.)
@app.command()
def query(
    question: str = typer.Argument(
        ..., help="Natural language question to ask the knowledge graph"
    ),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of results to return"
    ),
    wide: bool = typer.Option(
        False,
        "--wide",
        "-w",
        help="Wide mode: sets total table width to 160 (overridden by --size)",
    ),
    size: int = typer.Option(
        None,
        "--size",
        "-s",
        help="Custom total table width (overrides default and --wide). Minimum 20.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose rendering for Description (wrap/fold)"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Debug mode with detailed step-by-step output"
    ),
):
    """Ask intelligent questions to the CLAIRE knowledge graph using natural language."""

    # Delegate to ask() with resolved width/verbose; ask() is defined below.
    console.print("[bold blue]CLAIRE-KG Intelligent Query[/bold blue]")
    console.print(f"Question: [green]{question}[/green]")
    console.print(f"Limit: [cyan]{limit}[/cyan]")

    # Table width: --size overrides --wide over default 120; minimum 20.
    resolved_width = 120
    if size is not None:
        resolved_width = max(20, int(size))
    elif wide:
        resolved_width = 160
    console.print(
        f"[dim]Width: {resolved_width} (verbose={'on' if verbose else 'off'})[/dim]"
    )

    try:
        # Use LLM orchestrator for consistent Cypher generation path
        orchestrator = QueryOrchestrator()

        if not orchestrator.connect():
            console.print("[red]ERROR: Failed to connect to Neo4j database[/red]")
            raise typer.Exit(1)

        console.print("[green]OK: Connected to Neo4j database[/green]")

        # Process the query (verbose flag toggles detailed result display)
        console.print("[green]Processing query...[/green]")
        result = orchestrator.process_query(
            question, limit=limit, verbose=verbose, debug=debug
        )

        if result["success"]:
            console.print("[green]OK: Query completed successfully[/green]")
        else:
            console.print(f"[red]ERROR: Query failed: {result['error_message']}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        raise typer.Exit(1)
    finally:
        if "orchestrator" in locals():
            orchestrator.close()


@app.command()
def ask(
    question: Optional[str] = typer.Argument(
        None,
        help="Natural language question to ask the knowledge graph, or a question number (e.g., '1' for question #1 from examples)",
    ),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of results to return"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Debug mode with detailed step-by-step output"
    ),
    compare: bool = typer.Option(
        False,
        "--compare",
        "-c",
        help="Compare CLAIRE-KG vs Direct LLM (question directly to LLM with no database prompting)",
    ),
    eval: bool = typer.Option(
        False,
        "--eval",
        "-e",
        help="Enable Phase 3 DeepEval evaluation with GEval (adds quality metrics, increases cost)",
    ),
    no_geval: bool = typer.Option(
        False,
        "--no-geval",
        help="Disable GEval metric when using --eval (only Relevancy + Faithfulness)",
    ),
    save_results: Optional[str] = typer.Option(
        None,
        "--save",
        help="Save evaluation results to files. Creates <filename>.json, <filename>.md, and <filename>_debug.txt in tests/outputs/ (or CLAIRE_OUTPUT_DIR). Automatically enables --eval and --debug (no need to specify them separately). Example: --save HV01",
    ),
    show_json: bool = typer.Option(
        False,
        "--show-json",
        "-j",
        help="Show Phase 1 JSON output structure (hidden option)",
        hidden=True,
    ),
    debug_file: str = typer.Option(
        None,
        "--debug-file",
        help="Save debug output to file (requires --debug). Hidden option - use 'tee' instead: command 2>&1 | tee debug.txt",
        hidden=True,
    ),
    phase1: bool = typer.Option(
        False,
        "--phase1",
        help="Run Phase 1 only (query generation and execution), stop before Phase 2 enhancement",
    ),
    phase2: bool = typer.Option(
        False,
        "--phase2",
        help="Run Phase 2 only (answer enhancement), skip Phase 1 query generation",
    ),
    class_only: bool = typer.Option(
        False,
        "--class",
        help="Run question classifier only, show classification result and stop",
    ),
    no_metadata: bool = typer.Option(
        False,
        "--no-metadata",
        help="Disable metadata-based role detection in classifier (use pattern-based only)",
    ),
    examples: Optional[str] = typer.Option(
        None,
        "--examples",
        help="Show example questions. Use '--examples=' or '--examples list' to see categories. Options: 'all', 'list', 'CVE', 'CWE', 'CAPEC', 'ATT&CK', 'NICE', 'DCWF', or crosswalk like 'CVE->CWE' (use quotes for crosswalks: --examples 'CVE->CWE')",
    ),
    select: bool = typer.Option(
        False,
        "--select",
        "-s",
        help="Interactive mode: show examples and prompt to select a question number to ask",
    ),
    cache: bool = typer.Option(
        False,
        "--cache",
        help="Enable Cypher query cache for this run (cache is off by default; use for faster repeated questions)",
    ),
    phase1_model: Optional[str] = typer.Option(
        None,
        "--phase1-model",
        help="OpenAI model for Phase 1 (Cypher generation). Overrides PHASE1_MODEL env var. E.g. gpt-4o, gpt-5.2.",
    ),
    phase2_model: Optional[str] = typer.Option(
        None,
        "--phase2-model",
        help="OpenAI model for Phase 2 (answer enhancement). Overrides PHASE2_MODEL env var. E.g. gpt-4o, gpt-5.2.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Set Phase 1 and Phase 2 (generation) to this model. Phase 3 (evaluation) is unchanged unless you set --phase3-model. E.g. gpt-5.2.",
    ),
    phase3_model: Optional[str] = typer.Option(
        None,
        "--phase3-model",
        help="OpenAI model for Phase 3 DeepEval metrics (Relevancy, Faithfulness, GEval). Default is from env (e.g. gpt-4o). Use this to match grader to generation model.",
    ),
):
    """
    Ask questions about cybersecurity data from the CLAIRE Knowledge Graph.

    Use --examples to see example questions, or --examples= to see available categories.

    CLAIRE-KG answers questions about vulnerabilities (CVE), weaknesses (CWE),
    attack patterns (CAPEC), ATT&CK techniques, and workforce roles (NICE/DCWF).

    KEYWORDS TO USE:
    • Datasets: CVE, CWE, CAPEC, ATT&CK, NICE, DCWF, Mitigation, Asset
    • Entity IDs: CVE-2024-12345, CWE-79, CAPEC-100, T1574
    • Terms: vulnerability, weakness, technique, attack pattern, work role
    • Operations: "What is...", "Show me...", "How many...", "List...", "Count..."

    BEST PRACTICES:
    • Use specific entity IDs when possible (CVE-2024-20439, CWE-79, T1574)
    • Mention dataset names explicitly (CVE, CWE, ATT&CK, etc.)
    • Be clear about what you want (lookup, count, relationships, path finding)
    • For crosswalks: "What CVEs are related to CWE-79?" (links multiple datasets)

    📚 EXAMPLE QUESTIONS:
    Use --examples to see example questions from the test suite:
    • --examples= or --examples list  → See available categories
    • --examples CVE                   → View CVE vulnerability examples
    • --examples 'CVE->CWE'            → View crosswalk examples (use quotes!)
    • --examples all                   → View all 100+ example questions

    Note: Crosswalk examples (like CVE->CWE) must be quoted to prevent shell redirection.

    You can also ask questions by number:
    • ask 1                            → Ask question #1 from examples
    • ask 8 --debug                  → Ask question #8 with debug output

    💡 TIPS:
    • Use --class to see how your question is classified
    • Use --debug to see the generated Cypher query
    • Use --phase1 to see raw database results
    """
    # Handle --examples option
    # Check if --examples appears in sys.argv
    if "--examples" in sys.argv or any("--examples" in arg for arg in sys.argv):
        # Check if it was used with = (empty value) or without value
        examples_arg = None
        for i, arg in enumerate(sys.argv):
            if arg == "--examples":
                # Check if next arg is not another option (starts with -)
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                    examples_arg = sys.argv[i + 1]
                    break
                # If --examples is last arg or next is another option, treat as empty
                if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("-"):
                    examples_arg = ""
                    break
            elif arg.startswith("--examples="):
                examples_arg = arg.split("=", 1)[1] if "=" in arg else ""
                break

        # If examples was provided as parameter, use that (overrides sys.argv parsing)
        if examples is not None:
            examples_arg = examples

        # If examples_arg is None or empty, show list (help)
        if examples_arg is None or examples_arg.strip() == "":
            if select:
                # Show list with numbers and allow selection
                categories_list = _show_examples("list", console, return_list=True)
                if categories_list:
                    selected_category = _prompt_for_selection(
                        categories_list, console, is_category=True
                    )
                    if selected_category:
                        # Recursively call with the selected category
                        examples_to_show = _show_examples(
                            selected_category.strip(), console, return_list=True
                        )
                        if examples_to_show:
                            selected_question = _prompt_for_selection(
                                examples_to_show, console
                            )
                            if selected_question:
                                question = selected_question
                                # Continue with normal processing
                            else:
                                raise typer.Exit(0)
                        else:
                            raise typer.Exit(0)
                    else:
                        raise typer.Exit(0)
                else:
                    raise typer.Exit(0)
            else:
                _show_examples("list", console)
                raise typer.Exit(0)
        else:
            # Check if examples_arg is a number (category selection by number)
            category_name = None
            try:
                category_num = int(examples_arg.strip())
                # Get the category list and find the category by number
                categories_list = _show_examples("list", console, return_list=True)
                if categories_list:
                    for num, name in categories_list:
                        if num == category_num:
                            category_name = name
                            break
                    if category_name:
                        # Use the category name instead of the number
                        examples_arg = category_name
                    else:
                        console.print(f"[red]Category #{category_num} not found[/red]")
                        console.print(
                            "[yellow]Use --examples list to see available categories[/yellow]"
                        )
                        raise typer.Exit(1)
            except ValueError:
                # Not a number, use as-is
                pass

            # If --select is used, get the list and display it
            if select:
                # Get the list (which will also display it)
                examples_to_show = _show_examples(
                    examples_arg.strip(), console, return_list=True
                )

                # Prompt for selection
                if examples_to_show:
                    selected_question = _prompt_for_selection(examples_to_show, console)
                    if selected_question:
                        # Use the selected question as the question argument
                        question = selected_question
                        # Continue with normal processing (don't exit)
                    else:
                        # User cancelled or invalid selection
                        raise typer.Exit(0)
                else:
                    raise typer.Exit(0)
            else:
                # Normal display mode
                _show_examples(examples_arg.strip(), console, return_list=False)
                raise typer.Exit(0)

    # Question is required if not using --examples
    if question is None:
        console.print(
            "[red]ERROR: QUESTION argument is required (or use --examples)[/red]"
        )
        raise typer.Exit(1)

    # Check if question is a number (question ID from examples)
    question_number = None
    try:
        question_number = int(question.strip())
        # If it's a valid number, load the question from baseline examples
        question = _load_question_by_number(question_number, console)
        if question is None:
            console.print(
                f"[red]ERROR: Question #{question_number} not found in examples[/red]"
            )
            console.print(
                "[yellow]Use --examples all to see available question numbers[/yellow]"
            )
            raise typer.Exit(1)
        if debug:
            console.print(
                f"[dim]Loaded question #{question_number} from examples[/dim]"
            )
    except ValueError:
        # Not a number, use as-is
        pass

    # Validate flags
    if phase1 and phase2:
        console.print("[red]ERROR: Cannot use --phase1 and --phase2 together[/red]")
        raise typer.Exit(1)

    if debug_file and not debug:
        console.print("[red]ERROR: --debug-file requires --debug flag[/red]")
        raise typer.Exit(1)

    # Quick validation for placeholder questions (before any processing)
    question_lower = question.lower().strip()

    # Comprehensive placeholder detection
    placeholder_phrases = [
        "your question here",
        "insert question here",
        "enter question here",
        "type question here",
        "example question",
        "test question",
        "sample question",
        "placeholder",
        "question goes here",
        "ask a question",
        "please enter",
    ]

    # Also check for patterns like "insert/enter/type" + "question/query"
    placeholder_patterns = [
        r"\b(insert|enter|type|add|put)\s+(your\s+)?(question|query|text)\s+(here|below|above)\b",
        r"\b(question|query)\s+(here|below|above|to\s+start)\b",
        r"^\s*(example|test|sample|placeholder|demo)\s*(question|query)?\s*$",
    ]

    is_placeholder = (
        any(phrase in question_lower for phrase in placeholder_phrases)
        or question_lower in ["", "?", "??", "???", "test", "example", "sample"]
        or any(re.search(pattern, question_lower) for pattern in placeholder_patterns)
    )

    if len(question_lower) < 10 and any(
        word in question_lower
        for word in [
            "question",
            "query",
            "ask",
            "example",
            "test",
            "sample",
            "insert",
            "enter",
            "type",
        ]
    ):
        is_placeholder = True

    if is_placeholder:
        console.print(
            "[red]ERROR: This appears to be a placeholder question, not a real query.[/red]"
        )
        console.print(
            "[yellow]TIP: Please provide an actual cybersecurity question, such as:[/yellow]"
        )
        console.print("  • 'What are the most critical CVEs from 2024?'")
        console.print("  • 'Show me XSS vulnerabilities'")
        console.print("  • 'What is CWE-79?'")
        console.print("  • 'Explain buffer overflow attacks'")
        raise typer.Exit(1)

    # Check for help requests or questions about CLAIRE
    # Only trigger if it's clearly about the system itself, not cybersecurity topics
    simple_help = question_lower in [
        "help",
        "?",
        "help me",
        "show help",
        "what is claire",
        "what is claire-kg",
    ]
    system_help = any(
        phrase in question_lower
        for phrase in [
            "how to use",
            "how does this work",
            "what can you do",
            "how do i use",
            "what is this",
            "what does this do",
            "how does claire work",
            "what is claire",
            "tell me about claire",
            "explain claire",
            "information about claire",
        ]
    )
    # Combined help + claire keywords (but not if it's about cybersecurity topics)
    help_keywords = [
        "help",
        "what is",
        "how does",
        "how do",
        "what does",
        "explain",
        "tell me about",
        "information about",
    ]
    claire_keywords = [
        "claire",
        "claire-kg",
        "this system",
        "this tool",
        "you",
        "your purpose",
    ]
    combined_help = (
        any(help_kw in question_lower for help_kw in help_keywords)
        and any(claire_kw in question_lower for claire_kw in claire_keywords)
        and
        # Exclude if it contains cybersecurity entity keywords (not about CLAIRE)
        not any(
            entity in question_lower
            for entity in [
                "cve",
                "cwe",
                "capec",
                "attack",
                "vulnerability",
                "weakness",
                "technique",
                "tactic",
            ]
        )
    )

    is_help_request = simple_help or system_help or combined_help

    if is_help_request:
        console.print("[bold blue]CLAIRE-KG Help[/bold blue]")
        console.print()
        console.print("[bold cyan]What is CLAIRE-KG?[/bold cyan]")
        console.print(
            "CLAIRE-KG is a cybersecurity knowledge graph that answers questions using data from:"
        )
        console.print("  • CVE (Common Vulnerabilities and Exposures)")
        console.print("  • CWE (Common Weakness Enumeration)")
        console.print(
            "  • CAPEC (Common Attack Pattern Enumeration and Classification)"
        )
        console.print("  • MITRE ATT&CK Framework")
        console.print("  • NICE Framework (workforce roles, skills, knowledge)")
        console.print("  • DCWF (DoD Cyber Workforce Framework)")
        console.print()
        console.print("[bold cyan]How to use:[/bold cyan]")
        console.print("Ask cybersecurity questions in natural language, such as:")
        console.print()
        console.print("[green]Example Questions:[/green]")
        console.print("  • 'What are the most critical CVEs from 2024?'")
        console.print("  • 'Show me XSS vulnerabilities'")
        console.print("  • 'What is CWE-79 (Cross-site Scripting)?'")
        console.print("  • 'Explain buffer overflow attacks'")
        console.print(
            "  • 'What are the skills needed to defend against SQL injection?'"
        )
        console.print("  • 'Show me CVEs related to CWE-79'")
        console.print("  • 'What attack patterns exploit buffer overflows?'")
        console.print("  • 'What work roles are involved in vulnerability assessment?'")
        console.print()
        console.print("[bold cyan]Command Options:[/bold cyan]")
        console.print(
            "  [cyan]--debug, -d[/cyan]     Show detailed step-by-step processing"
        )
        console.print("  [cyan]--debug-file[/cyan]    Save debug output to file")
        console.print(
            "  [cyan]--limit[/cyan]          Maximum number of results (default: 10)"
        )
        console.print("  [cyan]--show-json, -j[/cyan] Show Phase 1 JSON structure")
        console.print()
        console.print("[bold cyan]For more information:[/bold cyan]")
        console.print(
            "  Run [cyan]uv run python -m claire_kg.cli --help[/cyan] for all commands"
        )
        console.print(
            "  Run [cyan]uv run python -m claire_kg.cli test debug-help[/cyan] for debug help"
        )
        raise typer.Exit(0)

    # Handle --class option: run classifier only
    if class_only:
        from .question_classifier import QuestionClassifier

        use_metadata = not no_metadata
        classifier = QuestionClassifier(use_metadata=use_metadata)
        classification = classifier.classify(question)

        console.print("[bold blue]CLAIRE-KG Question Classifier[/bold blue]")
        console.print(f"Question: [green]{question}[/green]")
        console.print()

        # Display classification results
        from rich.table import Table

        table = Table(title="Classification Results", box=box.ASCII)
        table.add_column("Category", style="cyan")
        table.add_column("Value", style="green")

        if classification.primary_datasets:
            table.add_row(
                "Primary Datasets", ", ".join(classification.primary_datasets)
            )
        else:
            table.add_row("Primary Datasets", "None")

        if classification.crosswalk_groups:
            table.add_row(
                "Crosswalk Groups", ", ".join(classification.crosswalk_groups)
            )
        else:
            table.add_row("Crosswalk Groups", "None")

        table.add_row("Complexity Level", classification.complexity_level)

        if classification.intent_types:
            table.add_row("Intent Types", ", ".join(classification.intent_types))
        else:
            table.add_row("Intent Types", "None")

        if classification.expected_schema_pack:
            table.add_row(
                "Expected Schema Pack", ", ".join(classification.expected_schema_pack)
            )
        else:
            table.add_row("Expected Schema Pack", "None")

        if classification.key_properties:
            table.add_row("Key Properties", ", ".join(classification.key_properties))
        else:
            table.add_row("Key Properties", "None")

        if classification.potential_failure_pattern:
            table.add_row(
                "Potential Failure Pattern", classification.potential_failure_pattern
            )
        else:
            table.add_row("Potential Failure Pattern", "None")

        console.print(table)
        return

    # Only show header in debug mode
    if debug:
        if phase1:
            mode_str = "Phase 1 Only"
        elif phase2:
            mode_str = "Phase 2 Only"
        else:
            mode_str = "Phase 1 -> Phase 2"
        console.print(f"[bold blue]CLAIRE-KG Ask ({mode_str})[/bold blue]")
        console.print(f"Question: [green]{question}[/green]")
        console.print(f"Limit: [cyan]{limit}[/cyan]")
        if debug_file:
            console.print(f"Debug file: [cyan]{debug_file}[/cyan]")
        if save_results:
            output_dir = os.getenv("CLAIRE_OUTPUT_DIR", "tests/outputs")
            save_stem = (
                save_results.rsplit(".", 1)[0]
                if "." in (save_results or "")
                else (save_results or "")
            )
            console.print(f"Save files to: [cyan]{output_dir}/[/cyan]")
            console.print(
                f"  - [cyan]{save_stem}.json[/cyan] (or _1, _2, etc. if exists)"
            )
            console.print(
                f"  - [cyan]{save_stem}.md[/cyan] (or _1, _2, etc. if exists)"
            )
            console.print(
                f"  - [cyan]{save_stem}_debug.txt[/cyan] (or _1, _2, etc. if exists)"
            )
        console.print()

    try:
        # Auto-enable eval if --save is used (save requires evaluation results)
        if save_results and not eval:
            eval = True
            console.print(
                "[yellow]Warning: --save requires --eval. Enabling --eval automatically.[/yellow]"
            )

        # Auto-enable debug if --save is used (needed to capture full output)
        if save_results and not debug:
            debug = True
            console.print("[cyan]Debug mode enabled automatically for --save[/cyan]")

        # Enable Phase 3 evaluation if --eval flag is set (overrides env var)
        if eval:
            os.environ["PHASE3_EVALUATION_ENABLED"] = "true"
            if debug:
                console.print(
                    "[cyan]Phase 3 (DeepEval) evaluation enabled via --eval flag[/cyan]"
                )

            # GEval is enabled by default with --eval, unless --no-geval is set
            if not no_geval:
                os.environ["GEVAL_ENABLED"] = "true"
                if debug:
                    console.print(
                        "[cyan]GEval metric enabled by default (use --no-geval to disable)[/cyan]"
                    )
            else:
                # Explicitly disable GEval if --no-geval is set
                os.environ["GEVAL_ENABLED"] = "false"
                if debug:
                    console.print(
                        "[yellow]GEval metric disabled via --no-geval flag[/yellow]"
                    )

        # Override Phase 1 / Phase 2 from CLI. Phase 3 (evaluation) defaults to gpt-4o when only
        # --model is set, so "only the model changed" means only generation—grading stays consistent.
        if model is not None:
            os.environ["PHASE1_MODEL"] = model
            os.environ["PHASE2_MODEL"] = model
            if phase3_model is not None:
                os.environ["OPENAI_MODEL_NAME"] = phase3_model
                if debug:
                    console.print(f"[cyan]Model (Phase 1, 2 & 3): {model}[/cyan]")
            else:
                # Phase 3 stays on gpt-4o so grading is consistent (same answer won't fail due to stricter grader)
                os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
                if debug:
                    console.print(
                        f"[cyan]Model (Phase 1 & 2): {model}[/cyan], "
                        f"[cyan]Phase 3 (DeepEval): gpt-4o[/cyan]"
                    )
        else:
            if phase1_model is not None:
                os.environ["PHASE1_MODEL"] = phase1_model
                if debug:
                    console.print(f"[cyan]Phase 1 model: {phase1_model}[/cyan]")
            if phase2_model is not None:
                os.environ["PHASE2_MODEL"] = phase2_model
                if debug:
                    console.print(f"[cyan]Phase 2 model: {phase2_model}[/cyan]")
            if phase3_model is not None:
                os.environ["OPENAI_MODEL_NAME"] = phase3_model
                if debug:
                    console.print(f"[cyan]Phase 3 (DeepEval) model: {phase3_model}[/cyan]")

        # Use LLM orchestrator with Phase 1 → Phase 2 architecture
        db = Neo4jConnection()
        use_metadata = not no_metadata

        # If --save is used, enable debug and set debug_file to capture output
        # We'll use a temporary debug file that will be copied to the final location
        actual_debug = debug
        actual_debug_file = debug_file
        if save_results and not debug_file:
            # Enable debug if --save is used (needed to capture debug output)
            actual_debug = True
            import tempfile

            actual_debug_file = tempfile.mktemp(suffix=".txt", prefix="claire_debug_")

        orchestrator = LLMOrchestrator(
            db=db,
            debug=actual_debug,
            debug_file=actual_debug_file,
            use_classifier_metadata=use_metadata,
        )

        # Only show connection/processing messages in debug mode
        if debug:
            console.print("[green]OK: Connected to Neo4j database[/green]")
            console.print("[green]Processing query with LLM enhancement...[/green]")
            console.print()

        # If compare mode, run both CLAIRE-KG and Direct LLM
        if compare:
            from .cypher_generator import CypherGenerator
            import time

            console.print(
                "[bold blue]Comparison Mode: CLAIRE-KG vs Direct LLM[/bold blue]"
            )
            console.print()

            # Run CLAIRE-KG
            console.print(
                "[bold cyan]1. Running CLAIRE-KG (Phase 1 -> Phase 2)...[/bold cyan]"
            )
            claire_start = time.time()
            result = orchestrator.process_question(question, limit=limit)
            claire_time = time.time() - claire_start

            # Check if this was a help request
            if (
                result.success
                and not result.cypher_query
                and result.enhanced_answer
                and "CLAIRE-KG Help" in result.enhanced_answer
            ):
                console.print(result.enhanced_answer)
                return

            if not result.success:
                if getattr(result, "phase1_no_results", False):
                    console.print(
                        Panel(
                            result.enhanced_answer,
                            title="CLAIRE-KG (no usable results)",
                            border_style="yellow",
                            box=box.ASCII,
                            width=88,
                        )
                    )
                    return
                console.print(
                    f"[red]ERROR: CLAIRE-KG failed: {result.error or 'Unknown error'}[/red]"
                )
                raise typer.Exit(1)

            # Run Direct LLM
            console.print(
                "[bold cyan]2. Running Direct LLM (no database)...[/bold cyan]"
            )
            direct_start = time.time()
            cypher_generator = CypherGenerator()
            if (
                not hasattr(cypher_generator, "client")
                or cypher_generator.client is None
            ):
                cypher_generator._initialize_client()

            direct_prompt = f"""Please answer the following cybersecurity question based on your knowledge:

{question}

Provide a clear, comprehensive answer with relevant details. If you can cite specific CVEs, CWEs, CAPEC IDs, or ATT&CK techniques, please do so."""

            direct_response = cypher_generator.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert cybersecurity knowledge assistant. 
Answer questions based on your general knowledge of cybersecurity.
Be accurate and cite sources when possible. If you don't know something, say so.""",
                    },
                    {"role": "user", "content": direct_prompt},
                ],
                temperature=0.3,
                max_completion_tokens=2000,
                timeout=30,
            )

            direct_answer = direct_response.choices[0].message.content.strip()
            direct_time = time.time() - direct_start
            direct_cost = (
                direct_response.usage.prompt_tokens * 0.50
                + direct_response.usage.completion_tokens * 1.50
            ) / 1_000_000

            console.print()
            console.print("[bold green]OK: Both queries completed[/bold green]")
            console.print()

            # Show metrics
            console.print("[bold cyan]Metrics:[/bold cyan]")
            console.print(
                f"  CLAIRE-KG:  {claire_time:.2f}s | ${result.llm_cost_usd or 0:.6f} | {len(result.raw_data)} results"
            )
            console.print(f"  Direct LLM: {direct_time:.2f}s | ${direct_cost:.6f}")
            console.print()

            # Show side-by-side comparison
            from rich.columns import Columns

            claire_panel = Panel(
                result.enhanced_answer,
                title="[green]CLAIRE-KG (with Database)[/green]",
                border_style="green",
                box=box.ASCII,
                width=88,
                expand=False,
                padding=(0, 1),
            )

            direct_panel = Panel(
                direct_answer,
                title="[yellow]Direct LLM (No Database)[/yellow]",
                border_style="yellow",
                box=box.ASCII,
                width=88,
                expand=False,
                padding=(0, 1),
            )

            console.print("[bold blue] Side-by-Side Comparison:[/bold blue]")
            console.print()
            console.print(
                Columns([claire_panel, direct_panel], equal=True, expand=False)
            )

        else:
            # Normal mode - just CLAIRE-KG
            # If --phase1, run Phase 1 only and stop
            if phase1:
                # Use orchestrator.process_question() to get classification and schema injection
                # Then extract only Phase 1 results (phase1_only=True stops before Phase 2)
                result = orchestrator.process_question(
                    question, limit=limit, phase1_only=True
                )

                # Check if early rejection happened (no datasets detected)
                if (
                    result.success
                    and not result.cypher_query
                    and result.enhanced_answer
                    and (
                        "I couldn't identify" in result.enhanced_answer
                        or "doesn't appear to be about" in result.enhanced_answer
                    )
                ):
                    # Early rejection message - display it
                    console.print(result.enhanced_answer)
                    return

                # For --phase1, we need to stop before Phase 2
                # The debug output from process_question() already shows Phase 1 results
                # but we'll still show Phase 1 JSON output separately
                if not debug:
                    console.print("[bold green]OK: Phase 1 Complete[/bold green]")
                    console.print()

                # Extract Phase 1 results for JSON output
                cypher_query = result.cypher_query
                raw_data = result.raw_data or []

                # Show Phase 1 results summary (process_question() already shows debug info)
                from rich.syntax import Syntax

                console.print("[bold green]OK: Phase 1 Complete[/bold green]")
                console.print()
                console.print("[cyan]Cypher Query:[/cyan]")
                console.print(
                    Syntax(cypher_query, "cypher", theme="monokai", word_wrap=True)
                )
                console.print()
                console.print(f"[cyan]Results: {len(raw_data)} record(s)[/cyan]")

                if raw_data:
                    from rich.table import Table

                    table = Table()
                    # Get column names from first record
                    for key in raw_data[0].keys():
                        table.add_column(key, style="green", overflow="fold")
                    for record in raw_data[:limit]:
                        table.add_row(
                            *[
                                str(record.get(key, ""))[:200]
                                for key in raw_data[0].keys()
                            ]
                        )
                    console.print(table)
                else:
                    console.print("[yellow]No results returned[/yellow]")

                # Show Phase 1 JSON output
                if show_json or debug:
                    import json

                    pagination_info = orchestrator._build_pagination_info(
                        cypher_query, raw_data
                    )
                    validation = orchestrator._validate_result(result, question)
                    # Get token comparison from orchestrator
                    token_comparison = getattr(
                        orchestrator, "_last_token_comparison", None
                    )
                    phase1_json = orchestrator._prepare_phase1_json_output(
                        question=question,
                        raw_data=raw_data,
                        cypher_query=cypher_query,
                        pagination_info=pagination_info,
                        validation=validation,
                        token_comparison=token_comparison,
                    )
                    console.print()
                    console.print("[bold yellow]Phase 1 JSON Output:[/bold yellow]")
                    console.print(json.dumps(phase1_json, indent=2))

                    # Show cost even if zero
                    cost = (
                        result.llm_cost_usd if result.llm_cost_usd is not None else 0.0
                    )
                    console.print()
                    console.print(f"[dim]Phase 1 Cost: ${cost:.6f}[/dim]")

                return

            # If --phase2, run Phase 2 only (skip Phase 1)
            if phase2:
                if debug:
                    orchestrator.debug_formatter.phase(
                        "Answer Enhancement",
                        "Question -> LLM (Direct Answer, No Database Query)",
                    )
                    orchestrator.debug_formatter.data("Question", question)
                    console.print()

                # --phase2 mode: User explicitly wants LLM-only answer (for questions not suitable for CLAIRE-KG)
                # Allow LLM to answer, but warn that it's not using CLAIRE-KG database
                if debug:
                    orchestrator.debug_formatter.info(
                        "WARNING: Phase 2 Only mode: Answering without CLAIRE-KG database context"
                    )

                # Call Phase 2 with empty data (user explicitly requested LLM-only)
                enhanced_answer = orchestrator._enhance_answer(question, [], "")

                if debug:
                    console.print()
                    console.print("[green]OK: Phase 2 Complete[/green]")
                    console.print()

                console.print(
                    Panel(
                        enhanced_answer,
                        title="Enhanced Answer" if debug else None,
                        border_style="green",
                        box=box.ASCII,
                        width=88,
                    )
                )

                if debug:
                    console.print()
                    # Cost will be shown in the debug output from _enhance_answer

                return

            # Normal mode: Process the question through Phase 1 -> Phase 2
            # If --eval is set, skip early rejection to allow Phase 2+3 to run even without datasets
            skip_rejection = eval  # Allow Phase 2+3 even if no datasets detected
            result = orchestrator.process_question(
                question, limit=limit, skip_early_rejection=skip_rejection
            )

            # Check if this was a help request (success=True but no cypher_query)
            if (
                result.success
                and not result.cypher_query
                and result.enhanced_answer
                and "CLAIRE-KG Help" in result.enhanced_answer
            ):
                # Help request - display the help message
                console.print(result.enhanced_answer)
                return

            # Check if this was an early rejection (no datasets detected)
            is_early_rejection = (
                result.success
                and not result.cypher_query
                and result.enhanced_answer
                and (
                    "I couldn't identify" in result.enhanced_answer
                    or "doesn't appear to be about" in result.enhanced_answer
                )
            )

            if result.success:
                # Debug mode shows all details
                if debug:
                    console.print("[green]OK: Query completed successfully[/green]")
                    console.print(
                        f"[cyan]Execution Time: {result.execution_time:.2f}s[/cyan]"
                    )
                    console.print()

                # Show Phase 1 JSON if requested
                if show_json or debug:
                    import json

                    console.print("[bold yellow]Phase 1 JSON Output:[/bold yellow]")
                    pagination_info = orchestrator._build_pagination_info(
                        result.cypher_query, result.raw_data
                    )
                    validation = orchestrator._validate_result(result, result.question)
                    # Get token comparison from orchestrator
                    token_comparison = getattr(
                        orchestrator, "_last_token_comparison", None
                    )
                    phase1_json = orchestrator._prepare_phase1_json_output(
                        question=result.question,
                        raw_data=result.raw_data,
                        cypher_query=result.cypher_query,
                        pagination_info=pagination_info,
                        validation=validation,
                        token_comparison=token_comparison,
                        evaluation_result=result.evaluation_result,
                        evaluation_cost=result.evaluation_cost,
                    )
                    console.print(json.dumps(phase1_json, indent=2))
                    console.print()

                # Save evaluation results if requested (creates .json, .md, and _debug.txt)
                # Note: We save even if evaluation_result is None (evaluation may have failed)
                # The debug file will contain information about why evaluation didn't run
                if save_results:
                    from .evaluator import (
                        save_evaluation_to_json,
                        save_evaluation_to_markdown,
                        save_no_evaluation_placeholder,
                    )
                    from pathlib import Path
                    import json

                    # Standardize output directory
                    output_dir = Path(os.getenv("CLAIRE_OUTPUT_DIR", "tests/outputs"))
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Generate base filename (remove extension if provided, use just the name)
                    base_path = Path(save_results)
                    if base_path.suffix:
                        base_name = base_path.stem
                    else:
                        base_name = base_path.name

                    # Helper function to find next available filename
                    def get_next_filename(base_name: str, extension: str) -> Path:
                        """Get next available filename, adding _1, _2, etc. if file exists."""
                        counter = 0
                        while True:
                            if counter == 0:
                                filename = output_dir / f"{base_name}{extension}"
                            else:
                                filename = (
                                    output_dir / f"{base_name}_{counter}{extension}"
                                )
                            if not filename.exists():
                                return filename
                            counter += 1

                    # 1. Save JSON file (with overwrite protection) – always write so pipeline gets 3 files
                    json_path = get_next_filename(base_name, ".json")
                    if result.evaluation_result:
                        save_evaluation_to_json(
                            result.evaluation_result,
                            str(json_path),
                            question=result.question,
                            answer=result.enhanced_answer,
                        )
                        console.print(f"💾 Evaluation JSON saved to: {json_path}")

                        # 2. Save Markdown file (with overwrite protection)
                        md_path = get_next_filename(base_name, ".md")
                        save_evaluation_to_markdown(
                            result.evaluation_result,
                            str(md_path),
                            question=result.question,
                            answer=result.enhanced_answer,
                        )
                        console.print(f"📄 Evaluation Markdown saved to: {md_path}")
                    else:
                        md_path = get_next_filename(base_name, ".md")
                        save_no_evaluation_placeholder(
                            str(json_path),
                            str(md_path),
                            question=result.question,
                            answer=result.enhanced_answer,
                        )
                        console.print(
                            f"💾 Placeholder JSON saved to: {json_path} (no evaluation result)"
                        )
                        console.print(f"📄 Placeholder Markdown saved to: {md_path}")
                        console.print(
                            "[yellow]   Evaluation was not run or failed; placeholder files written so pipeline receives 3 files.[/yellow]"
                        )

                    # 3. Save debug output with full Phase 1 JSON
                    debug_path = get_next_filename(base_name, "_debug.txt")

                    # Build comprehensive debug file with Phase 1 JSON and all debug output
                    with open(debug_path, "w", encoding="utf-8") as f:
                        f.write("=" * 80 + "\n")
                        f.write("CLAIRE-KG EVALUATION DEBUG OUTPUT\n")
                        f.write(
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        )
                        f.write("=" * 80 + "\n\n")

                        f.write(f"Question: {result.question}\n")
                        f.write(f"Limit: {limit}\n")
                        f.write(f"Execution Time: {result.execution_time:.2f}s\n")
                        if result.llm_cost_usd:
                            f.write(f"Total Cost: ${result.llm_cost_usd:.6f}\n")
                        if result.evaluation_cost:
                            f.write(f"Evaluation Cost: ${result.evaluation_cost:.6f}\n")
                        if result.llm_tokens_used:
                            f.write(f"Tokens Used: {result.llm_tokens_used}\n")
                        f.write("\n" + "=" * 80 + "\n\n")

                        # Add detailed debug output from file if available (comes first, includes all phases)
                        if (
                            hasattr(orchestrator, "debug_formatter")
                            and orchestrator.debug_formatter
                        ):
                            orchestrator.debug_formatter.close()

                            # Try to read from temp debug file (contains all debug output)
                            if actual_debug_file and Path(actual_debug_file).exists():
                                f.write("=" * 80 + "\n")
                                f.write("DETAILED DEBUG OUTPUT (All Phases)\n")
                                f.write("=" * 80 + "\n\n")
                                with open(
                                    actual_debug_file, "r", encoding="utf-8"
                                ) as debug_f:
                                    f.write(debug_f.read())
                                # Clean up temp file
                                try:
                                    Path(actual_debug_file).unlink()
                                except:
                                    pass
                                f.write("\n" + "=" * 80 + "\n\n")
                            elif (
                                orchestrator.debug_formatter.debug_file
                                and Path(
                                    orchestrator.debug_formatter.debug_file
                                ).exists()
                            ):
                                f.write("=" * 80 + "\n")
                                f.write("DETAILED DEBUG OUTPUT (All Phases)\n")
                                f.write("=" * 80 + "\n\n")
                                with open(
                                    orchestrator.debug_formatter.debug_file,
                                    "r",
                                    encoding="utf-8",
                                ) as debug_f:
                                    f.write(debug_f.read())
                                f.write("\n" + "=" * 80 + "\n\n")

                        # Phase 1 JSON Output (full details - complete structure)
                        f.write("=" * 80 + "\n")
                        f.write("PHASE 1: QUERY GENERATION AND EXECUTION\n")
                        f.write("=" * 80 + "\n\n")
                        if result.cypher_query:
                            f.write(f"Cypher Query:\n{result.cypher_query}\n\n")

                        # Get full Phase 1 JSON (complete structure with all metadata)
                        pagination_info = orchestrator._build_pagination_info(
                            result.cypher_query, result.raw_data
                        )
                        validation = orchestrator._validate_result(
                            result, result.question
                        )
                        token_comparison = getattr(
                            orchestrator, "_last_token_comparison", None
                        )
                        phase1_json = orchestrator._prepare_phase1_json_output(
                            question=result.question,
                            raw_data=result.raw_data,
                            cypher_query=result.cypher_query,
                            pagination_info=pagination_info,
                            validation=validation,
                            token_comparison=token_comparison,
                            evaluation_result=result.evaluation_result,
                            evaluation_cost=result.evaluation_cost,
                        )
                        f.write("Phase 1 JSON Output (Complete):\n")
                        f.write(json.dumps(phase1_json, indent=2, ensure_ascii=False))
                        f.write("\n\n")

                        f.write("=" * 80 + "\n")
                        f.write("PHASE 2: ANSWER ENHANCEMENT\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Enhanced Answer:\n{result.enhanced_answer}\n\n")

                        f.write("=" * 80 + "\n")
                        f.write("PHASE 3: DEEPEVAL EVALUATION\n")
                        f.write("=" * 80 + "\n\n")

                        # Add evaluation results (complete structure)
                        if result.evaluation_result:
                            eval_dict = result.evaluation_result.to_dict()
                            f.write("Evaluation Results (Complete):\n")
                            f.write(json.dumps(eval_dict, indent=2, ensure_ascii=False))
                            f.write("\n\n")
                        else:
                            f.write("Evaluation Results: NOT AVAILABLE\n")
                            f.write("\nPossible reasons:\n")
                            f.write(
                                "- Evaluation was not enabled (--eval flag not set)\n"
                            )
                            f.write(
                                "- Evaluation failed (check error messages above)\n"
                            )
                            f.write("- Evaluation timed out or encountered an error\n")
                            f.write("\n")

                    console.print(f"📄 Debug output saved to: {debug_path}")
                    console.print(f"📁 All files saved to: {output_dir}")

                # Show answer - but don't call it "Phase 2" if it was an early rejection
                if not is_early_rejection:
                    # Normal Phase 2 enhanced answer
                    if debug:
                        console.print(
                            "[bold green]Phase 2 Enhanced Answer (with citations):[/bold green]"
                        )
                        console.print()

                    # Check if this is a warning message (no results) and format differently
                    is_warning = (
                        "⚠️" in result.enhanced_answer
                        or "No database results" in result.enhanced_answer
                    )
                    border_style = "yellow" if is_warning else "green"

                    console.print(
                        Panel(
                            result.enhanced_answer,
                            title="Enhanced Answer" if debug else None,
                            border_style=border_style,
                            box=box.ASCII,
                            width=88,
                        )
                    )
                else:
                    # Early rejection message (not Phase 2)
                    if debug:
                        console.print(
                            "[bold yellow]INFO: Early Rejection (No datasets detected):[/bold yellow]"
                        )
                        console.print()
                    console.print(
                        Panel(
                            result.enhanced_answer,
                            title="Message" if debug else None,
                            border_style="yellow",
                            box=box.ASCII,
                            width=88,
                        )
                    )

                # Show summary only in debug mode
                if debug:
                    console.print()
                    console.print(f"[dim]Found {len(result.raw_data)} result(s)[/dim]")
                    # Show cost even if zero
                    cost = (
                        result.llm_cost_usd if result.llm_cost_usd is not None else 0.0
                    )
                    console.print(f"[dim]LLM Cost: ${cost:.6f}[/dim]")
                    console.print(
                        f"[dim]Execution Time: {result.execution_time:.2f}s[/dim]"
                    )
            else:
                # Not result.success: either soft-fail (Phase 1 no usable results) or hard error
                phase1_no_results = getattr(result, "phase1_no_results", False)
                if phase1_no_results:
                    # Soft-fail: Phase 1 ran but no usable results. Show answer, honor --save, exit 0.
                    console.print(
                        Panel(
                            result.enhanced_answer,
                            title="No results (evaluation can still score this)",
                            border_style="yellow",
                            box=box.ASCII,
                            width=88,
                        )
                    )
                    if save_results:
                        from pathlib import Path
                        import json as json_module

                        from .evaluator import save_no_evaluation_placeholder

                        output_dir = Path(
                            os.getenv("CLAIRE_OUTPUT_DIR", "tests/outputs")
                        )
                        output_dir.mkdir(parents=True, exist_ok=True)
                        base_path = Path(save_results)
                        base_name = (
                            base_path.stem if base_path.suffix else base_path.name
                        )

                        def get_next_filename(base_name: str, extension: str) -> Path:
                            """Return next available path base_name.ext or base_name_N.ext that does not exist."""
                            counter = 0
                            while True:
                                filename = output_dir / (
                                    f"{base_name}{extension}"
                                    if counter == 0
                                    else f"{base_name}_{counter}{extension}"
                                )
                                if not filename.exists():
                                    return filename
                                counter += 1

                        json_path = get_next_filename(base_name, ".json")
                        md_path = get_next_filename(base_name, ".md")
                        save_no_evaluation_placeholder(
                            str(json_path),
                            str(md_path),
                            question=result.question,
                            answer=result.enhanced_answer,
                        )
                        console.print(
                            f"💾 Placeholder JSON saved to: {json_path} (Phase 1 no usable results)"
                        )
                        console.print(f"📄 Markdown saved to: {md_path}")

                        debug_path = get_next_filename(base_name, "_debug.txt")
                        with open(debug_path, "w", encoding="utf-8") as f:
                            f.write("=" * 80 + "\n")
                            f.write("CLAIRE-KG (Phase 1 no usable results)\n")
                            f.write(
                                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            )
                            f.write("=" * 80 + "\n\n")
                            f.write(f"Question: {result.question}\n")
                            f.write(f"Execution Time: {result.execution_time:.2f}s\n")
                            fallbacks = getattr(result, "fallbacks_attempted", None)
                            if fallbacks:
                                f.write(
                                    f"Fallbacks attempted: {', '.join(fallbacks)}\n"
                                )
                            else:
                                f.write("Fallbacks attempted: (none)\n")
                            if result.cypher_query:
                                f.write(f"\nCypher Query:\n{result.cypher_query}\n\n")
                            pagination_info = orchestrator._build_pagination_info(
                                result.cypher_query, result.raw_data
                            )
                            validation = orchestrator._validate_result(
                                result, result.question
                            )
                            token_comparison = getattr(
                                orchestrator, "_last_token_comparison", None
                            )
                            phase1_json = orchestrator._prepare_phase1_json_output(
                                question=result.question,
                                raw_data=result.raw_data,
                                cypher_query=result.cypher_query,
                                pagination_info=pagination_info,
                                validation=validation,
                                token_comparison=token_comparison,
                                evaluation_result=None,
                                evaluation_cost=None,
                            )
                            f.write("Phase 1 JSON:\n")
                            f.write(
                                json_module.dumps(
                                    phase1_json, indent=2, ensure_ascii=False
                                )
                            )
                            f.write("\n\nEnhanced Answer:\n")
                            f.write(result.enhanced_answer)
                            f.write("\n")
                        console.print(f"📄 Debug output saved to: {debug_path}")
                        console.print(f"📁 All files saved to: {output_dir}")
                    return
                console.print(
                    f"[red]ERROR: Query failed: {result.error or 'Unknown error'}[/red]"
                )
                if debug and result.error:
                    console.print(f"[red]Error details: {result.error}[/red]")
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        if debug:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)
    finally:
        if "db" in locals():
            db.close()
        if debug_file and "orchestrator" in locals():
            console.print(f"[green]OK: Debug output saved to {debug_file}[/green]")

        # Clean up temporary debug file if it was created for --save
        # (Already cleaned up in the save_results block, but double-check here)
        if (
            save_results
            and "actual_debug_file" in locals()
            and actual_debug_file
            and actual_debug_file != debug_file
        ):
            try:
                if Path(actual_debug_file).exists():
                    Path(actual_debug_file).unlink()
            except:
                pass


# Shared Rich console for all commands (box style set per Table/Panel)
console = Console()


# --- evaluate: Phase 3 DeepEval (Relevancy, Faithfulness, optional GEval) ---
@app.command()
def evaluate(
    question: str = typer.Option(
        ..., "--question", "-q", help="The question that was asked"
    ),
    answer: str = typer.Option(..., "--answer", "-a", help="The answer to evaluate"),
    context: Optional[List[str]] = typer.Option(
        None,
        "--context",
        "-c",
        help="Context string(s) used to produce the answer. Can be specified multiple times. For no retrieval (e.g. Direct LLM), omit or pass a placeholder.",
    ),
    context_file: Optional[str] = typer.Option(
        None,
        "--context-file",
        help="Path to a file containing context (one context item per line, or JSON array)",
    ),
    geval: bool = typer.Option(
        True,
        "--geval/--no-geval",
        help="Enable/disable GEval metric (enabled by default)",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Save evaluation results to files (creates <name>.json and <name>.md in tests/outputs/)",
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Debug mode with detailed output"
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON to stdout (for programmatic use)",
    ),
):
    """
    Evaluate an answer using DeepEval metrics (Phase 3 only).
    
    This command runs CLAIRE-KG's Phase 3 evaluation on externally-provided
    question/answer/context, without running Phase 1 (Cypher generation) or
    Phase 2 (answer enhancement).
    
    Use this to evaluate answers from other systems (DirectLLM, RAG, etc.)
    with the same metrics and output format as CLAIRE-KG's internal evaluation.
    
    EXAMPLES:
    
    # Evaluate a DirectLLM answer (no retrieval context)
    claire_kg evaluate -q "What is CVE-2024-1234?" -a "CVE-2024-1234 is a vulnerability..."
    
    # Evaluate a RAG answer with context
    claire_kg evaluate -q "What is CWE-79?" -a "CWE-79 is XSS..." \\
        -c "CWE-79: Cross-site Scripting (XSS)" \\
        -c "Description: The software does not neutralize user input..."
    
    # Read context from a file
    claire_kg evaluate -q "..." -a "..." --context-file chunks.txt
    
    # Save results to files
    claire_kg evaluate -q "..." -a "..." --save HV01_deepeval_direct
    
    # Output JSON for programmatic use
    claire_kg evaluate -q "..." -a "..." --json
    """
    from .evaluator import (
        QueryEvaluator,
        save_evaluation_to_json,
        save_evaluation_to_markdown,
    )
    import json as json_module

    # Build context list
    context_list: List[str] = []

    # Add context from --context options
    if context:
        context_list.extend(context)

    # Add context from --context-file if provided
    if context_file:
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                file_content = f.read().strip()
                # Try to parse as JSON array first
                try:
                    parsed = json_module.loads(file_content)
                    if isinstance(parsed, list):
                        context_list.extend([str(item) for item in parsed])
                    else:
                        # Single item, add as-is
                        context_list.append(str(parsed))
                except json_module.JSONDecodeError:
                    # Not JSON, treat as one context item per non-empty line
                    for line in file_content.split("\n"):
                        line = line.strip()
                        if line:
                            context_list.append(line)
        except FileNotFoundError:
            console.print(f"[red]ERROR: Context file not found: {context_file}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]ERROR: Failed to read context file: {e}[/red]")
            raise typer.Exit(1)

    # If no context provided, use placeholder for "no retrieval"
    if not context_list:
        context_list = ["No retrieval context; answer from model knowledge."]
        if debug:
            console.print(
                "[yellow]No context provided - using placeholder for no-retrieval evaluation[/yellow]"
            )

    if not output_json:
        console.print(
            "[bold blue]CLAIRE-KG Phase 3 Evaluation (Evaluate Only)[/bold blue]"
        )
        console.print()
        console.print(
            f"[cyan]Question:[/cyan] {question[:100]}{'...' if len(question) > 100 else ''}"
        )
        console.print(
            f"[cyan]Answer:[/cyan] {answer[:100]}{'...' if len(answer) > 100 else ''}"
        )
        console.print(f"[cyan]Context Items:[/cyan] {len(context_list)}")
        if debug:
            for i, ctx in enumerate(context_list[:3]):
                console.print(
                    f"  [dim]{i+1}. {ctx[:80]}{'...' if len(ctx) > 80 else ''}[/dim]"
                )
            if len(context_list) > 3:
                console.print(f"  [dim]... and {len(context_list) - 3} more[/dim]")
        console.print()

    try:
        # Enable GEval based on flag
        os.environ["GEVAL_ENABLED"] = "true" if geval else "false"

        # Initialize evaluator
        evaluator = QueryEvaluator(
            enabled=True,
            lazy_init=True,
            debug=debug,
            enable_geval=geval,
            strict_mode=False,
        )

        if not output_json:
            console.print("[green]Running DeepEval evaluation...[/green]")
            console.print()

        # Run evaluation
        evaluation_result, evaluation_cost = evaluator.evaluate_external(
            question=question,
            answer=answer,
            context=context_list,
        )

        # Display results
        if output_json:
            # JSON output for programmatic use
            result_dict = evaluation_result.to_dict()
            result_dict["question"] = question
            result_dict["answer"] = answer
            result_dict["context"] = context_list
            result_dict["evaluation_cost"] = evaluation_cost
            print(json_module.dumps(result_dict, indent=2, ensure_ascii=False))
        else:
            # Human-readable output
            console.print("[bold]Evaluation Results:[/bold]")
            console.print()

            # Overall status
            status_color = "green" if evaluation_result.passed else "red"
            status_text = "PASSED" if evaluation_result.passed else "FAILED"
            console.print(f"[{status_color}]Status: {status_text}[/{status_color}]")
            console.print(f"Overall Score: [cyan]{evaluation_result.score:.3f}[/cyan]")
            console.print()

            # Individual metrics
            console.print("[bold]Metrics:[/bold]")
            thresholds = {"relevancy": 0.65, "faithfulness": 0.7, "geval": 0.7}
            for metric_name, score in evaluation_result.metrics.items():
                threshold = thresholds.get(metric_name, 0.5)
                passed = score >= threshold
                color = "green" if passed else "red"
                status = "PASS" if passed else "FAIL"
                console.print(
                    f"  {metric_name.capitalize()}: [{color}]{score:.3f}[/{color}] (threshold: {threshold}) [{color}]{status}[/{color}]"
                )

            # Metric status for timeouts/errors
            for metric_name, status in evaluation_result.metric_status.items():
                if (
                    status not in ("success", "not_enabled")
                    and metric_name not in evaluation_result.metrics
                ):
                    console.print(
                        f"  {metric_name.capitalize()}: [yellow]{status}[/yellow]"
                    )

            console.print()

            # Issues
            if evaluation_result.issues:
                console.print("[bold]Issues:[/bold]")
                for issue in evaluation_result.issues:
                    console.print(f"  [yellow]⚠ {issue}[/yellow]")
                console.print()

            # Limited context note
            if evaluation_result.limited_context:
                console.print(
                    "[dim]Note: Faithfulness capped due to no retrieval context (answer from model knowledge)[/dim]"
                )
                console.print()

            # Cost
            if evaluation_cost > 0:
                console.print(f"[dim]Evaluation cost: ${evaluation_cost:.4f}[/dim]")

        # Save results if requested
        if save:
            output_dir = Path(os.getenv("CLAIRE_OUTPUT_DIR", "tests/outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find next available filename
            base_name = Path(save).stem if Path(save).suffix else save

            def get_next_filename(base: str, ext: str) -> Path:
                """Return next available path base+ext or base_N+ext that does not exist."""
                counter = 0
                while True:
                    if counter == 0:
                        filename = output_dir / f"{base}{ext}"
                    else:
                        filename = output_dir / f"{base}_{counter}{ext}"
                    if not filename.exists():
                        return filename
                    counter += 1

            json_path = get_next_filename(base_name, ".json")
            md_path = get_next_filename(base_name, ".md")

            # Save JSON
            save_evaluation_to_json(
                evaluation_result, str(json_path), question=question, answer=answer
            )

            # Save Markdown
            save_evaluation_to_markdown(
                evaluation_result, str(md_path), question=question, answer=answer
            )

            if not output_json:
                console.print()
                console.print(f"[green]Results saved to:[/green]")
                console.print(f"  [cyan]{json_path}[/cyan]")
                console.print(f"  [cyan]{md_path}[/cyan]")

        # Exit with appropriate code
        if not evaluation_result.passed:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        if output_json:
            error_result = {"error": str(e), "passed": False, "score": 0.0}
            print(json_module.dumps(error_result, indent=2))
        else:
            console.print(f"[red]ERROR: Evaluation failed: {e}[/red]")
        raise typer.Exit(1)


# -----------------------------------------------------------------------------
# Setup commands: ingest, crosswalk, embeddings, clean, list_crosswalks, etc.
# -----------------------------------------------------------------------------


@setup_app.command()
def ingest(
    dataset: str = typer.Argument(
        ..., help="Dataset to ingest (capec, cwe, attack, cve, nice, dcwf)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be ingested without actually doing it"
    ),
    batch_size: int = typer.Option(
        1000, "--batch-size", help="Batch size for APOC ingestion"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Ingest a cybersecurity dataset into the knowledge graph."""

    console.print("[bold blue]CLAIRE-KG Dataset Ingestion[/bold blue]")
    console.print(f"Dataset: [green]{dataset}[/green]")
    console.print(f"Dry Run: [yellow]{dry_run}[/yellow]")
    console.print(f"Batch Size: [cyan]{batch_size}[/cyan]")

    # Allowed dataset names (must match ingest_dataset() in ingest.py)
    valid_datasets = ["capec", "cwe", "attack", "cve", "nice", "dcwf"]
    if dataset not in valid_datasets:
        console.print(
            f"[red]Error: Invalid dataset '{dataset}'. Valid options: {', '.join(valid_datasets)}[/red]"
        )
        raise typer.Exit(1)

    # Data is in Docker volume at /import
    console.print("[green]Using data from Docker volume at /import[/green]")

    try:
        # Initialize database connection
        db = Neo4jConnection()

        # Initialize ingester
        ingester = DatasetIngester(db, batch_size=batch_size, verbose=verbose)

        # Execute ingestion
        if dry_run:
            console.print(f"[yellow]DRY RUN: Would ingest {dataset} dataset[/yellow]")
            # TODO: Show what would be ingested
        else:
            console.print(f"[green]Starting {dataset} ingestion...[/green]")
            result = ingester.ingest_dataset(dataset)

            if result.success:
                console.print(f"[green]OK: Successfully ingested {dataset}[/green]")
                console.print(f"Nodes created: {result.nodes_created}")
                console.print(f"Relationships created: {result.relationships_created}")
                # Schema changed; invalidate caches so Cypher generator uses new node/rel counts
                invalidate_all_schema_caches()
                console.print("[dim]Schema cache invalidated[/dim]")
            else:
                console.print(
                    f"[red]ERROR: Failed to ingest {dataset}: {result.error}[/red]"
                )
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@setup_app.command()
def crosswalk(
    crosswalk_type: str = typer.Argument(
        ...,
        help="Crosswalk type (capec-attack, capec-relationships, cve-cwe, cve-assets, dcwf-nice)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be crosswalked without actually doing it",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Create crosswalk relationships between datasets."""

    console.print("[bold blue]CLAIRE-KG Crosswalk Creation[/bold blue]")
    console.print(f"Crosswalk: [green]{crosswalk_type}[/green]")
    console.print(f"Dry Run: [yellow]{dry_run}[/yellow]")

    # Allowed crosswalk names (must match ingest.py crosswalk logic)
    valid_crosswalks = [
        "capec-attack",
        "capec-relationships",
        "capec-mitigations",
        "attack-mitigations",
        "workrole-attack",
        "workrole-capec",
        "cve-attack",
        "cve-capec",
        "cwe-mitigations",
        "cwe-categories",
        "cve-cwe",
        "cve-assets",
        "dcwf-nice",
        "dcwf-cross-domain",
        "nice-cross-domain",
    ]
    if crosswalk_type not in valid_crosswalks:
        console.print(
            f"[red]Error: Invalid crosswalk '{crosswalk_type}'. Valid options: {', '.join(valid_crosswalks)}[/red]"
        )
        raise typer.Exit(1)

    try:
        # Initialize database connection
        db = Neo4jConnection()

        # Initialize ingester
        ingester = DatasetIngester(db, verbose=verbose)

        # Execute crosswalk
        if dry_run:
            console.print(
                f"[yellow]DRY RUN: Would create {crosswalk_type} crosswalk[/yellow]"
            )
        else:
            console.print(f"[green]Starting {crosswalk_type} crosswalk...[/green]")
            result = ingester.create_crosswalk(crosswalk_type)

            if result.success:
                console.print(
                    f"[green]OK: Successfully created {crosswalk_type} crosswalk[/green]"
                )
                console.print(f"Relationships created: {result.relationships_created}")
                # Invalidate schema cache since relationships may have changed
                invalidate_all_schema_caches()
                console.print("[dim]Schema cache invalidated[/dim]")
            else:
                console.print(
                    f"[red]ERROR: Failed to create {crosswalk_type} crosswalk: {result.error}[/red]"
                )
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@setup_app.command()
def embeddings(
    dataset: Optional[str] = typer.Argument(
        None, help="Specific dataset to generate embeddings for (optional)"
    ),
    model: str = typer.Option(
        "text-embedding-3-small",
        "--model",
        help="OpenAI embedding model to use (e.g., 'text-embedding-3-small', 'text-embedding-3-large')",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be embedded without actually doing it"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Generate vector embeddings using OpenAI embeddings API.

    Requires OPENAI_API_KEY environment variable.
    """

    console.print("[bold blue]CLAIRE-KG Embeddings Generation[/bold blue]")
    console.print(f"Dataset: [green]{dataset or 'all'}[/green]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Method: [yellow]OpenAI API[/yellow]")
    console.print(f"Dry Run: [yellow]{dry_run}[/yellow]")

    try:
        # Initialize database connection
        db = Neo4jConnection()

        # Initialize ingester
        ingester = DatasetIngester(db, verbose=verbose)

        # Execute embeddings generation
        if dry_run:
            console.print(
                f"[yellow]DRY RUN: Would generate embeddings for {dataset or 'all datasets'}[/yellow]"
            )
        else:
            console.print("[green]Starting embeddings generation...[/green]")
            result = ingester.generate_embeddings(dataset, model)

            if result.success:
                console.print("[green]OK: Successfully generated embeddings[/green]")
                console.print(f"Nodes processed: {result.nodes_processed}")
            else:
                console.print(
                    f"[red]ERROR: Failed to generate embeddings: {result.error}[/red]"
                )
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show detailed debug information including isolated nodes",
    ),
):
    """Show the current status of the knowledge graph with detailed statistics."""

    console.print("[bold blue]CLAIRE-KG Status[/bold blue]")

    try:
        # Initialize database connection
        db = Neo4jConnection()

        # Get basic status information
        status_info = db.get_status()

        # Create main status table
        table = Table(title="Knowledge Graph Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")

        for metric, count in status_info.items():
            table.add_row(metric, str(count))

        console.print(table)

        # Get relationship breakdown
        console.print("\n[bold cyan]Relationship Breakdown:[/bold cyan]")
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relType, count(r) as count
        ORDER BY count DESC
        """
        rel_results = db.execute_cypher(rel_query)

        rel_table = Table()
        rel_table.add_column("Relationship Type", style="yellow")
        rel_table.add_column("Count", style="green")

        for record in rel_results:
            rel_type = record["relType"]
            count = record["count"]
            rel_table.add_row(rel_type, str(count))

        console.print(rel_table)

        # Only show isolated nodes in debug mode
        if debug:
            # Get isolated nodes (excluding "Awaiting Analysis" CVEs)
            console.print(
                "\n[bold red]Isolated Nodes (excluding 'Awaiting Analysis' CVEs):[/bold red]"
            )
            isolated_query = """
            MATCH (n)
            WHERE NOT (n)--()
            AND NOT (n:Vulnerability AND n.status = 'Awaiting Analysis')
            RETURN labels(n)[0] as nodeType, count(*) as isolatedCount
            ORDER BY isolatedCount DESC
            """
            isolated_results = db.execute_cypher(isolated_query)

            isolated_table = Table()
            isolated_table.add_column("Node Type", style="red")
            isolated_table.add_column("Isolated Count", style="red")

            total_isolated = 0
            for record in isolated_results:
                node_type = record["nodeType"] or "Unknown"
                count = record["isolatedCount"]
                total_isolated += count
                if count > 0:
                    isolated_table.add_row(node_type, str(count))

            isolated_table.add_row("TOTAL ISOLATED", str(total_isolated))

            # Calculate isolation rate
            total_nodes = status_info.get("Total Nodes", 0)
            isolation_rate = (
                (total_isolated / total_nodes * 100) if total_nodes > 0 else 0
            )
            isolated_table.add_row("ISOLATION RATE", f"{isolation_rate:.2f}%")

            console.print(isolated_table)

            # Summary
            total_rels = status_info.get("Total Relationships", 0)
            avg_rels = total_rels / total_nodes if total_nodes > 0 else 0

            console.print("\n[bold green]Graph Connectivity Summary:[/bold green]")
            console.print(f"Total Nodes: {total_nodes:,}")
            console.print(f"Total Relationships: {total_rels:,}")
            console.print(f"Isolation Rate: {isolation_rate:.2f}%")
            console.print(f"Average Relationships per Node: {avg_rels:.1f}")
        else:
            # Show basic summary without isolated nodes
            total_nodes = status_info.get("Total Nodes", 0)
            total_rels = status_info.get("Total Relationships", 0)
            avg_rels = total_rels / total_nodes if total_nodes > 0 else 0

            console.print("\n[bold green]Graph Connectivity Summary:[/bold green]")
            console.print(f" Total Nodes: {total_nodes:,}")
            console.print(f" Total Relationships: {total_rels:,}")
            console.print(f" Average Relationships per Node: {avg_rels:.1f}")
            console.print(
                "\n[yellow]TIP: Use --debug flag to see isolated nodes analysis[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@setup_app.command()
def clean():
    """Clean the database by removing all nodes and relationships."""

    console.print("[bold blue]CLAIRE-KG Database Clean[/bold blue]")
    console.print(
        "[yellow]WARNING: This will remove ALL nodes and relationships from the database![/yellow]"
    )

    try:
        # Initialize database connection
        db = Neo4jConnection()

        # Clean the database
        console.print("[green]Cleaning database...[/green]")
        db.execute_cypher("MATCH (n) DETACH DELETE n")

        # Verify it's clean
        node_count = db.execute_cypher("MATCH (n) RETURN count(n) as count")[0]["count"]
        rel_count = db.execute_cypher("MATCH ()-[r]->() RETURN count(r) as count")[0][
            "count"
        ]

        console.print("[green]OK: Database cleaned successfully[/green]")
        console.print(f"Nodes remaining: {node_count}")
        console.print(f"Relationships remaining: {rel_count}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@setup_app.command()
def list_crosswalks():
    """List all available crosswalks with descriptions."""

    console.print("[bold blue]CLAIRE-KG Available Crosswalks[/bold blue]")

    crosswalks = [
        ("capec-attack", "Connect CAPEC attack patterns to ATT&CK techniques"),
        ("capec-relationships", "Connect CAPEC attack patterns to each other"),
        ("capec-mitigations", "Connect CAPEC attack patterns to mitigations"),
        ("attack-mitigations", "Connect ATT&CK techniques to mitigations"),
        ("workrole-attack", "Connect workforce roles to ATT&CK techniques"),
        ("workrole-capec", "Connect workforce roles to CAPEC attack patterns"),
        ("cve-attack", "Connect CVEs to ATT&CK techniques via CWE and CAPEC"),
        ("cve-capec", "Connect CVEs to CAPEC attack patterns via CWE"),
        ("cwe-mitigations", "Connect CWE weaknesses to mitigations"),
        (
            "cwe-categories",
            "Create CWE categories for semantic search and natural language queries",
        ),
        ("cve-cwe", "Connect CVEs to CWE weaknesses"),
        ("cve-assets", "Connect CVEs to infrastructure assets (CPE)"),
        ("dcwf-nice", "Connect DCWF workforce to NICE framework"),
        ("dcwf-cross-domain", "Connect DCWF to other security domains"),
        ("nice-cross-domain", "Connect NICE to other security domains"),
    ]

    # Create table
    table = Table(title="Available Crosswalks")
    table.add_column("Crosswalk", style="cyan")
    table.add_column("Description", style="white")

    for crosswalk, description in crosswalks:
        table.add_row(crosswalk, description)

    console.print(table)
    console.print(
        "\n[bold green]Usage:[/bold green] python -m claire_kg.cli crosswalk <crosswalk_name>"
    )


# -----------------------------------------------------------------------------
# Test commands: debug (dataset checks), debug_help (troubleshooting)
# -----------------------------------------------------------------------------


@test_app.command()
def debug(
    dataset: str = typer.Argument(
        ..., help="Dataset to debug (capec, cwe, attack, cve, nice, dcwf)"
    ),
):
    """Debug dataset ingestion issues."""

    console.print("[bold blue]CLAIRE-KG Dataset Debug[/bold blue]")
    console.print(f"Dataset: [green]{dataset}[/green]")

    try:
        # Initialize database connection
        db = Neo4jConnection()

        # Check if data file exists and is readable
        if dataset == "attack":
            file_path = "/import/attack/enterprise-attack.json"
            console.print(f"[cyan]Checking ATTACK data file: {file_path}[/cyan]")

            # Test file access
            query = f"""
            CALL apoc.load.json('{file_path}') YIELD value
            RETURN keys(value) as keys, size(keys(value)) as key_count
            LIMIT 1
            """
            try:
                result = db.execute_cypher(query)
                if result:
                    console.print(f"[green]OK: File accessible: {result[0]}[/green]")

                    # Check objects array
                    query = f"""
                    CALL apoc.load.json('{file_path}') YIELD value
                    UNWIND value.objects as obj
                    RETURN obj.type as obj_type, count(*) as count
                    ORDER BY count DESC
                    LIMIT 10
                    """
                    result = db.execute_cypher(query)
                    console.print("[cyan]Object types in file:[/cyan]")
                    for record in result:
                        obj_type = record["obj_type"]
                        count = record["count"]
                        console.print(f"  {obj_type}: {count}")
                else:
                    console.print("[red]ERROR: File not accessible or empty[/red]")
            except Exception as e:
                console.print(f"[red]ERROR: Error accessing file: {e}[/red]")

        # Check current nodes for this dataset
        console.print(f"[cyan]Current {dataset} nodes in database:[/cyan]")
        if dataset == "attack":
            query = """
            MATCH (n)
            WHERE 'Technique' IN labels(n) OR 'Tactic' IN labels(n) OR 'SubTechnique' IN labels(n)
            RETURN labels(n)[0] as nodeType, count(n) as count
            ORDER BY count DESC
            """
        else:
            query = f"""
            MATCH (n)
            WHERE '{dataset.upper()}' IN labels(n) OR any(label IN labels(n) WHERE label CONTAINS '{dataset.upper()}')
            RETURN labels(n)[0] as nodeType, count(n) as count
            ORDER BY count DESC
            """

        result = db.execute_cypher(query)
        if result:
            for record in result:
                node_type = record["nodeType"]
                count = record["count"]
                console.print(f"  {node_type}: {count}")
        else:
            console.print(f"  No {dataset} nodes found")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@setup_app.command()
def ingest_all(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be ingested without actually doing it"
    ),
    batch_size: int = typer.Option(
        1000, "--batch-size", help="Batch size for APOC ingestion"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Ingest all cybersecurity datasets in the correct order."""

    console.print("[bold blue]CLAIRE-KG Complete Dataset Ingestion[/bold blue]")
    console.print(f"Dry Run: [yellow]{dry_run}[/yellow]")
    console.print(f"Batch Size: [cyan]{batch_size}[/cyan]")

    # Define the correct order for ingestion
    datasets = ["cve", "cwe", "capec", "attack", "nice", "dcwf"]

    console.print(
        f"[bold]Ingesting {len(datasets)} datasets in correct order...[/bold]"
    )

    total_nodes = 0
    total_relationships = 0

    for i, dataset in enumerate(datasets, 1):
        console.print(
            f"[bold cyan][{i}/{len(datasets)}] Ingesting {dataset.upper()} dataset...[/bold cyan]"
        )

        try:
            # Initialize database connection
            db = Neo4jConnection()

            # Initialize ingester
            ingester = DatasetIngester(db, batch_size=batch_size, verbose=verbose)

            # Ingest the dataset
            if dry_run:
                console.print(
                    f"[yellow]DRY RUN: Would ingest {dataset} dataset[/yellow]"
                )
                result = type(
                    "Result",
                    (),
                    {"success": True, "nodes_created": 0, "relationships_created": 0},
                )()
            else:
                result = ingester.ingest_dataset(dataset)

            if result.success:
                nodes_created = result.nodes_created or 0
                relationships_created = result.relationships_created or 0
                total_nodes += nodes_created
                total_relationships += relationships_created
                console.print(
                    f"[green]OK: {dataset.upper()}: {nodes_created:,} nodes, {relationships_created:,} relationships[/green]"
                )
            else:
                console.print(
                    f"[red]ERROR: {dataset.upper()} failed: {result.error}[/red]"
                )
                raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]ERROR: Error ingesting {dataset.upper()}: {e}[/red]")
            raise typer.Exit(1)

    console.print("[bold green]Complete ingestion finished![/bold green]")
    console.print(
        f"[green]Total: {total_nodes:,} nodes, {total_relationships:,} relationships created[/green]"
    )


@setup_app.command()
def crosswalk_all(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be crosswalked without actually doing it",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run all crosswalks in the correct order."""

    console.print("[bold blue]CLAIRE-KG Complete Crosswalk Creation[/bold blue]")
    console.print(f"Dry Run: [yellow]{dry_run}[/yellow]")

    # Define the correct order for crosswalks
    crosswalks = [
        "cve-cwe",
        "cwe-categories",
        "cve-attack",
        "cve-capec",
        "cve-assets",
        "capec-attack",
        "capec-relationships",
        "dcwf-nice",
        "workrole-attack",
        "workrole-capec",
        "cve-attack",
        "dcwf-cross-domain",
        "nice-cross-domain",
        "capec-mitigations",
        "attack-mitigations",
        "cwe-mitigations",
    ]

    console.print(
        f"[bold]Running {len(crosswalks)} crosswalks in correct order...[/bold]"
    )

    total_relationships = 0

    for i, crosswalk in enumerate(crosswalks, 1):
        console.print(
            f"[bold cyan][{i}/{len(crosswalks)}] Running {crosswalk} crosswalk...[/bold cyan]"
        )

        try:
            # Initialize database connection
            db = Neo4jConnection()

            # Initialize ingester
            ingester = DatasetIngester(db, verbose=verbose)

            # Run the crosswalk
            if dry_run:
                console.print(
                    f"[yellow]DRY RUN: Would run {crosswalk} crosswalk[/yellow]"
                )
                result = type(
                    "Result", (), {"success": True, "relationships_created": 0}
                )()
            else:
                result = ingester.create_crosswalk(crosswalk)

            if result.success:
                relationships_created = result.relationships_created or 0
                total_relationships += relationships_created
                console.print(
                    f"[green]OK: {crosswalk}: {relationships_created:,} relationships created[/green]"
                )
            else:
                console.print(f"[red]ERROR: {crosswalk} failed: {result.error}[/red]")
                raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]ERROR: Error running {crosswalk}: {e}[/red]")
            raise typer.Exit(1)

    console.print("[bold green]Complete crosswalk finished![/bold green]")
    console.print(
        f"[green]Total: {total_relationships:,} relationships created[/green]"
    )


@setup_app.command()
def rag_cve(
    limit: int = typer.Option(1000, "--limit", help="Number of CVEs to process"),
    batch_size: int = typer.Option(
        100, "--batch-size", help="Batch size for embedding storage"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Generate and store vector embeddings for CVE descriptions."""

    console.print("[bold blue]CLAIRE-KG RAG CVE Vector Generation[/bold blue]")
    console.print(f"Limit: [cyan]{limit}[/cyan]")
    console.print(f"Batch Size: [cyan]{batch_size}[/cyan]")

    try:
        console.print(
            "[red]ERROR: This command is deprecated - RAGSystem has been archived[/red]"
        )
        console.print(
            "[yellow]TIP: RAG functionality is not currently available[/yellow]"
        )
        raise typer.Exit(1)

        # Archived - RAGSystem no longer available
        # rag = RAGSystem()

        console.print("[green]OK: Connected to Neo4j database[/green]")

        # Initialize with CVE data
        console.print(f"[green]Initializing RAG system with {limit} CVEs...[/green]")

        if rag.initialize(limit=limit):
            console.print("[green]OK: RAG system initialized successfully[/green]")
            console.print(f"[green] Processed {len(rag.cve_data)} CVE records[/green]")
            console.print("[green]Stored embeddings in Neo4j database[/green]")

            # Show summary
            console.print("\n[bold green] Summary:[/bold green]")
            console.print(f"   CVEs processed: [cyan]{len(rag.cve_data)}[/cyan]")
            console.print("   Embedding dimensions: [cyan]2000[/cyan]")
            console.print("  Storage: [cyan]Neo4j database[/cyan]")
            console.print("  Status: [green]Ready for search[/green]")

        else:
            console.print("[red]ERROR: Failed to initialize RAG system[/red]")
            raise typer.Exit(1)

        console.print(
            "\n[bold green]OK: CVE vector generation completed successfully![/bold green]"
        )
        console.print(
            "[yellow]TIP: Use 'rag-search' command to search the embeddings[/yellow]"
        )

    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        raise typer.Exit(1)
    finally:
        if "rag" in locals():
            rag.close()


# -----------------------------------------------------------------------------
# Query subcommand group: rag (archived), legacy, langchain
# -----------------------------------------------------------------------------


@query_app.command()
def rag(
    query: str = typer.Argument(..., help="Search query for CVE vulnerabilities"),
    top_k: int = typer.Option(10, "--top-k", help="Number of results to return"),
    limit: int = typer.Option(1000, "--limit", help="Number of CVEs to search in"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Search for CVE vulnerabilities using RAG (Retrieval Augmented Generation)."""

    console.print("[bold blue]CLAIRE-KG RAG Search[/bold blue]")
    console.print(f"Query: [green]{query}[/green]")
    console.print(f"Top K: [cyan]{top_k}[/cyan]")
    console.print(f"Limit: [cyan]{limit}[/cyan]")

    try:
        console.print(
            "[red]ERROR: This command is deprecated - RAGSystem has been archived[/red]"
        )
        console.print(
            "[yellow]TIP: RAG functionality is not currently available[/yellow]"
        )
        raise typer.Exit(1)

        # Archived - RAGSystem no longer available
        # rag = RAGSystem()

        # Load CVE data
        console.print("[green] Loading CVE data...[/green]")
        rag.cve_data = rag.extract_cve_data(limit=limit)

        if not rag.cve_data:
            console.print("[red]ERROR: No CVE data found[/red]")
            raise typer.Exit(1)

        console.print(f"[green]OK: Loaded {len(rag.cve_data)} CVE records[/green]")

        # Regenerate embeddings for search
        console.print("[green]Preparing embeddings for search...[/green]")
        descriptions = [item["v.descriptions"] for item in rag.cve_data]

        from sklearn.feature_extraction.text import TfidfVectorizer

        # Recreate the same vectorizer used for storage
        rag.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
        )

        # Generate embeddings
        rag.embeddings = rag.vectorizer.fit_transform(descriptions)
        rag.is_initialized = True

        console.print("[green]OK: RAG system ready for search[/green]")

        # Perform search
        console.print(f"\n[bold cyan] Searching for: '{query}'[/bold cyan]")
        results = rag.search_similar(query, top_k=top_k)

        if not results:
            console.print(f"[yellow]No results found for '{query}'[/yellow]")
            return

        # Display results
        console.print(f"\n[bold green] Found {len(results)} results:[/bold green]")

        # Create results table
        table = Table(title=f"RAG Search Results for '{query}'")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Score", style="green", width=6)
        table.add_column("CVE ID", style="yellow", width=12)
        table.add_column("Severity", style="red", width=8)
        table.add_column("Description", style="white", width=60)

        for i, result in enumerate(results):
            # Determine result quality
            if result.similarity_score > 0.3:
                score_style = "bold green"
            elif result.similarity_score > 0.1:
                score_style = "green"
            else:
                score_style = "yellow"

            # Truncate description
            desc = (
                result.description[:80] + "..."
                if len(result.description) > 80
                else result.description
            )

            table.add_row(
                str(i + 1),
                f"[{score_style}]{result.similarity_score:.3f}[/{score_style}]",
                result.cve_id,
                result.severity or "Unknown",
                desc,
            )

        console.print(table)

        # Show top result details
        if results:
            best_result = results[0]
            console.print("\n[bold green]Top Result:[/bold green]")
            console.print(f"  CVE ID: [cyan]{best_result.cve_id}[/cyan]")
            console.print(
                f"  Similarity Score: [green]{best_result.similarity_score:.3f}[/green]"
            )
            console.print(f"  Severity: [red]{best_result.severity or 'Unknown'}[/red]")
            console.print(f"  CWE: [yellow]{best_result.cwe_id or 'Unknown'}[/yellow]")
            console.print(f"  Description: [white]{best_result.description}[/white]")

        # Quality analysis
        perfect_matches = [r for r in results if r.similarity_score > 0.3]
        good_matches = [r for r in results if 0.1 <= r.similarity_score <= 0.3]
        poor_matches = [r for r in results if r.similarity_score < 0.1]

        console.print("\n[bold cyan] Search Quality Analysis:[/bold cyan]")
        console.print(
            f"  Perfect matches (>0.3): [green]{len(perfect_matches)}[/green]"
        )
        console.print(
            f"  OK: Good matches (0.1-0.3): [yellow]{len(good_matches)}[/yellow]"
        )
        console.print(f"  ERROR: Poor matches (<0.1): [red]{len(poor_matches)}[/red]")

        console.print(
            "\n[bold green]OK: RAG search completed successfully![/bold green]"
        )

    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        raise typer.Exit(1)
    finally:
        if "rag" in locals():
            rag.close()


@query_app.command()
def legacy(
    question: str = typer.Argument(
        ..., help="Natural language question using legacy orchestrator (template-based)"
    ),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of results to return"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Debug mode with detailed step-by-step output"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Ask questions using the legacy orchestrator (template-based approach)."""

    console.print(
        "[red]ERROR: This command is deprecated - the legacy orchestrator has been archived[/red]"
    )
    console.print("[yellow]TIP: Please use the main 'ask' command instead:[/yellow]")
    console.print(
        f'  [cyan]uv run python -m claire_kg.cli ask "{question}" --limit {limit}[/cyan]'
    )
    raise typer.Exit(1)


@query_app.command()
def langchain(
    question: str = typer.Argument(
        ...,
        help="Natural language question to ask the knowledge graph using LLM-generated Cypher queries",
    ),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of results to return"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Debug mode with detailed step-by-step output"
    ),
):
    """Ask intelligent questions using LLM-generated Cypher queries (uses OpenAI SDK directly)."""

    console.print("[bold blue]CLAIRE-KG LLM Query[/bold blue]")
    console.print(f"Question: [green]{question}[/green]")
    console.print(f"Limit: [cyan]{limit}[/cyan]")

    try:
        # Initialize LLM orchestrator
        orchestrator = QueryOrchestrator()

        if not orchestrator.connect():
            console.print("[red]ERROR: Failed to connect to Neo4j database[/red]")
            raise typer.Exit(1)

        console.print("[green]OK: Connected to Neo4j database[/green]")

        # Process the query
        console.print("[green]Processing query with LLM...[/green]")

        result = orchestrator.process_query(question, limit, debug)

        if result["success"]:
            console.print("[green]OK: Query completed successfully[/green]")
        else:
            console.print(f"[red]ERROR: Query failed: {result['error_message']}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        raise typer.Exit(1)
    finally:
        if "orchestrator" in locals():
            orchestrator.close()


@test_app.command()
def debug_help():
    """Show help for debug mode and troubleshooting."""
    console.print("[bold blue]CLAIRE-KG Debug Mode Help[/bold blue]")
    console.print()
    console.print("[bold green]Debug Mode Features:[/bold green]")
    console.print("• [cyan]--debug, -d[/cyan]: Enable detailed step-by-step output")
    console.print(
        "• [cyan]--verbose, -v[/cyan]: Enable verbose output (less detailed than debug)"
    )
    console.print()
    console.print("[bold green]Debug Mode Shows:[/bold green]")
    console.print(
        "• [yellow]Step 1: Intent Classification[/yellow] - How the system understands your query"
    )
    console.print(
        "• [yellow]Step 2: Entity Linking[/yellow] - How terms are mapped to canonical IDs"
    )
    console.print(
        "• [yellow]Step 3: Graph Retrieval[/yellow] - Cypher query execution and results"
    )
    console.print(
        "• [yellow]Step 4: Context Bundling[/yellow] - How results are packaged"
    )
    console.print()
    console.print("[bold green]Example Usage:[/bold green]")
    console.print(
        '[cyan]uv run python -m claire_kg.cli query "Show me XSS vulnerabilities" --debug[/cyan]'
    )
    console.print(
        '[cyan]uv run python -m claire_kg.cli query "Find critical buffer overflow" --limit 3 --debug[/cyan]'
    )
    console.print()
    console.print("[bold green]Troubleshooting:[/bold green]")
    console.print("• If queries return no results, check the debug output for:")
    console.print("  - Intent classification confidence (should be > 0.3)")
    console.print("  - Entity linking results (should find relevant entities)")
    console.print("  - Cypher query parameters (should match your intent)")
    console.print("  - Raw results from database (should contain data)")
    console.print()
    console.print("[bold green]Performance Tips:[/bold green]")
    console.print("• Use [cyan]--limit[/cyan] to control result count")
    console.print("• Debug mode adds ~0.1s overhead")
    console.print("• Verbose mode is faster than debug mode")


# -----------------------------------------------------------------------------
# Helper functions: example questions, selection prompts, load by number
# -----------------------------------------------------------------------------


def _show_examples(examples_arg: str, console: Console, return_list: bool = False):
    """Show example questions from baseline test questions.

    Args:
        examples_arg: Category or crosswalk to show
        console: Rich console for output
        return_list: If True, return list of (num, question) tuples instead of just displaying

    Returns:
        List of (num, question) tuples if return_list=True, None otherwise
    """
    # Find the baseline questions file
    project_root = Path(__file__).parent.parent.parent
    baseline_file = project_root / "docs" / "testing" / "baseline_test_questions.md"

    if not baseline_file.exists():
        console.print(
            f"[red]Error: Baseline questions file not found at {baseline_file}[/red]"
        )
        return [] if return_list else None

    try:
        content = baseline_file.read_text()
    except Exception as e:
        console.print(f"[red]Error reading baseline questions file: {e}[/red]")
        return [] if return_list else None

    # Parse examples
    examples_arg_lower = examples_arg.lower().strip()

    if examples_arg_lower == "list":
        # Show available categories with numbers
        categories_list = []

        # Single Datasets
        single_datasets = [
            ("CVE", "Vulnerability examples"),
            ("CWE", "Weakness examples"),
            ("CAPEC", "Attack pattern examples"),
            ("ATT&CK", "Technique examples"),
            ("NICE", "NICE workforce examples"),
            ("DCWF", "DCWF workforce examples"),
        ]

        # Crosswalks
        crosswalks = [
            ("CVE->CWE", "CVE to CWE crosswalk"),
            ("CVE->ATTACK", "CVE to ATT&CK crosswalk"),
            ("CAPEC->ATTACK", "CAPEC to ATT&CK crosswalk"),
            ("NICE->ATTACK", "NICE to ATT&CK crosswalk"),
            ("CVE->ASSET", "CVE to Asset crosswalk"),
        ]

        # Other
        other = [
            ("all", "Show all examples"),
        ]

        # Build numbered list
        num = 1
        for name, desc in single_datasets:
            categories_list.append((num, name, desc))
            num += 1
        for name, desc in crosswalks:
            categories_list.append((num, name, desc))
            num += 1
        for name, desc in other:
            categories_list.append((num, name, desc))
            num += 1

        # Display categories
        console.print("[bold blue]Available Example Categories:[/bold blue]")
        console.print()
        console.print("[bold]Single Datasets:[/bold]")
        for num, name, desc in categories_list[: len(single_datasets)]:
            console.print(f"  [cyan]{num}.[/cyan] {name} - {desc}")
        console.print()
        console.print("[bold]Crosswalks:[/bold]")
        start_idx = len(single_datasets)
        for num, name, desc in categories_list[start_idx : start_idx + len(crosswalks)]:
            alt_name = name.replace("->", "↔")
            console.print(f"  [cyan]{num}.[/cyan] {name} or {alt_name} - {desc}")
        console.print()
        console.print("[bold]Other:[/bold]")
        start_idx = len(single_datasets) + len(crosswalks)
        for num, name, desc in categories_list[start_idx:]:
            console.print(f"  [cyan]{num}.[/cyan] {name} - {desc}")

        # If return_list is True, return the categories for selection
        if return_list:
            return [(idx, name) for idx, name, _ in categories_list]

        return None

    # Extract examples based on category
    examples_to_show = []

    if examples_arg_lower == "all":
        # Show all examples grouped by category
        # Extract questions with their categories from code blocks
        lines = content.split("\n")
        in_code_block = False
        current_category = None
        current_section = None

        # Dictionary to store questions by category
        questions_by_category = {}

        for line in lines:
            # Track section headers (## or ###)
            # Match 2 or 3 # characters
            section_match = re.match(r"^(#{2,3})\s+(.+)$", line)
            if section_match:
                level, section_text = section_match.groups()
                # Extract category from section headers - focus on datasets/crosswalks, not difficulty
                if (
                    len(level) == 3
                ):  # Level 3 headers (###) - these define the actual categories
                    # Extract category from patterns like:
                    # "### CVE (Vulnerabilities)" -> "CVE"
                    # "### ATT&CK" -> "ATT&CK"
                    # "### Vulnerability Domain Crosswalks" -> "CVE Crosswalks"

                    # First check for specific dataset patterns
                    # Handle patterns like "### CVE (Vulnerabilities)" or just "### CVE"
                    section_lower = section_text.lower()
                    if "cve" in section_lower:
                        current_category = "CVE"
                    elif "cwe" in section_lower:
                        current_category = "CWE"
                    elif "capec" in section_lower:
                        current_category = "CAPEC"
                    elif "attack" in section_lower or "att&ck" in section_lower:
                        current_category = "ATT&CK"
                    elif "nice" in section_lower:
                        current_category = "NICE"
                    elif "dcwf" in section_lower:
                        current_category = "DCWF"
                    elif "Crosswalk" in section_text or "Cross-Domain" in section_text:
                        # Extract crosswalk name from section
                        if "Vulnerability Domain" in section_text:
                            current_category = "CVE Crosswalks"
                        elif "ATT&CK Domain" in section_text:
                            current_category = "ATT&CK Crosswalks"
                        elif "Mitigation" in section_text:
                            current_category = "Mitigation Crosswalks"
                        elif (
                            "Workforce" in section_text
                            and "Cross-Domain" not in section_text
                        ):
                            current_category = "Workforce Crosswalks"
                        elif "Cross-Domain Workforce" in section_text:
                            current_category = "Cross-Domain Workforce"
                    elif (
                        "Multi-Hop" in section_text
                        or "3-Hop" in section_text
                        or "4-Hop" in section_text
                    ):
                        if "3-Hop" in section_text and "CVE" in section_text:
                            current_category = "Multi-Hop (CVE→CWE→CAPEC→ATT&CK)"
                        elif (
                            "4-Hop" in section_text or "Cross-Framework" in section_text
                        ):
                            current_category = "Multi-Hop (Cross-Framework)"
                        elif "Workforce-Linked" in section_text:
                            current_category = "Multi-Hop (Workforce-Linked)"
                        elif "Advanced" in section_text:
                            current_category = "Multi-Hop (Advanced)"
                        else:
                            current_category = "Multi-Hop Reasoning"
                    elif (
                        "SPECIAL CASES" in section_text or "Vocabulary" in section_text
                    ):
                        if "Multiple Mapping" in section_text:
                            current_category = "Vocabulary (Multiple Mapping)"
                        elif "Cross-Dataset Ambiguity" in section_text:
                            current_category = "Vocabulary (Cross-Dataset)"
                        elif "Complex Multi-Term" in section_text:
                            current_category = "Vocabulary (Multi-Term)"
                        elif "Prompt-Chaining" in section_text:
                            current_category = "Vocabulary (Prompt-Chaining)"
                        else:
                            current_category = "Vocabulary & Ambiguity"
                # Ignore level 2 headers (##) - they're just difficulty markers, not categories

                # Initialize category if not exists
                if current_category and current_category not in questions_by_category:
                    questions_by_category[current_category] = []
                continue

            # Track code blocks (questions are in code blocks)
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            # Extract from code blocks only
            if in_code_block:
                # Match numbered questions: "1. Question text" or "1. Question text?"
                pattern = r"^(\d+)\.\s+(.+)$"
                match = re.match(pattern, line.strip())
                if match:
                    num, question = match.groups()
                    question_clean = question.strip()
                    # Filter out non-questions
                    if not question_clean.startswith("**") and len(question_clean) > 10:
                        # Only store if we have a category
                        if current_category:
                            if current_category not in questions_by_category:
                                questions_by_category[current_category] = []
                            questions_by_category[current_category].append(
                                (int(num), question_clean)
                            )
                        else:
                            # Fallback: store in "Uncategorized" if no category set
                            if "Uncategorized" not in questions_by_category:
                                questions_by_category["Uncategorized"] = []
                            questions_by_category["Uncategorized"].append(
                                (int(num), question_clean)
                            )

        # Convert to flat list for display, but preserve category info
        # We'll group them in the display function
        for category, questions in questions_by_category.items():
            for num, question in questions:
                examples_to_show.append((num, question, category))
    else:
        # Parse category or crosswalk
        # Normalize crosswalk notation
        crosswalk = examples_arg.upper().replace("↔", "->").replace("<->", "->")

        # Single dataset (case-insensitive)
        dataset_map = {
            "cve": r"### CVE \(Vulnerabilities\)",
            "cwe": r"### CWE \(Weaknesses\)",
            "capec": r"### CAPEC \(Attack Patterns\)",
            "attack": r"### ATT&CK",
            "nice": r"### NICE",
            "dcwf": r"### DCWF",
        }

        # Crosswalk patterns (normalized to uppercase)
        crosswalk_map = {
            "CVE->CWE": r"### Vulnerability Domain Crosswalks",
            "CVE->ASSET": r"### Vulnerability Domain Crosswalks",
            "CAPEC->ATTACK": r"### ATT&CK Domain Crosswalks",
            "CAPEC->ATT&CK": r"### ATT&CK Domain Crosswalks",
            "CVE->ATTACK": r"### ATT&CK Domain Crosswalks",
            "CVE->ATT&CK": r"### ATT&CK Domain Crosswalks",
            "NICE->ATTACK": r"### Workforce Crosswalks",
            "NICE->ATT&CK": r"### Workforce Crosswalks",
            "DCWF->ATTACK": r"### Cross-Domain Workforce Crosswalks",
            "DCWF->ATT&CK": r"### Cross-Domain Workforce Crosswalks",
        }

        # Find section
        section_pattern = None
        if crosswalk in crosswalk_map:
            section_pattern = crosswalk_map[crosswalk]
        elif examples_arg_lower in dataset_map:
            section_pattern = dataset_map[examples_arg_lower]

        if section_pattern:
            # Find the section
            lines = content.split("\n")
            in_section = False
            in_code_block = False
            section_end_patterns = [
                r"^## ",
                r"^---",
            ]

            for i, line in enumerate(lines):
                # Check if we've hit the end of the section (before processing)
                if in_section and any(
                    re.match(pattern, line) for pattern in section_end_patterns
                ):
                    break

                # Check if we've hit another ### header (end of this section)
                if in_section and re.match(r"^###\s+", line):
                    break

                if re.search(section_pattern, line):
                    in_section = True
                    continue

                if in_section:
                    # Track code blocks (questions are in code blocks)
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue

                    # Extract numbered questions from code blocks only
                    if in_code_block:
                        match = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
                        if match:
                            num, question = match.groups()
                            question_clean = question.strip()
                            # Filter out non-questions
                            if (
                                not question_clean.startswith("**")
                                and len(question_clean) > 10
                            ):
                                examples_to_show.append((int(num), question_clean))
        else:
            console.print(f"[yellow]Unknown category: {examples_arg}[/yellow]")
            console.print("Use --examples list to see available categories")
            return

    # Display examples
    if not examples_to_show:
        console.print(f"[yellow]No examples found for: {examples_arg}[/yellow]")
        return [] if return_list else None

    # Always display examples first
    console.print(f"[bold blue]Example Questions: {examples_arg.upper()}[/bold blue]")
    console.print()

    # Check if we have categories (for "all" command)
    has_categories = any(len(item) > 2 for item in examples_to_show)

    if has_categories and examples_arg_lower == "all":
        # Group by category
        categories = {}
        for item in examples_to_show:
            if len(item) == 3:
                num, question, category = item
                if category not in categories:
                    categories[category] = []
                categories[category].append((num, question))
            else:
                # Fallback for items without category
                num, question = item
                if "Uncategorized" not in categories:
                    categories["Uncategorized"] = []
                categories["Uncategorized"].append((num, question))

        # Display by category
        total_count = 0
        for category in sorted(categories.keys()):
            questions = categories[category]
            console.print(
                f"[bold yellow]{category}[/bold yellow] ({len(questions)} questions)"
            )
            for num, question in questions:
                console.print(f"  [cyan]{num}.[/cyan] {question}")
            console.print()
            total_count += len(questions)

        console.print(
            f"[dim]Total: {total_count} examples across {len(categories)} categories[/dim]"
        )
    else:
        # Display flat list (for specific categories)
        for item in examples_to_show:
            if len(item) == 3:
                num, question, _ = item
            else:
                num, question = item
            console.print(f"[cyan]{num}.[/cyan] {question}")

        console.print()
        console.print(f"[dim]Total: {len(examples_to_show)} examples[/dim]")

    # If return_list is True, return the flattened list
    if return_list:
        # Flatten the list (remove category info if present)
        flat_list = []
        for item in examples_to_show:
            if len(item) == 3:
                num, question, _ = item
            else:
                num, question = item
            flat_list.append((num, question))
        return flat_list

    return None


def _prompt_for_selection(
    examples_list: list, console: Console, is_category: bool = False
) -> Optional[str]:
    """Prompt user to select a number from the list.

    Args:
        examples_list: List of (num, item) tuples (questions or categories)
        console: Rich console for output
        is_category: If True, this is a category selection, otherwise it's a question selection

    Returns:
        Selected item text (question or category name), or None if cancelled/invalid
    """
    if not examples_list:
        return None

    console.print()
    if is_category:
        console.print("[bold cyan]Select a category number:[/bold cyan]")
    else:
        console.print("[bold cyan]Select a question number to ask:[/bold cyan]")

    # Show the list again for reference if it's a category selection
    if is_category and len(examples_list) <= 12:
        console.print("[dim]Categories:[/dim]")
        for num, item in examples_list:
            console.print(f"  [dim]{num}.[/dim] {item}")
        console.print()

    try:
        selection = input("Enter number (or 'q' to quit): ").strip()

        if selection.lower() in ["q", "quit", "exit", ""]:
            console.print("[yellow]Cancelled[/yellow]")
            return None

        selected_num = int(selection)

        # Find the item with this number
        for num, item in examples_list:
            if num == selected_num:
                if is_category:
                    console.print(f"[green]Selected category #{num}: {item}[/green]")
                else:
                    console.print(f"[green]Selected question #{num}: {item}[/green]")
                console.print()
                return item

        # Number not found
        if is_category:
            console.print(f"[red]Category #{selected_num} not found in the list[/red]")
        else:
            console.print(f"[red]Question #{selected_num} not found in the list[/red]")
        return None

    except ValueError:
        console.print("[red]Invalid input. Please enter a number.[/red]")
        return None
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        return None


def _load_question_by_number(question_num: int, console: Console) -> Optional[str]:
    """Load a specific question by number from baseline test questions."""
    # Find the baseline questions file
    project_root = Path(__file__).parent.parent.parent
    baseline_file = project_root / "docs" / "testing" / "baseline_test_questions.md"

    if not baseline_file.exists():
        return None

    try:
        content = baseline_file.read_text()
    except Exception as e:
        console.print(f"[red]Error reading baseline questions file: {e}[/red]")
        return None

    # Extract question by number from code blocks
    lines = content.split("\n")
    in_code_block = False

    for line in lines:
        # Track code blocks (questions are in code blocks)
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        # Extract from code blocks only
        if in_code_block:
            # Match numbered questions: "1. Question text" or "1. Question text?"
            pattern = r"^(\d+)\.\s+(.+)$"
            match = re.match(pattern, line.strip())
            if match:
                num, question = match.groups()
                if int(num) == question_num:
                    question_clean = question.strip()
                    # Filter out non-questions
                    if not question_clean.startswith("**") and len(question_clean) > 10:
                        return question_clean

    return None


if __name__ == "__main__":
    app()
