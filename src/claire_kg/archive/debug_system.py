"""
Debug System for CLAIRE-KG LLM-First Pipeline
Provides detailed logging and visualization of each step in the process.
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.tree import Tree
from rich.text import Text
import json


@dataclass
class DebugStep:
    """Represents a single step in the debug pipeline."""

    step_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "running"  # running, success, error, warning
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def complete(
        self,
        status: str = "success",
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Mark step as complete."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        self.output_data = output_data
        self.error_message = error_message
        self.metadata = metadata


class DebugSystem:
    """Debug system for tracking pipeline execution."""

    def __init__(self, enabled: bool = True, verbose: bool = False):
        """Create debug tracker; optionally enabled and verbose."""
        self.enabled = enabled
        self.verbose = verbose
        self.console = Console()
        self.steps: List[DebugStep] = []
        self.current_step: Optional[DebugStep] = None
        self.pipeline_start_time: Optional[float] = None
        self.pipeline_end_time: Optional[float] = None

    def start_pipeline(self, question: str):
        """Start debugging a new pipeline execution."""
        if not self.enabled:
            return

        self.pipeline_start_time = time.time()
        self.steps = []
        self.current_step = None

        self.console.print("\n" + "=" * 80)
        self.console.print(
            f" [bold blue]CLAIRE-KG LLM-First Pipeline Debug[/bold blue]"
        )
        self.console.print(f"[bold green]Question:[/bold green] {question}")
        self.console.print(
            f"[bold green]Started:[/bold green] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.console.print("=" * 80)

    def start_step(
        self,
        step_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Start a new debug step."""
        if not self.enabled:
            return

        # Complete previous step if exists
        if self.current_step and self.current_step.status == "running":
            self.current_step.complete("warning", error_message="Step interrupted")

        # Start new step
        self.current_step = DebugStep(
            step_name=step_name,
            start_time=time.time(),
            input_data=input_data,
            metadata=metadata,
        )
        self.steps.append(self.current_step)

        # Show step start
        self.console.print(
            f"\n🔄 [bold yellow]Step {len(self.steps)}: {step_name}[/bold yellow]"
        )
        if self.verbose and input_data:
            self._show_input_data(input_data)
        if self.verbose and metadata:
            self._show_metadata(metadata)

    def complete_step(
        self,
        status: str = "success",
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Complete the current debug step."""
        if not self.enabled or not self.current_step:
            return

        self.current_step.complete(status, output_data, error_message, metadata)

        # Show step completion
        duration = self.current_step.duration
        status_emoji = {
            "success": "OK:",
            "error": "ERROR:",
            "warning": "WARNING:",
            "running": "🔄",
        }.get(status, "❓")

        self.console.print(
            f"{status_emoji} [bold green]Completed:[/bold green] {self.current_step.step_name} ({duration:.2f}s)"
        )

        if self.verbose and output_data:
            self._show_output_data(output_data)
        if error_message:
            self.console.print(f"ERROR: [bold red]Error:[/bold red] {error_message}")
        if self.verbose and metadata:
            self._show_metadata(metadata)

    def end_pipeline(self, final_result: Optional[Dict[str, Any]] = None):
        """End the pipeline execution."""
        if not self.enabled:
            return

        self.pipeline_end_time = time.time()
        total_duration = self.pipeline_end_time - self.pipeline_start_time

        # Complete any running step
        if self.current_step and self.current_step.status == "running":
            self.current_step.complete(
                "warning", error_message="Pipeline ended abruptly"
            )

        # Show pipeline summary
        self.console.print("\n" + "=" * 80)
        self.console.print(f"[bold blue]Pipeline Complete[/bold blue]")
        self.console.print(
            f"[bold green]Total Duration:[/bold green] {total_duration:.2f}s"
        )
        self.console.print(f" [bold green]Steps:[/bold green] {len(self.steps)}")

        # Show step summary
        self._show_step_summary()

        # Show final result if provided
        if final_result:
            self._show_final_result(final_result)

        self.console.print("=" * 80)

    def _show_input_data(self, input_data: Dict[str, Any]):
        """Show input data in a formatted way."""
        if not input_data:
            return

        self.console.print("📥 [bold blue]Input Data:[/bold blue]")

        for key, value in input_data.items():
            if isinstance(value, str) and len(value) > 100:
                # Show truncated string
                self.console.print(f"  {key}: {value[:100]}...")
            elif isinstance(value, (dict, list)) and len(str(value)) > 200:
                # Show summary for large objects
                self.console.print(
                    f"  {key}: {type(value).__name__} ({len(value) if isinstance(value, list) else len(value.keys())} items)"
                )
            else:
                self.console.print(f"  {key}: {value}")

    def _show_output_data(self, output_data: Dict[str, Any]):
        """Show output data in a formatted way."""
        if not output_data:
            return

        self.console.print("📤 [bold blue]Output Data:[/bold blue]")

        for key, value in output_data.items():
            if key == "cypher_query" and isinstance(value, str):
                # Show Cypher query with syntax highlighting
                syntax = Syntax(value, "cypher", theme="monokai", line_numbers=True)
                self.console.print(f"  {key}:")
                self.console.print(syntax)
            elif isinstance(value, str) and len(value) > 100:
                self.console.print(f"  {key}: {value[:100]}...")
            elif isinstance(value, (dict, list)) and len(str(value)) > 200:
                self.console.print(
                    f"  {key}: {type(value).__name__} ({len(value) if isinstance(value, list) else len(value.keys())} items)"
                )
            else:
                self.console.print(f"  {key}: {value}")

    def _show_metadata(self, metadata: Dict[str, Any]):
        """Show metadata in a formatted way."""
        if not metadata:
            return

        self.console.print(" [bold blue]Metadata:[/bold blue]")
        for key, value in metadata.items():
            self.console.print(f"  {key}: {value}")

    def _show_step_summary(self):
        """Show a summary of all steps."""
        if not self.steps:
            return

        table = Table(title="Pipeline Steps Summary")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Details", style="white")

        for i, step in enumerate(self.steps, 1):
            status_emoji = {
                "success": "OK:",
                "error": "ERROR:",
                "warning": "WARNING:",
                "running": "🔄",
            }.get(step.status, "❓")

            duration = f"{step.duration:.2f}s" if step.duration else "N/A"

            details = []
            if step.error_message:
                details.append(f"Error: {step.error_message}")
            if step.metadata:
                for key, value in step.metadata.items():
                    details.append(f"{key}: {value}")

            details_str = "; ".join(details) if details else "OK"

            table.add_row(
                f"{i}. {step.step_name}",
                f"{status_emoji} {step.status}",
                duration,
                details_str,
            )

        self.console.print(table)

    def _show_final_result(self, final_result: Dict[str, Any]):
        """Show the final result."""
        self.console.print("\n🎯 [bold blue]Final Result:[/bold blue]")

        if "answer" in final_result:
            # Show the answer in a panel
            answer_panel = Panel(
                final_result["answer"], title="Generated Answer", border_style="green"
            )
            self.console.print(answer_panel)

        # Show other result data
        for key, value in final_result.items():
            if key != "answer":
                if key == "sources" and isinstance(value, list):
                    self.console.print(
                        f"📚 [bold green]Sources:[/bold green] {', '.join(value)}"
                    )
                elif key == "confidence" and isinstance(value, (int, float)):
                    confidence_color = (
                        "green" if value > 0.8 else "yellow" if value > 0.6 else "red"
                    )
                    self.console.print(
                        f"🎯 [bold {confidence_color}]Confidence:[/bold {confidence_color}] {value:.2%}"
                    )
                else:
                    self.console.print(f"  {key}: {value}")

    def get_debug_report(self) -> Dict[str, Any]:
        """Get a complete debug report."""
        return {
            "pipeline_duration": (
                self.pipeline_end_time - self.pipeline_start_time
                if self.pipeline_end_time
                else None
            ),
            "total_steps": len(self.steps),
            "steps": [asdict(step) for step in self.steps],
            "success_rate": (
                len([s for s in self.steps if s.status == "success"]) / len(self.steps)
                if self.steps
                else 0
            ),
        }

    def save_debug_report(self, filename: str):
        """Save debug report to file."""
        report = self.get_debug_report()
        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)


# Global debug system instance
debug_system = DebugSystem()


def enable_debug(enabled: bool = True, verbose: bool = False):
    """Enable or disable debug mode."""
    global debug_system
    debug_system.enabled = enabled
    debug_system.verbose = verbose


def get_debug_system() -> DebugSystem:
    """Get the global debug system instance."""
    return debug_system
