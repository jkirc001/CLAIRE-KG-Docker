"""Run CLAIRE-KG via subprocess: invoke CLI ask command and parse Phase 1 JSON + Phase 2 answer.

Used by external callers (e.g. evaluators, scripts) to run a question through the
full pipeline without importing the orchestrator. Discovers project root via
CLAIRE_KG_PATH, explicit kg_path, or parent of this file; runs `uv run python -m
claire_kg.cli ask <question> --show-json --debug`; parses stdout for Phase 1 JSON,
Phase 2 enhanced answer (box drawing), and cost/token lines.

Entry points: get_project_paths(kg_path) → ProjectPaths; run_kg(question, limit, ...)
→ (phase1_json, phase2_answer, error, metadata).
"""

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# -----------------------------------------------------------------------------
# Project path resolution and CLI runner
# -----------------------------------------------------------------------------


def get_project_paths(kg_path: Optional[str] = None):
    """Resolve CLAIRE-KG project root: explicit path, CLAIRE_KG_PATH env, or parent of this file."""

    class ProjectPaths:
        """Holds resolved kg_path (Path or None)."""

        def __init__(self, kg_path: Optional[Path]):
            """Store resolved project root path (or None)."""
            self.kg_path = kg_path

    if kg_path:
        return ProjectPaths(Path(kg_path))

    import os

    kg_path_env = os.getenv("CLAIRE_KG_PATH")
    if kg_path_env:
        return ProjectPaths(Path(kg_path_env))

    # Default: parent of src/claire_kg (project root) if pyproject.toml present
    current_path = Path(__file__).parent.parent.parent
    if (current_path / "pyproject.toml").exists():
        return ProjectPaths(current_path)

    return ProjectPaths(None)


def run_kg(
    question: str,
    limit: int = 10,
    kg_path: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Dict[str, Any]]:
    """Run a question through CLAIRE-KG using CLI subprocess.

    Args:
        question: The question to ask
        limit: Limit for query results
        kg_path: Override path to CLAIRE-KG project
        timeout: Timeout in seconds (None for no timeout)

    Returns:
        Tuple of (phase1_json, phase2_answer, error, metadata)
        - phase1_json: Phase 1 JSON output (Cypher query results)
        - phase2_answer: Phase 2 natural language answer
        - error: Error message if execution failed
        - metadata: Execution metadata (time, cost, tokens, etc.)
    """
    start_time = time.time()
    metadata: Dict[str, Any] = {
        "execution_time": 0.0,
        "cost_usd": 0.0,
        "success": False,
    }

    try:
        # Get project path
        paths = get_project_paths(kg_path=kg_path)
        if paths.kg_path is None:
            error_msg = "CLAIRE-KG project path not found. Please set CLAIRE_KG_PATH or ensure project is in sibling directory."
            return None, None, error_msg, metadata

        if not paths.kg_path.exists():
            error_msg = f"CLAIRE-KG path does not exist: {paths.kg_path}"
            return None, None, error_msg, metadata

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "claire_kg.cli",
                "ask",
                question,
                "--limit",
                str(limit),
                "--show-json",
                "--debug",
            ]

            result = subprocess.run(
                cmd,
                cwd=str(paths.kg_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                # Check if stderr only contains warnings (not actual errors)
                stderr_lower = result.stderr.lower() if result.stderr else ""
                if "warning" in stderr_lower and "virtual_env" in stderr_lower:
                    pass  # Ignore virtual-env warning; treat as success
                else:
                    error_msg = f"CLAIRE-KG failed: {result.stderr}"
                    return None, None, error_msg, metadata

            # Check if we got output
            if not result.stdout:
                error_msg = "CLAIRE-KG returned no output"
                return None, None, error_msg, metadata

            output = result.stdout

            # Parse Phase 1 JSON: between "Phase 1 JSON Output:" and "Phase 2 Enhanced Answer"
            phase1_json = None
            if "Phase 1 JSON Output:" in output:
                json_start_marker = "Phase 1 JSON Output:"
                start_idx = output.find(json_start_marker) + len(json_start_marker)
                phase2_marker = "Phase 2 Enhanced Answer"
                end_idx = output.find(phase2_marker, start_idx)

                if end_idx < 0:
                    end_idx = len(output)

                json_section = output[start_idx:end_idx].strip()
                first_brace = json_section.find("{")
                if first_brace >= 0:
                    # Balance braces while ignoring braces inside quoted strings
                    brace_count = 0
                    last_brace = len(json_section)
                    in_string = False
                    escape_next = False

                    for i, char in enumerate(json_section[first_brace:], first_brace):
                        if escape_next:
                            escape_next = False
                            continue
                        elif char == "\\":
                            escape_next = True
                            continue
                        elif char == '"':
                            in_string = not in_string
                        elif not in_string:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    last_brace = i + 1
                                    break

                    json_str = json_section[first_brace:last_brace].strip()
                    # Remove control chars; then escape newlines/tabs inside strings (CLI often emits raw newlines)
                    json_str_clean = re.sub(
                        r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", json_str
                    )
                    fixed_chars = []
                    in_string = False
                    escape_next = False

                    i = 0
                    while i < len(json_str_clean):
                        char = json_str_clean[i]

                        if escape_next:
                            # After backslash - this character is part of an escape sequence
                            fixed_chars.append(char)
                            escape_next = False
                            i += 1
                        elif char == "\\":
                            # Backslash - check what follows
                            if i + 1 < len(json_str_clean):
                                next_char = json_str_clean[i + 1]
                                # Valid JSON escape sequences: ", \, /, b, f, n, r, t, u
                                if next_char in '"\\/bfnrtu':
                                    # Valid escape sequence
                                    fixed_chars.append(char)
                                    escape_next = True
                                    i += 1
                                else:
                                    # Invalid escape sequence (like \[ or \]) - escape the backslash itself
                                    # This handles Windows paths and other literal backslashes
                                    fixed_chars.append("\\\\")
                                    # Don't set escape_next - the next char will be handled normally
                                    i += 1
                            else:
                                # Backslash at end - escape it
                                fixed_chars.append("\\\\")
                                i += 1
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                            fixed_chars.append(char)
                            i += 1
                        elif in_string:
                            if char == "\n":
                                fixed_chars.append("\\n")
                                i += 1
                            elif char == "\r":
                                fixed_chars.append("\\r")
                                i += 1
                            elif char == "\t":
                                fixed_chars.append("\\t")
                                i += 1
                            elif ord(char) < 32:
                                fixed_chars.append(f"\\u{ord(char):04x}")
                                i += 1
                            else:
                                fixed_chars.append(char)
                                i += 1
                        else:
                            fixed_chars.append(char)
                            i += 1

                    json_str_fixed = "".join(fixed_chars)
                    try:
                        phase1_json = json.loads(json_str_fixed)
                    except (json.JSONDecodeError, ValueError):
                        pass

            # Phase 2: extract answer from Rich box after "Phase 2 Enhanced Answer (with citations):"
            phase2_answer = None
            if "+" in output and "|" in output:
                lines = output.split("\n")
                phase2_header_idx = None
                for i, line in enumerate(lines):
                    if (
                        "Phase 2 Enhanced Answer" in line
                        and "citations" in line.lower()
                    ):
                        phase2_header_idx = i
                        break

                if phase2_header_idx is not None:
                    box_start = None
                    box_end = None
                    for i in range(phase2_header_idx + 1, len(lines)):
                        line = lines[i]
                        if "+" in line and "-" in line:  # Box boundary line
                            if box_start is None:
                                box_start = i
                            else:
                                box_end = i
                                break

                    if box_start is not None and box_end is not None:
                        answer_lines = []
                        for i in range(box_start + 1, box_end):
                            line = lines[i]
                            if "|" in line:
                                parts = line.split("|")
                                if len(parts) >= 2:
                                    text = (
                                        "|".join(parts[1:-1]).strip()
                                        if len(parts) > 2
                                        else parts[1].strip()
                                    )
                                    if text:
                                        answer_lines.append(text)
                        phase2_answer = (
                            "\n".join(answer_lines).strip() if answer_lines else None
                        )

                if phase2_answer is None:
                    box_boundaries = [
                        i for i, line in enumerate(lines) if "+" in line and "-" in line
                    ]
                    if len(box_boundaries) >= 2:
                        box_start = box_boundaries[-2]
                        box_end = box_boundaries[-1]
                        answer_lines = []
                        for i in range(box_start + 1, box_end):
                            line = lines[i]
                            if "|" in line:
                                parts = line.split("|")
                                if len(parts) >= 2:
                                    text = (
                                        "|".join(parts[1:-1]).strip()
                                        if len(parts) > 2
                                        else parts[1].strip()
                                    )
                                    if text:
                                        answer_lines.append(text)
                        phase2_answer = (
                            "\n".join(answer_lines).strip() if answer_lines else None
                        )
            else:
                phase2_answer = output.strip() if output.strip() else None

            # Cost: sum indented "Cost: $X.XX" lines (Phase 1 + Phase 2), else "LLM Cost:"
            cost_usd = 0.0
            individual_costs = re.findall(
                r"^\s+Cost:\s+\$?([\d.]+)", output, re.MULTILINE
            )
            if individual_costs:
                cost_usd = sum(float(c) for c in individual_costs)
            else:
                total_cost_match = re.search(r"LLM Cost:\s+\$?([\d.]+)", output)
                if total_cost_match:
                    cost_usd = float(total_cost_match.group(1))

            metadata["cost_usd"] = cost_usd
            metadata["success"] = phase1_json is not None
            metadata["execution_time"] = time.time() - start_time
            return phase1_json, phase2_answer, None, metadata

        except subprocess.TimeoutExpired:
            error_msg = f"CLAIRE-KG execution timed out after {timeout} seconds"
            metadata["execution_time"] = time.time() - start_time
            return None, None, error_msg, metadata
        except Exception as e:
            error_msg = f"Error running CLAIRE-KG: {e}"
            metadata["execution_time"] = time.time() - start_time
            return None, None, error_msg, metadata

    except Exception as e:
        error_msg = f"Unexpected error in KG runner: {e}"
        metadata["execution_time"] = time.time() - start_time
        return None, None, error_msg, metadata
