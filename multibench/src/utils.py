"""Shared dataset, prompting, candidate-selection, and grading utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


BENCHMARK_SPLITS = {
    "gsm8k": ("train", "test"),
    "math500": ("train", "test"),
    "folio": ("train", "validation"),
    "reclor": ("train", "validation"),
    "humaneval": ("test",),
    "mbpp": ("train", "validation", "test", "prompt"),
}

LABELS = tuple("ABCDEFGHIJKLMNOPQ")

MATH_SYSTEM_PROMPT = """Solve the math problem independently and return exactly one JSON object.
Use this ordered SWAP schema:
Goal, Initial state, Initial graph, Plan, then one or more numbered
Action i, State i, Graph i triples, and Final answer.
Every graph must contain Statement and Entailment mappings. Statement IDs are
globally unique s1, s2, ... identifiers. A given fact has entailment
"Given condition"; a derived fact has a non-empty list of earlier statement
IDs. Return JSON only, without markdown fences or text outside the object."""

GENERIC_SYSTEM_PROMPT = """Complete the task independently and return exactly one JSON object.
Use this ordered SWAP schema:
Goal, Initial state, Initial graph, Plan, then one or more numbered
Action i, State i, Graph i triples, and Final answer.
Every graph must contain Statement and Entailment mappings. Statement IDs are
globally unique s1, s2, ... identifiers. A given fact has entailment
"Given condition"; a derived fact has a non-empty list of earlier statement
IDs. Return JSON only, without markdown fences or text outside the object."""

FINAL_ANSWER_INSTRUCTIONS = {
    "gsm8k": "The Final answer must contain only the concise numeric answer.",
    "math500": "The Final answer must contain only the concise mathematical answer.",
    "folio": "The Final answer must be exactly one of: True, False, Uncertain.",
    "reclor": (
        "The Final answer must be exactly the zero-based index 0, 1, 2, or 3 "
        "of the selected option."
    ),
    "humaneval": (
        "The Final answer must be a string containing the complete executable "
        "Python implementation, without Markdown fences or explanation."
    ),
    "mbpp": (
        "The Final answer must be a string containing the complete executable "
        "Python solution, without Markdown fences or explanation."
    ),
}


def build_generator_messages(question: str, benchmark: str) -> list[dict[str, str]]:
    system = MATH_SYSTEM_PROMPT if benchmark in {"gsm8k", "math500"} else GENERIC_SYSTEM_PROMPT
    instruction = FINAL_ANSWER_INSTRUCTIONS.get(benchmark)
    if instruction:
        system = f"{system}\n{instruction}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Problem:\n{question.strip()}"},
    ]


def _metadata(benchmark: str, split: str, index: int, dataset: str, config: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "split": split,
        "source_index": index,
        "source_dataset": dataset,
        "source_config": config,
    }


def load_benchmark(benchmark: str, split: str) -> list[dict[str, Any]]:
    """Load one benchmark and normalize it to id/question/gold_answer records."""
    from datasets import load_dataset

    supported = BENCHMARK_SPLITS.get(benchmark)
    if supported is None:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if split not in supported:
        raise ValueError(f"Unsupported {benchmark} split {split!r}; choose from {supported}")

    records: list[dict[str, Any]] = []
    if benchmark == "gsm8k":
        dataset_id, config = "openai/gsm8k", "main"
        dataset = load_dataset(dataset_id, config, split=split)
        for index, row in enumerate(dataset):
            gold = str(row["answer"]).rsplit("####", 1)[-1].strip()
            records.append({
                **_metadata(benchmark, split, index, dataset_id, config),
                "id": f"gsm8k_{split}_{index}",
                "question": row["question"],
                "gold_answer": gold,
            })
    elif benchmark == "math500":
        dataset_id, config = "sxiong/MATH-500", "default"
        dataset = load_dataset(dataset_id, split=split)
        for index, row in enumerate(dataset):
            records.append({
                **_metadata(benchmark, split, index, dataset_id, config),
                "id": str(row["unique_id"]).removesuffix(".json").replace("/", "_"),
                "question": row["problem"],
                "gold_answer": row["answer"],
                "subject": row.get("subject"),
                "level": row.get("level"),
            })
    elif benchmark == "folio":
        dataset_id, config = "tasksource/folio", "default"
        dataset = load_dataset(dataset_id, split=split)
        for index, row in enumerate(dataset):
            question = (
                "Determine whether the conclusion is true, false, or uncertain "
                "based only on the premises.\n\n"
                f"Premises:\n{row['premises'].strip()}\n\n"
                f"Conclusion:\n{row['conclusion'].strip()}"
            )
            records.append({
                **_metadata(benchmark, split, index, dataset_id, config),
                "id": f"folio_{split}_{row['example_id']}",
                "question": question,
                "gold_answer": row["label"],
                "story_id": row["story_id"],
                "example_id": row["example_id"],
            })
    elif benchmark == "reclor":
        dataset_id, config = "sxiong/ReClor", "default"
        dataset = load_dataset(dataset_id, split=split)
        for index, row in enumerate(dataset):
            choices = list(row["answers"])
            formatted = "\n".join(f"{i}. {choice}" for i, choice in enumerate(choices))
            label = int(row["label"])
            question = (
                f"Context:\n{row['context'].strip()}\n\n"
                f"Question:\n{row['question'].strip()}\n\nOptions:\n{formatted}"
            )
            records.append({
                **_metadata(benchmark, split, index, dataset_id, config),
                "id": f"reclor_{row['id_string']}",
                "question": question,
                "gold_answer": label if label >= 0 else None,
                "choices": choices,
            })
    elif benchmark == "humaneval":
        dataset_id, config = "openai/openai_humaneval", "openai_humaneval"
        dataset = load_dataset(dataset_id, config, split=split)
        for index, row in enumerate(dataset):
            task_number = str(row["task_id"]).split("/")[-1]
            records.append({
                **_metadata(benchmark, split, index, dataset_id, config),
                "id": f"humaneval_{split}_{task_number}",
                "question": "Complete the following Python function correctly:\n\n" + row["prompt"].rstrip(),
                "gold_answer": row["canonical_solution"],
                "code_prompt": row["prompt"],
                "test_code": row["test"],
                "entry_point": row["entry_point"],
            })
    elif benchmark == "mbpp":
        dataset_id, config = "google-research-datasets/mbpp", "full"
        dataset = load_dataset(dataset_id, config, split=split)
        for index, row in enumerate(dataset):
            tests = list(row["test_list"])
            question = row["text"].strip() + "\n\nThe solution must pass these public tests:\n" + "\n".join(tests)
            records.append({
                **_metadata(benchmark, split, index, dataset_id, config),
                "id": f"mbpp_{split}_{row['task_id']}",
                "question": question,
                "gold_answer": row["code"],
                "test_list": tests,
                "test_setup_code": row.get("test_setup_code", ""),
                "challenge_test_list": list(row.get("challenge_test_list", [])),
            })
    return records


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(text, str) or not text.strip():
        return None, "empty response"
    stripped = text.strip()
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        stripped = stripped[newline + 1 :] if newline >= 0 else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    start = stripped.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        value, end = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as error:
        return None, f"invalid JSON: {error.msg} at position {error.pos}"
    if stripped[start + end :].strip() not in {"", "```"}:
        return None, "text found after JSON object"
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def validate_trajectory(trajectory: Any) -> dict[str, Any]:
    """Validate the ordered SWAP fields without consulting a gold answer."""
    format_errors: list[str] = []
    graph_errors: list[str] = []
    if not isinstance(trajectory, dict):
        format_errors.append("trajectory must be an object")
        return _validation_payload(format_errors, graph_errors, 0)
    keys = list(trajectory)
    if len(keys) < 8:
        format_errors.append("trajectory has too few top-level fields")
        return _validation_payload(format_errors, graph_errors, 0)
    if keys[:4] != ["Goal", "Initial state", "Initial graph", "Plan"]:
        format_errors.append("first fields must be Goal, Initial state, Initial graph, Plan")
    if keys[-1] != "Final answer":
        format_errors.append("last field must be Final answer")
    middle = keys[4:-1]
    step_count = len(middle) // 3 if len(middle) % 3 == 0 else 0
    if not step_count:
        format_errors.append("reasoning fields must form Action/State/Graph triples")
    for index in range(1, step_count + 1):
        expected = [f"Action {index}", f"State {index}", f"Graph {index}"]
        if middle[(index - 1) * 3 : index * 3] != expected:
            format_errors.append(f"step {index} fields must be {expected}")
    text_fields = ["Goal", "Initial state", "Plan", "Final answer"]
    for index in range(1, step_count + 1):
        text_fields.extend([f"Action {index}", f"State {index}"])
    for name in text_fields:
        if not isinstance(trajectory.get(name), str) or not trajectory.get(name, "").strip():
            format_errors.append(f"{name} must be a non-empty string")
    if not format_errors:
        graph_names = ["Initial graph"] + [f"Graph {i}" for i in range(1, step_count + 1)]
        seen: set[str] = set()
        for graph_index, name in enumerate(graph_names):
            graph = trajectory.get(name)
            if not isinstance(graph, dict):
                graph_errors.append(f"{name} must be an object")
                continue
            statements, entailments = graph.get("Statement"), graph.get("Entailment")
            if not isinstance(statements, dict) or not statements:
                graph_errors.append(f"{name}.Statement must be a non-empty object")
                continue
            if not isinstance(entailments, dict) or set(statements) != set(entailments):
                graph_errors.append(f"{name} Statement and Entailment IDs must match")
                continue
            current = set(statements)
            for statement_id, statement in statements.items():
                if not re.fullmatch(r"s[1-9][0-9]*", str(statement_id)):
                    graph_errors.append(f"{name} has invalid statement ID {statement_id!r}")
                if statement_id in seen:
                    graph_errors.append(f"statement ID {statement_id} is repeated")
                if not isinstance(statement, str) or not statement.strip():
                    graph_errors.append(f"statement {statement_id} must be non-empty")
                entailment = entailments.get(statement_id)
                if graph_index == 0 and entailment != "Given condition":
                    graph_errors.append(f"initial statement {statement_id} must use Given condition")
                if graph_index > 0:
                    if not isinstance(entailment, list) or not entailment:
                        graph_errors.append(f"derived statement {statement_id} needs dependencies")
                    else:
                        for dependency in entailment:
                            if dependency not in seen and dependency not in current:
                                graph_errors.append(f"{statement_id} references unknown {dependency}")
            seen.update(current)
    return _validation_payload(format_errors, graph_errors, step_count)


def _validation_payload(format_errors: list[str], graph_errors: list[str], step_count: int) -> dict[str, Any]:
    return {
        "format_valid": not format_errors,
        "graph_valid": not format_errors and not graph_errors,
        "format_errors": format_errors,
        "graph_errors": graph_errors,
        "step_count": step_count,
    }


def answer_key(answer: Any, benchmark: str) -> str:
    value = str(answer or "").strip()
    if not value:
        return ""
    if benchmark in {"humaneval", "mbpp"}:
        normalized = "\n".join(line.rstrip() for line in value.splitlines()).strip()
        return "code:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    normalized = value.lower().rstrip(".")
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace("\\(", "").replace("\\)", "")
    normalized = re.sub(r"\s+", "", normalized)
    if benchmark == "reclor":
        match = re.fullmatch(r"(?:option|answer|choice)?[\(\[]?([0-3a-d])[\)\]]?", normalized)
        if match:
            token = match.group(1)
            return str({"a": 0, "b": 1, "c": 2, "d": 3}.get(token, token))
    return normalized


def completion_quality(cumulative_logprob: Any, token_count: int) -> float:
    if cumulative_logprob is None or token_count < 1:
        return float("-inf")
    return float(cumulative_logprob) / token_count


def build_candidate(
    record: dict[str, Any],
    text: str,
    token_ids: Sequence[int],
    cumulative_logprob: Any,
    source: str,
    temperature: float,
    sample_index: int,
) -> dict[str, Any]:
    trajectory, parse_error = extract_json_object(text)
    validation = validate_trajectory(trajectory) if trajectory is not None else None
    answer = trajectory.get("Final answer", "") if trajectory is not None else ""
    return {
        "source": source,
        "temperature": temperature,
        "sample_index": sample_index,
        "response": text,
        "trajectory": trajectory,
        "parse_error": parse_error,
        "validation": validation,
        "answer": answer,
        "answer_key": answer_key(answer, record["benchmark"]),
        "token_count": len(token_ids),
        "policy_quality": completion_quality(cumulative_logprob, len(token_ids)),
    }


def stable_seed(*values: Any) -> int:
    payload = "\0".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def finite_quality(candidate: dict[str, Any]) -> float:
    value = float(candidate.get("policy_quality", float("-inf")))
    return value if math.isfinite(value) else float("-inf")


def majority_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [candidate for candidate in candidates if candidate.get("answer_key")]
    if not valid:
        return max(candidates, key=finite_quality)
    counts = Counter(str(candidate["answer_key"]) for candidate in valid)
    qualities = {
        key: sum(finite_quality(candidate) for candidate in valid if candidate["answer_key"] == key)
        for key in counts
    }
    winning = max(counts, key=lambda key: (counts[key], qualities[key], key))
    return max(
        (candidate for candidate in valid if candidate["answer_key"] == winning),
        key=finite_quality,
    )


def _clean(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_trajectory(candidate: dict[str, Any], include_graph: bool) -> str:
    trajectory = candidate.get("trajectory")
    if not isinstance(trajectory, dict):
        return str(candidate.get("response") or "").strip() + f"\nClaimed answer: {candidate.get('answer', '')}"
    lines: list[str] = []
    if trajectory.get("Plan"):
        lines.append(f"Plan: {_clean(trajectory['Plan'])}")
    if include_graph and trajectory.get("Initial graph"):
        lines.append(f"Initial graph: {_clean(trajectory['Initial graph'])}")
    order = {"Action": 0, "State": 1, "Graph": 2}
    indexed: list[tuple[int, int, str]] = []
    pattern = r"(Action|State|Graph) (\d+)" if include_graph else r"(Action|State) (\d+)"
    for key in trajectory:
        match = re.fullmatch(pattern, str(key))
        if match:
            indexed.append((int(match.group(2)), order[match.group(1)], str(key)))
    for _, _, key in sorted(indexed):
        lines.append(f"{key}: {_clean(trajectory[key])}")
    lines.append(f"Claimed answer: {candidate.get('answer', '')}")
    return "\n".join(lines)


def truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    marker = "\n...[truncated]...\n"
    remaining = max_chars - len(marker)
    head = int(remaining * 0.7)
    return text[:head] + marker + text[-(remaining - head) :]


def build_selector_example(
    result: dict[str, Any],
    max_options: int = 8,
    max_trajectory_chars: int = 1400,
    include_graph: bool = True,
    seed: int = 42,
    variant_index: int = 0,
) -> dict[str, Any] | None:
    candidates = result["candidates"]
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate.get("answer_key"):
            by_answer[str(candidate["answer_key"])].append(candidate)
    if len(by_answer) < 2:
        return None
    ranked = sorted(
        by_answer,
        key=lambda key: (len(by_answer[key]), max(finite_quality(row) for row in by_answer[key]), key),
        reverse=True,
    )[:max_options]
    record = result["record"]
    prefix_id = f"{record['benchmark']}:{record['split']}:{record['id']}"
    rng = random.Random(stable_seed(seed, prefix_id, variant_index))
    rng.shuffle(ranked)
    options: list[dict[str, Any]] = []
    for index, key in enumerate(ranked):
        group = by_answer[key]
        representative = max(
            group,
            key=lambda candidate: (
                finite_quality(candidate),
                int(candidate.get("sample_index", -1)),
            ),
        )
        options.append({
            "label": LABELS[index],
            "answer_key": key,
            "answer": representative.get("answer", ""),
            "support": len(group),
            "trajectory": truncate_text(
                compact_trajectory(representative, include_graph), max_trajectory_chars
            ),
        })
    lines = [
        (
            "Compare the candidate trajectories directly. Identify what they agree on and "
            "the decisive reasoning or arithmetic difference between them. Support counts "
            "are only a weak prior; do not solve the problem again from scratch."
        ),
        "Give a concise comparison and end with exactly `Selection: Candidate X`, where X is the option label.",
        "",
        f"Question: {result['record']['question']}",
        "",
    ]
    for option in options:
        lines.extend([
            f"Candidate {option['label']} (support={option['support']}):",
            option["trajectory"],
            "",
        ])
    return {
        "id": result["record"]["id"],
        "prompt": "\n".join(lines).strip(),
        "options": [{key: option[key] for key in ("label", "answer_key", "answer", "support")} for option in options],
    }


def parse_selected_label(text: str) -> str | None:
    matches = re.findall(r"\bCandidate\s*([A-Q])\b", text.strip(), flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    if len(text.strip()) <= 32:
        match = re.search(r"\b([A-Q])\b", text.strip(), flags=re.IGNORECASE)
        return match.group(1).upper() if match else None
    return None


def ensure_grading_available(benchmark: str) -> None:
    """Fail before model loading when an optional benchmark grader is missing."""
    if benchmark != "math500":
        return
    try:
        from grading.grader import grade_answer as _grade_math_answer  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "MATH500 grading requires pylatexenc==2.10 in the active environment"
        ) from error


def grade_answer(answer: str, record: dict[str, Any]) -> dict[str, Any]:
    """Grade one final prediction; imports the legacy math grader lazily."""
    benchmark = record["benchmark"]
    if record.get("gold_answer") is None and benchmark != "humaneval":
        return {"correct": None, "error": "gold answer is not public"}
    try:
        if benchmark == "gsm8k":
            predicted = answer_key(answer, benchmark)
            gold = answer_key(record["gold_answer"], benchmark)
            correct, error = predicted == gold, None
        elif benchmark == "math500":
            from grading.grader import grade_answer as grade_math_answer

            correct, error = grade_math_answer(answer, str(record["gold_answer"])), None
        elif benchmark == "folio":
            correct = answer.strip().lower().rstrip(".") == str(record["gold_answer"]).strip().lower()
            error = None
        elif benchmark == "reclor":
            correct, error = _grade_reclor(answer, record), None
        elif benchmark == "humaneval":
            correct, error = _run_python(_humaneval_source(answer, record))
        elif benchmark == "mbpp":
            correct, error = _run_python(_mbpp_source(answer, record))
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        return {"correct": bool(correct), "error": error}
    except Exception as exception:
        return {"correct": False, "error": f"{type(exception).__name__}: {exception}"}


def _grade_reclor(answer: str, record: dict[str, Any]) -> bool:
    key = answer_key(answer, "reclor")
    if key in {"0", "1", "2", "3"}:
        return int(key) == int(record["gold_answer"])
    choices = record.get("choices", [])
    normalized = answer.strip().lower().rstrip(".")
    return any(
        index == int(record["gold_answer"]) and normalized == str(choice).strip().lower().rstrip(".")
        for index, choice in enumerate(choices)
    )


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*)\n```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip("\n")
    # HumanEval answers may be an indented function body. Preserve its leading
    # spaces so textwrap.dedent can recover the body indentation correctly.
    return text.strip("\n")


def _run_python(source: str, timeout_seconds: int = 12) -> tuple[bool, str | None]:
    with tempfile.TemporaryDirectory(prefix="swap-multibench-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    (
                        "ulimit -t 8; ulimit -v 2097152; ulimit -f 32768; "
                        "ulimit -n 64; exec \"$1\" -I \"$2\""
                    ),
                    "swap-multibench",
                    sys.executable,
                    str(path),
                ],
                cwd=directory,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONHASHSEED": "0",
                    "PYTHONIOENCODING": "utf-8",
                },
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return False, f"execution timed out after {timeout_seconds}s"
    if completed.returncode == 0:
        return True, None
    output = (completed.stderr or completed.stdout).strip()
    return False, output[-2000:] or f"process exited {completed.returncode}"


def _humaneval_source(answer: str, record: dict[str, Any]) -> str:
    code = _strip_code_fence(answer)
    prompt, entry_point = record["code_prompt"], record["entry_point"]
    if re.search(rf"(?m)^(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", code):
        definition = re.search(rf"(?m)^(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", prompt)
        candidate = (prompt[: definition.start()] if definition else "") + code
    else:
        candidate = prompt + textwrap.indent(textwrap.dedent(code), "    ")
    return candidate.rstrip() + "\n\n" + record["test_code"].strip() + f"\n\ncheck({entry_point})\n"


def _mbpp_source(answer: str, record: dict[str, Any]) -> str:
    sections: list[str] = [_strip_code_fence(answer), record.get("test_setup_code", "")]
    sections.extend(record["test_list"])
    sections.extend(record.get("challenge_test_list", []))
    return "\n\n".join(section for section in sections if str(section).strip()) + "\n"


def accuracy_summary(grades: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(grades)
    public = [row for row in rows if row.get("correct") is not None]
    correct = sum(bool(row["correct"]) for row in public)
    return {
        "total": len(rows),
        "graded": len(public),
        "correct": correct,
        "accuracy": correct / len(public) if public else None,
    }
