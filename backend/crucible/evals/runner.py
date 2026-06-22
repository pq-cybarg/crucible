from __future__ import annotations
from typing import Callable

from crucible.evals.scoring import extract_choice, mc_accuracy

Solver = Callable[[str], str]
LETTERS = "ABCDE"


def format_mc_prompt(item: dict) -> str:
    lines = [item["question"], ""]
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{LETTERS[i]}) {choice}")
    lines.append("\nAnswer with the single letter of the correct choice.")
    return "\n".join(lines)


def run_mc_benchmark(items: list[dict], solver: Solver) -> dict:
    results = []
    predictions: list[str] = []
    answers: list[str] = []
    for item in items:
        raw = solver(format_mc_prompt(item))
        predicted = extract_choice(raw) or ""
        results.append({"id": item["id"], "predicted": predicted,
                        "answer": item["answer"], "correct": predicted == item["answer"]})
        predictions.append(predicted)
        answers.append(item["answer"])
    return {"accuracy": mc_accuracy(predictions, answers), "n": len(items), "results": results}
