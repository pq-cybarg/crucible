# The canonical industry-standard benchmark suite (EleutherAI lm-evaluation-harness
# task names). These are the same tasks behind public leaderboards.
CANONICAL_SUITE: list[dict] = [
    {"task": "mmlu", "label": "MMLU", "detail": "57 subjects · knowledge", "primary": "acc"},
    {"task": "mmlu_pro", "label": "MMLU-Pro", "detail": "harder, 10-choice", "primary": "acc"},
    {"task": "gpqa_main_zeroshot", "label": "GPQA", "detail": "graduate science", "primary": "acc"},
    {"task": "gsm8k", "label": "GSM8K", "detail": "grade-school math", "primary": "exact_match"},
    {"task": "hendrycks_math", "label": "MATH", "detail": "competition math", "primary": "exact_match"},
    {"task": "arc_challenge", "label": "ARC-Challenge", "detail": "science reasoning", "primary": "acc_norm"},
    {"task": "hellaswag", "label": "HellaSwag", "detail": "commonsense", "primary": "acc_norm"},
    {"task": "truthfulqa_mc2", "label": "TruthfulQA", "detail": "truthfulness", "primary": "acc"},
    {"task": "winogrande", "label": "WinoGrande", "detail": "coreference", "primary": "acc"},
    {"task": "ifeval", "label": "IFEval", "detail": "instruction following", "primary": "prompt_level_strict_acc"},
    {"task": "bbh", "label": "BIG-Bench-Hard", "detail": "hard reasoning", "primary": "acc_norm"},
]

SUITE_TASKS = [s["task"] for s in CANONICAL_SUITE]
