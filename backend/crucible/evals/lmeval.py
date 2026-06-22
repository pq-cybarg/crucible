# Integration with EleutherAI lm-evaluation-harness — the canonical benchmark tool.
# run_lmeval() drives a live OpenAI-compatible endpoint; parse_lmeval_results() is a
# pure transform over lm-eval's results dict (unit-tested without a model).


def parse_lmeval_results(results: dict) -> list[dict]:
    """Flatten lm-eval's {task: {"metric,filter": value, "metric_stderr,filter": se, ...}}
    into [{task, metric, filter, value, stderr}]."""
    rows: list[dict] = []
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for key, val in metrics.items():
            if key == "alias" or "_stderr," in key or key.endswith("_stderr"):
                continue
            if key.split(",")[0] in ("sample_len", "samples"):
                continue
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            metric, _, filt = key.partition(",")
            stderr_key = f"{metric}_stderr,{filt}" if filt else f"{metric}_stderr"
            se = metrics.get(stderr_key)
            rows.append({
                "task": task,
                "metric": metric,
                "filter": filt or None,
                "value": float(val),
                "stderr": float(se) if isinstance(se, (int, float)) and not isinstance(se, bool) else None,
            })
    return rows


def run_lmeval(endpoint: str, tasks: list[str], limit: int | None = None,
               model_name: str = "local", backend: str = "chat") -> list[dict]:
    """backend='chat' (generative, gsm8k/ifeval) or 'completions' (loglikelihood MC:
    mmlu/arc/hellaswag — needs an endpoint exposing /v1/completions with logprobs)."""
    import lm_eval

    base = endpoint.rstrip("/")
    if backend == "completions":
        model_args = {"model": model_name, "base_url": f"{base}/v1/completions",
                      "num_concurrent": 4, "tokenized_requests": False}
        out = lm_eval.simple_evaluate(model="local-completions", model_args=model_args,
                                      tasks=tasks, limit=limit)
    else:
        model_args = {"model": model_name, "base_url": f"{base}/v1/chat/completions",
                      "num_concurrent": 4, "tokenized_requests": False}
        out = lm_eval.simple_evaluate(model="local-chat-completions", model_args=model_args,
                                      tasks=tasks, limit=limit, apply_chat_template=True)
    return parse_lmeval_results(out["results"])
