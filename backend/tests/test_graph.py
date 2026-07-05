import pytest
from crucible.graph import (cascade, execute_graph, final_outputs, majority, make_acceptor,
                            stage_text, topo_order, vote)


PIPE = [
    {"id": "stt", "inputs": [], "kind": "model"},
    {"id": "llm", "inputs": ["stt"], "kind": "model"},
    {"id": "tts", "inputs": ["llm"], "kind": "model"},
]


def test_topo_order_linear():
    assert topo_order(PIPE) == ["stt", "llm", "tts"]


def test_topo_detects_cycle():
    cyc = [{"id": "a", "inputs": ["b"]}, {"id": "b", "inputs": ["a"]}]
    with pytest.raises(ValueError):
        topo_order(cyc)


def test_topo_unknown_dep():
    with pytest.raises(ValueError):
        topo_order([{"id": "a", "inputs": ["ghost"]}])


def test_execute_chains_outputs():
    # each stage appends its id — proves inputs flow through the DAG in order
    def run(stage, inputs):
        prev = next(iter(inputs.values()))
        return f"{prev}->{stage['id']}"
    out = execute_graph(PIPE, run, initial="in")
    assert out["stt"] == "in->stt"
    assert out["tts"] == "in->stt->llm->tts"


def test_diamond_graph_merges():
    stages = [
        {"id": "root", "inputs": []},
        {"id": "a", "inputs": ["root"]},
        {"id": "b", "inputs": ["root"]},
        {"id": "merge", "inputs": ["a", "b"]},
    ]
    def run(stage, inputs):
        return "+".join(sorted(str(v) for v in inputs.values())) or stage["id"]
    out = execute_graph(stages, run, initial="x")
    assert topo_order(stages)[0] == "root"
    assert out["merge"] == "x+x"        # merge received BOTH branches (a and b), each == "x"


def test_final_outputs_are_terminals():
    def run(stage, inputs): return stage["id"]
    out = execute_graph(PIPE, run)
    assert final_outputs(PIPE, out) == {"tts": "tts"}


# --- cascade (cheap -> escalate) ---

def test_make_acceptor_len_include_exclude():
    acc = make_acceptor(min_len=3, must_include=["yes"], must_exclude=["i don't know"])
    assert acc("yes indeed") is True
    assert acc("no") is False                       # too short + missing 'yes'
    assert acc("yes but i don't know really") is False   # excluded phrase
    assert acc("   ") is False


def test_cascade_stops_at_first_accepted():
    calls = []
    def p(label, out):
        def produce():
            calls.append(label); return out
        return produce
    # cheap model already good -> no escalation, strong model never called
    res = cascade([("cheap", p("cheap", "great answer")), ("strong", p("strong", "x"))],
                  make_acceptor(min_len=3))
    assert res["chosen"] == "cheap" and res["escalated"] is False and res["accepted"] is True
    assert calls == ["cheap"]                        # strong producer not invoked


def test_cascade_escalates_when_cheap_rejected():
    def p(out):
        return lambda: out
    # cheap hedges ("i don't know") -> escalate to strong
    acc = make_acceptor(min_len=3, must_exclude=["i don't know"])
    res = cascade([("cheap", p("i don't know")), ("strong", p("the answer is 42"))], acc)
    assert res["chosen"] == "strong" and res["escalated"] is True and res["accepted"] is True
    assert res["tried"] == ["cheap", "strong"]


def test_cascade_returns_last_when_none_accepted():
    acc = make_acceptor(must_include=["impossible-token"])
    res = cascade([("a", lambda: "one"), ("b", lambda: "two")], acc)
    assert res["accepted"] is False and res["chosen"] == "b" and res["output"] == "two"


# --- verifier ensemble (fan out -> vote) ---

def test_majority_and_ties():
    assert majority(["a", "b", "a", "c", "a"]) == "a"
    assert majority(["x", "y"]) == "x"               # tie -> first seen
    assert majority([" ", "", None]) == ""


def test_vote_strategies_and_agreement():
    maj = vote(["yes", "yes", "no"], "majority")
    assert maj["result"] == "yes" and maj["n"] == 3 and maj["agreement"] == round(2 / 3, 3)
    assert vote(["a", "b"], "concat")["result"] == "a\n\nb"
    assert vote(["", "first-nonempty", "x"], "first")["result"] == "first-nonempty"
    assert vote([], "majority")["agreement"] == 0.0


def test_stage_text_unwraps_rich_stage_outputs():
    assert stage_text({"output": "casc"}) == "casc"      # cascade dict
    assert stage_text({"result": "voted"}) == "voted"    # vote dict
    assert stage_text("plain") == "plain"
    assert stage_text(None) == ""
