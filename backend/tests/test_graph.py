import pytest
from crucible.graph import execute_graph, final_outputs, topo_order


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
