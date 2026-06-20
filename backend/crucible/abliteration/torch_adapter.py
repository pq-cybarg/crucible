# Real ModelAdapter backed by transformers/torch. Implements the protocol used by
# AbliterationPipeline + diagnosis so refusal extraction / orthogonalization run on
# actual HF weights. Requires torch + transformers (installed separately from the
# core, which stays numpy-only).
import numpy as np


def _pick_device(explicit: str | None):
    import torch
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TorchModelAdapter:
    def __init__(self, model, tokenizer, device: str | None = None):
        self.device = _pick_device(device)
        self.model = model.to(self.device).eval()
        self.tok = tokenizer
        self.hidden_size = int(model.config.hidden_size)
        self.num_layers = int(model.config.num_hidden_layers)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "TorchModelAdapter":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        return cls(model, tok, device)

    def _encode(self, prompt: str):
        if getattr(self.tok, "chat_template", None):
            enc = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt")
        else:
            enc = self.tok(prompt, return_tensors="pt")
        # Normalize to a plain input_ids tensor (apply_chat_template may return a
        # BatchEncoding/dict in transformers >=5, or a bare tensor in older versions).
        if hasattr(enc, "input_ids"):
            return enc.input_ids
        if isinstance(enc, dict):
            return enc["input_ids"]
        return enc

    def hidden_at(self, input_ids, layer: int) -> np.ndarray:
        import torch
        with torch.no_grad():
            out = self.model(input_ids.to(self.device), output_hidden_states=True)
        return out.hidden_states[layer][0, -1, :].float().cpu().numpy()

    def activations(self, prompts: list[str], layer: int) -> np.ndarray:
        return np.array([self.hidden_at(self._encode(p), layer) for p in prompts])

    def writing_matrices(self) -> list[str]:
        names: list[str] = []
        for i in range(self.num_layers):
            names.append(f"model.layers.{i}.self_attn.o_proj.weight")
            names.append(f"model.layers.{i}.mlp.down_proj.weight")
        return names

    def _param(self, name: str):
        params = dict(self.model.named_parameters())
        if name not in params:
            raise KeyError(name)
        return params[name]

    def get_matrix(self, name: str) -> np.ndarray:
        return self._param(name).detach().float().cpu().numpy()

    def set_matrix(self, name: str, W: np.ndarray) -> None:
        import torch
        p = self._param(name)
        with torch.no_grad():
            p.copy_(torch.tensor(np.asarray(W), dtype=p.dtype, device=p.device))

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
