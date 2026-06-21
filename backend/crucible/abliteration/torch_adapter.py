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

    def all_layer_activations(self, prompts: list[str]) -> np.ndarray:
        """One forward per prompt; returns (n_prompts, n_layers+1, hidden) last-token states."""
        import torch
        rows = []
        for p in prompts:
            ids = self._encode(p).to(self.device)
            with torch.no_grad():
                out = self.model(ids, output_hidden_states=True)
            rows.append([hs[0, -1, :].float().cpu().numpy() for hs in out.hidden_states])
        return np.array(rows)

    def _encode_messages(self, messages: list[dict]):
        if getattr(self.tok, "chat_template", None):
            enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        else:
            enc = self.tok("\n".join(m.get("content", "") for m in messages), return_tensors="pt")
        if hasattr(enc, "input_ids"):
            return enc.input_ids
        if isinstance(enc, dict):
            return enc["input_ids"]
        return enc

    def generate_chat(self, messages: list[dict], max_new_tokens: int = 128,
                     band_dirs: dict | None = None, coefficient: float = 1.0) -> str:
        """Chat-format generation with an OPTIONAL runtime steering recipe (per-layer
        banded ablation via hooks). Nondestructive: hooks removed after the call."""
        import torch
        ids = self._encode_messages(messages).to(self.device)
        handles = []
        if band_dirs:
            layers = self.decoder_layers()

            def make_hook(D):
                def hook(module, inp, out):
                    is_t = isinstance(out, tuple)
                    h = out[0] if is_t else out
                    proj = (h.to(torch.float32) @ D.T) @ D
                    h2 = h - coefficient * proj.to(h.dtype)
                    return (h2,) + tuple(out[1:]) if is_t else h2
                return hook

            for j, d in band_dirs.items():
                D = torch.tensor(np.asarray(d), dtype=torch.float32, device=self.device)
                handles.append(layers[int(j)].register_forward_hook(make_hook(D)))
        try:
            with torch.no_grad():
                out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=getattr(self.tok, "eos_token_id", None))
            return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        finally:
            for hd in handles:
                hd.remove()

    def inject_generate(self, prompt: str, direction, coefficient: float, layers,
                       max_new_tokens: int = 40) -> str:
        """Feature INSERTION (restoration): ADD coefficient*direction to the residual at
        the given layers — the additive complement of ablation. Nondestructive (hooks)."""
        import torch
        D = torch.tensor(np.asarray(direction), dtype=torch.float32, device=self.device)
        if D.ndim == 2:
            D = D.sum(0)
        decoders = self.decoder_layers()
        handles = []

        def hook(module, inp, out):
            is_t = isinstance(out, tuple)
            h = out[0] if is_t else out
            h2 = h + coefficient * D.to(h.dtype)
            return (h2,) + tuple(out[1:]) if is_t else h2

        for j in layers:
            handles.append(decoders[int(j)].register_forward_hook(hook))
        try:
            return self.generate(prompt, max_new_tokens)
        finally:
            for hd in handles:
                hd.remove()

    def token_layer_activations(self, prompt: str, direction) -> dict:
        """For one forward pass, project every token's residual at every layer onto the
        refusal direction -> a (n_layers+1 x n_tokens) heatmap of WHERE/WHEN refusal fires."""
        import torch
        ids = self._encode(prompt).to(self.device)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        D = torch.tensor(np.asarray(direction), dtype=torch.float32, device=self.device)
        matrix = [(hs[0].to(torch.float32) @ D).detach().cpu().numpy().tolist() for hs in out.hidden_states]
        tokens = [self.tok.decode([int(t)]) for t in ids[0].tolist()]
        return {"matrix": matrix, "tokens": tokens}

    def unembed_matrix(self) -> np.ndarray:
        return self.model.get_output_embeddings().weight.detach().float().cpu().numpy()

    def token_decode(self, token_id: int) -> str:
        return self.tok.decode([int(token_id)])

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

    def decoder_layers(self):
        # Qwen2/Llama-style: model.model.layers
        return self.model.model.layers

    def ablate_generate(self, prompt: str, directions, coefficient: float = 1.0,
                        max_new_tokens: int = 40) -> str:
        """Generate while projecting the refusal subspace out of every layer's residual
        via forward hooks. NONDESTRUCTIVE: weights are never modified; hooks are removed
        afterwards, so this is fully reversible (detach == swap back)."""
        import torch
        D = torch.tensor(np.asarray(directions), dtype=torch.float32, device=self.device)

        def hook(module, inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            proj = (h.to(torch.float32) @ D.T) @ D
            h2 = (h - coefficient * proj.to(h.dtype))
            return (h2,) + tuple(out[1:]) if is_tuple else h2

        handles = [layer.register_forward_hook(hook) for layer in self.decoder_layers()]
        try:
            return self.generate(prompt, max_new_tokens)
        finally:
            for hd in handles:
                hd.remove()

    def ablate_generate_banded(self, prompt: str, band_dirs: dict, coefficient: float = 1.0,
                              max_new_tokens: int = 40) -> str:
        """Per-layer, nondestructive ablation: each decoder layer in band_dirs gets a hook
        projecting out ITS OWN refusal subspace. Weights untouched; hooks removed after."""
        import torch
        layers = self.decoder_layers()
        handles = []

        def make_hook(D):
            def hook(module, inp, out):
                is_t = isinstance(out, tuple)
                h = out[0] if is_t else out
                proj = (h.to(torch.float32) @ D.T) @ D
                h2 = h - coefficient * proj.to(h.dtype)
                return (h2,) + tuple(out[1:]) if is_t else h2
            return hook

        for j, dirs in band_dirs.items():
            D = torch.tensor(np.asarray(dirs), dtype=torch.float32, device=self.device)
            handles.append(layers[int(j)].register_forward_hook(make_hook(D)))
        try:
            return self.generate(prompt, max_new_tokens)
        finally:
            for hd in handles:
                hd.remove()

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        import torch
        ids = self._encode(prompt).to(self.device)
        eos = getattr(self.tok, "eos_token_id", None)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=eos)
        new = out[0, ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True)

    def generate_chat(self, messages: list[dict], max_new_tokens: int = 128,
                     band_dirs=None, coefficient: float = 1.0) -> str:
        """OpenAI-style chat generation. If band_dirs is set, applies reversible runtime
        ablation via forward hooks (nondestructive) to every served token."""
        import torch
        if getattr(self.tok, "chat_template", None):
            enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            ids = enc.input_ids if hasattr(enc, "input_ids") else enc
        else:
            text = "\n".join(m.get("content", "") for m in messages)
            ids = self.tok(text, return_tensors="pt").input_ids
        ids = ids.to(self.device)

        handles = []
        if band_dirs:
            layers = self.decoder_layers()

            def make_hook(D):
                def hook(module, inp, out):
                    is_t = isinstance(out, tuple)
                    h = out[0] if is_t else out
                    proj = (h.to(torch.float32) @ D.T) @ D
                    h2 = h - coefficient * proj.to(h.dtype)
                    return (h2,) + tuple(out[1:]) if is_t else h2
                return hook

            for j, dirs in band_dirs.items():
                D = torch.tensor(np.asarray(dirs), dtype=torch.float32, device=self.device)
                handles.append(layers[int(j)].register_forward_hook(make_hook(D)))
        try:
            with torch.no_grad():
                out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=getattr(self.tok, "eos_token_id", None))
            return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        finally:
            for hd in handles:
                hd.remove()

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
