"""Small, local LoRA adapters for TrOCR; base checkpoints remain unchanged."""
import json
import hashlib
from pathlib import Path


def install_adapter(model, rank=8, alpha=16):
    import torch
    if rank < 1 or alpha <= 0:
        raise ValueError('adapter rank and alpha must be positive')

    class AdapterLinear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features))
            self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank))
            torch.nn.init.normal_(self.lora_a, std=0.01)

        def forward(self, value):
            return self.base(value) + ((value @ self.lora_a.t()) @ self.lora_b.t()) * (alpha/rank)

    targets = [(name, module) for name, module in model.decoder.named_modules()
               if isinstance(module, torch.nn.Linear) and name.endswith(('q_proj', 'v_proj'))]
    if not targets:
        raise ValueError('checkpoint has no supported decoder attention projections')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, module in targets:
        parent_name, attribute = name.rsplit('.', 1)
        setattr(model.decoder.get_submodule(parent_name), attribute, AdapterLinear(module))
    return {name: parameter for name, parameter in model.named_parameters()
            if name.endswith(('lora_a', 'lora_b'))}


def load_adapter(model, directory, base_model):
    from safetensors.torch import load_file
    import torch
    directory = Path(directory)
    metadata = json.loads((directory / 'adapter.json').read_text())
    if metadata['base_model'] != base_model:
        raise ValueError('adapter base checkpoint does not match AWP_TROCR_MODEL')
    if hashlib.sha256((directory / 'adapter.safetensors').read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('adapter checksum mismatch')
    parameters = install_adapter(model, metadata['rank'], metadata['alpha'])
    weights = load_file(str(directory / 'adapter.safetensors'))
    if set(parameters) != set(weights):
        raise ValueError('adapter tensors do not match the model architecture')
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != weights[name].shape or not torch.isfinite(weights[name]).all():
                raise ValueError('invalid adapter tensor: ' + name)
            parameter.copy_(weights[name])
            parameter.requires_grad_(False)
