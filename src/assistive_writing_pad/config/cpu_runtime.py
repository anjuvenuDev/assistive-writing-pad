"""Shared CPU limits and opt-in integer inference for 64-bit Raspberry Pi."""

import os
import platform


def configure_cpu(torch):
    threads = int(os.environ.get("AWP_TORCH_THREADS", str(min(4, os.cpu_count() or 1))))
    if threads < 1:
        raise ValueError("AWP_TORCH_THREADS must be positive")
    torch.set_num_threads(threads)


def optimize_cpu_model(model, torch):
    """Quantize Linear weights in place to avoid a second full model in RAM.

    Explicitly opt in: different quantization engines can change predictions,
    so target-device accuracy/latency must be evaluated before deployment.
    """
    configure_cpu(torch)
    enabled = os.environ.get("AWP_DYNAMIC_INT8", "0").lower() in {"1", "true", "yes"}
    if not enabled:
        return model
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("AWP_DYNAMIC_INT8 requires CPU inference")
    default_engine = "qnnpack" if platform.machine().lower() in {"aarch64", "arm64"} else "x86"
    engine = os.environ.get("AWP_QUANTIZED_ENGINE", default_engine)
    if engine not in torch.backends.quantized.supported_engines or engine == "none":
        raise ValueError(f"This PyTorch build does not support quantized engine {engine!r}")
    torch.backends.quantized.engine = engine
    torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
    )
    return model
