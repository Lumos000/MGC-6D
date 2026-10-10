"""Narrow CUDA retry helpers shared by reconstruction and unit tests."""

from __future__ import annotations

import gc


def run_cudnn_stream_retry(build, reset_runner, device: str, torch_module):
    try:
        return build()
    except RuntimeError as exc:
        if "CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH" not in str(exc):
            raise
        reset_runner()
        gc.collect()
        torch = torch_module
        if not torch.cuda.is_available():
            raise
        cuda_device = torch.device(device)
        torch.cuda.synchronize(cuda_device)
        torch.cuda.empty_cache()
        default_stream = torch.cuda.default_stream(cuda_device)
        print(
            "[WARN] RaySt3R hit CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH; "
            "retrying once with a fresh runner on the default stream and cuDNN disabled."
        )
        with torch.cuda.device(cuda_device), torch.cuda.stream(default_stream):
            with torch.backends.cudnn.flags(enabled=False):
                result = build()
            torch.cuda.synchronize(cuda_device)
        return result


__all__ = ["run_cudnn_stream_retry"]
