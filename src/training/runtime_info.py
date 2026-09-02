"""Capture the compute platform a run executed on, for its config.json.

Why this is worth recording: results are not comparable across compute backends. CPU runs
use oneDNN kernels; GPU runs use cuDNN, which selects different convolution algorithms with
different nondeterminism. Pooling a CPU run and a GPU run into one mean +/- std silently
confounds the platform with whatever variable the comparison is actually about - the same
class of mistake as the source-imbalance split bug and the BatchNorm training=False
regression already documented in PROJECT_PLAN.md section 8.

Recording it per run makes the boundary auditable after the fact, and lets
aggregate_seeds.py refuse to pool replicates that did not run on the same backend.

This module deliberately touches nothing about model construction or training.
"""
import platform

import tensorflow as tf


def compute_platform() -> dict:
    """Describe the backend this process will train on."""
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = []
    for g in gpus:
        try:
            details = tf.config.experimental.get_device_details(g)
            gpu_names.append(details.get("device_name", g.name))
        except Exception:
            # get_device_details is best-effort and not supported for every device type.
            gpu_names.append(g.name)

    return {
        "device": "GPU" if gpus else "CPU",
        "gpu_names": gpu_names,
        "n_gpus": len(gpus),
        "tf_version": tf.__version__,
        "os": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
    }


def describe(info: dict) -> str:
    """One-line human summary for run logs."""
    where = f"{info['device']}"
    if info["gpu_names"]:
        where += f" ({', '.join(info['gpu_names'])})"
    return f"{where} — TF {info['tf_version']} on {info['os']} {info['os_release']}, Python {info['python']}"
