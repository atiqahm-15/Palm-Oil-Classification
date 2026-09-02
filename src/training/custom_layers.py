"""Import side effect: register keras_cv_attention_models' custom layers with Keras.

`tf.keras.models.load_model()` can only deserialize a layer whose class is in Keras's
serialization registry, and a class lands there when the module defining it is imported --
that is what runs its @register_keras_serializable decorator. FastViT (kecam) is built from
custom layers such as ChannelAffine, so loading a saved FastViT run without importing kecam
first fails with:

    Could not locate class 'ChannelAffine'. Make sure custom classes and functions are
    decorated with `@keras.saving.register_keras_serializable()`. If they are already
    decorated, make sure they are all imported so that the decorator is run before trying
    to load them.

The import is guarded because kecam is optional and deliberately not universal: it is
installed in the WSL GPU venv (/opt/tfgpu) but not in the Windows env, and only FastViT runs
need it. A missing kecam must not stop MobileNetV2 / EfficientNetB0 runs from being evaluated,
calibrated, or plotted on a machine that never trains FastViT.

Import this module for its side effect, before any load_model() call:

    import custom_layers  # noqa: F401

This is the same pattern as `import ordinal`, which registers CoralLayer/CoralLogitsToProbs
for the CORAL experiment's runs.
"""

KECAM_AVAILABLE = False

try:
    # Importing the fastvit module pulls in kecam's attention_layers/mlp_family modules,
    # which is where ChannelAffine and the rest actually register themselves.
    from keras_cv_attention_models import fastvit  # noqa: F401

    KECAM_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent by design
    pass
