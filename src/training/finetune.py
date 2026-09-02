"""Block-level fine-tuning: unfreeze the last K blocks of a Keras backbone.

Replaces the earlier "unfreeze the last N layers" scheme, which was not comparable
across architectures: N=30 meant ~19.5% of MobileNetV2 (154 layers) but only ~12.6% of
EfficientNetB0 (237 layers), so the two baselines were fine-tuned to different depths
while appearing to share a hyperparameter. Blocks are the architectures' own semantic
unit and both happen to expose exactly 16 of them, so "last K blocks" is directly
comparable — and matches how the literature specifies fine-tuning depth (e.g. Suharjito
et al. unfreeze 3 convolutional blocks).

Layer-name prefixes are the block markers:
  MobileNetV2     block_1_... .. block_16_...        then a Conv_1/Conv_1_bn/out_relu tail
  EfficientNetB0  block1a_... .. block7a_...         then a top_conv/top_bn/top_activation tail
  FastViT_T8      stack1_block1_... .. stack4_block2_...  (no tail; last block ends the model)
The tail layers after the final block are part of the backbone's feature output, so they
unfreeze along with the last K blocks rather than being left frozen behind them.

IMPORTANT -- K is only comparable across architectures with the same block count. The two
CNNs both expose exactly 16 blocks, which is what made "last K blocks" a fair shared
hyperparameter. FastViT_T8 exposes 10 (4 stacks of depth 2/2/4/2), so the same K is a
*deeper* proportional unfreeze. Measured through unfreeze_last_blocks() itself:

    MobileNetV2     16 blocks   K=3  ->  29/154 layers   18.8%
    EfficientNetB0  16 blocks   K=3  ->  46/237 layers   19.4%
    FastViT_T8      10 blocks   K=2  ->  38/246 layers   15.4%
    FastViT_T8      10 blocks   K=3  ->  69/246 layers   28.0%

Note the CNNs' ~19% falls *between* FastViT's K=2 and K=3 -- no integer K matches it, since
FastViT's blocks are individually much larger. K=2 is the nearer of the two. Do not assume
K=3 means for FastViT what it means for the CNNs; a FastViT run must state which it used.
PROJECT_PLAN.md's decisions log records the campaign that runs both and brackets the CNNs.

(Layer counts are not simply blocks x size: stack transitions add downsample layers that
belong to no stackN_blockM group, which is why K=2 unfreezes 38 layers and not stack4's 50.)
"""
import re

# (regex, key builder) per naming scheme; key must sort in the model's own block order.
# Order matters only for disambiguation; the three patterns are mutually exclusive because
# each is anchored at ^ ("stack1_block1_" cannot match the bare "block..." patterns).
_BLOCK_PATTERNS = [
    (re.compile(r"^block_(\d+)_"), lambda m: f"{int(m.group(1)):02d}"),          # MobileNetV2
    (re.compile(r"^block(\d+)([a-z])_"), lambda m: f"{int(m.group(1)):02d}{m.group(2)}"),  # EfficientNet
    (re.compile(r"^stack(\d+)_block(\d+)_"),                                     # FastViT (kecam)
     lambda m: f"{int(m.group(1)):02d}{int(m.group(2)):02d}"),
]


def _block_tag(layer_name: str) -> str | None:
    for pattern, build_key in _BLOCK_PATTERNS:
        m = pattern.match(layer_name)
        if m:
            return build_key(m)
    return None


def block_tags_in_order(base) -> list[str]:
    """Distinct block tags, in the order they appear in the model."""
    tags = []
    for layer in base.layers:
        tag = _block_tag(layer.name)
        if tag is not None and (not tags or tags[-1] != tag):
            tags.append(tag)
    return tags


def unfreeze_last_blocks(base, n_blocks: int) -> dict:
    """Freeze `base`, then unfreeze its last `n_blocks` blocks and any trailing tail layers.

    Returns a summary dict for the run config so each run records the depth it actually
    fine-tuned at, not just the requested K.
    """
    tags = block_tags_in_order(base)
    if not tags:
        raise ValueError(
            f"No block-structured layer names found in {base.name}. Add its naming scheme "
            f"to _BLOCK_PATTERNS before using block-level fine-tuning with this backbone."
        )
    if not 1 <= n_blocks <= len(tags):
        raise ValueError(f"n_blocks must be in 1..{len(tags)} for {base.name}; got {n_blocks}")

    first_unfrozen_tag = tags[-n_blocks]
    cutoff = next(
        i for i, layer in enumerate(base.layers) if _block_tag(layer.name) == first_unfrozen_tag
    )

    base.trainable = True
    for layer in base.layers[:cutoff]:
        layer.trainable = False

    return {
        "n_blocks_unfrozen": n_blocks,
        "total_blocks": len(tags),
        "first_unfrozen_block": first_unfrozen_tag,
        "layers_unfrozen": len(base.layers) - cutoff,
        "total_layers": len(base.layers),
    }
