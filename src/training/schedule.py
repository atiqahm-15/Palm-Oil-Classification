"""Two-stage training schedule -- the single definition of this project's training recipe.

Every Objective-2 architecture (train_mobilenetv2.py, train_efficientnetb0.py,
train_fastvit.py) drives its two stages through run_two_stage() here, so "all three
architectures share an identical recipe" is enforced by construction rather than by
copy-paste discipline across three files. Change the recipe here and it changes for every
architecture at once -- which is the only way the Phase 2 comparison table stays fair.

## Recipe v1 -> v2 (Sep 2026)

v1 -- every run in results/ up to and including the Aug-2026 GPU campaign -- capped
training at 10 frozen + 10 fine-tune epochs with a constant 1e-5 fine-tuning LR. Reading
the saved history.json of all 15 GPU runs (the 3 architectures plus the K=2 and K=4 depth
arms) afterwards showed:

    Stage 2 early-stopped:                    0/15
    Stage 1 early-stopped:                    6/15
    val_loss still falling at the final epoch: 13/15
    best val_accuracy AT the final epoch:     11/15

Stage 2 ran to its epoch cap in every single run: EarlyStopping(patience=4) never once
fired during fine-tuning. That it *did* fire in Stage 1 for 6 of those runs is what makes
this conclusive rather than a suspected dead callback -- the frozen head genuinely
plateaus and the callback correctly ends it, while the fine-tuning stage never got the
chance. The v1 numbers are therefore a floor on each architecture's accuracy, not its
ceiling.

The only two runs whose val_loss had turned back up (mobilenetv2_..._gpu_seed42 and
efficientnetb0_..._gpu_seed43) are the sole hints of a real plateau anywhere in the
campaign, and both had sibling seeds still climbing -- so MobileNetV2 is the nearest to
its ceiling without being demonstrably at it, and EfficientNetB0 is not near its own.

v2 trains to actual convergence instead:
  - epoch caps raised well clear of where runs actually stop (10 -> 30 frozen, 10 -> 40 fine-tune)
  - EarlyStopping patience 4 -> 8, so a plateau has to be real before it ends the run
  - fine-tuning LR 1e-5 -> 1e-4 under a cosine decay to ~0 across the stage

Two variables move at once (budget and LR). That is deliberate: v2 is an upgraded recipe
applied uniformly to all three architectures, not a controlled single-variable ablation.
The *architecture* comparison stays fair because every architecture gets the same v2. To
attribute the gain between "longer" and "faster" specifically, run one seed with
`--lr-schedule constant --finetune-lr 1e-5` and the v2 epoch caps.

Note this also puts the K=4 depth-control finding back in play. "EfficientNetB0 does not
benefit from deeper fine-tuning" was measured entirely under v1, where unfreezing more
layers meant more parameters sharing the same 10 epochs at 1e-5 -- too little budget to
exploit the extra capacity. Deeper fine-tuning needs longer training to pay off, so that
conclusion has to be re-measured under v2 before it goes in the report.

Runs record `recipe_version` in config.json, and aggregate_seeds.py refuses to pool across
versions: a v1 run and a v2 run are not replicates of the same process, for exactly the
reason a CPU run and a GPU run are not.
"""
import math

import tensorflow as tf

from finetune import unfreeze_last_blocks

RECIPE_VERSION = 2

DEFAULTS = {
    "epochs_frozen": 30,
    "epochs_finetune": 40,
    "frozen_lr": 1e-3,
    "finetune_lr": 1e-4,
    "patience": 8,
    "lr_schedule": "cosine",
}

# Flags that reproduce recipe v1 exactly, for re-running an old baseline under this code.
V1_FLAGS = ("--epochs-frozen 10 --epochs-finetune 10 --patience 4 "
            "--frozen-lr 1e-3 --finetune-lr 1e-5 --lr-schedule constant")


def add_training_args(parser, *, default_finetune_blocks: int) -> None:
    """Attach the args shared by every train_*.py, so the recipe surface is defined once."""
    parser.add_argument("--epochs-frozen", type=int, default=DEFAULTS["epochs_frozen"],
                        help="cap, not a target -- EarlyStopping decides when Stage 1 ends")
    parser.add_argument("--epochs-finetune", type=int, default=DEFAULTS["epochs_finetune"],
                        help="cap, not a target -- EarlyStopping decides when Stage 2 ends")
    parser.add_argument("--finetune-blocks", type=int, default=default_finetune_blocks)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frozen-lr", type=float, default=DEFAULTS["frozen_lr"])
    parser.add_argument("--finetune-lr", type=float, default=DEFAULTS["finetune_lr"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                        help="EarlyStopping patience on val_accuracy, applied to both stages")
    parser.add_argument("--lr-schedule", choices=["cosine", "plateau", "constant"],
                        default=DEFAULTS["lr_schedule"],
                        help="Stage 2 LR schedule. cosine: decay --finetune-lr to ~0 across "
                             "the stage. plateau: constant LR plus ReduceLROnPlateau. "
                             "constant: fixed LR (recipe v1 behaviour).")


def _finetune_optimizer(args, steps_per_epoch: int) -> tf.keras.optimizers.Optimizer:
    if args.lr_schedule == "cosine":
        # decay_steps spans the full epoch cap. If EarlyStopping fires first the cosine
        # simply ends part-way down, which is fine -- the alternative (decaying over the
        # actual run length) is not knowable in advance.
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.finetune_lr,
            decay_steps=max(1, args.epochs_finetune * steps_per_epoch),
        )
    else:
        lr = args.finetune_lr
    return tf.keras.optimizers.Adam(learning_rate=lr)


def run_two_stage(*, model, base, train_ds, valid_ds, ckpt_path, args, loss, n_train):
    """Run Stage 1 (frozen) then Stage 2 (fine-tune last K blocks) under the recipe above.

    Returns (history_frozen, history_finetune, finetune_info, schedule_info). The caller
    keeps ownership of model construction, evaluation, and config.json -- only the training
    recipe lives here.
    """
    steps_per_epoch = math.ceil(n_train / args.batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.frozen_lr),
        loss=loss,
        metrics=["accuracy"],
    )

    # One callbacks list is reused across both stages, exactly as recipe v1 did.
    # EarlyStopping resets its own state in on_train_begin, but ModelCheckpoint does not:
    # its `best` carries across the two fit() calls, so Stage 2 overwrites the checkpoint
    # only once it beats Stage 1's best val_accuracy. That is intended -- the saved weights
    # are the best across *both* stages -- and must not be "fixed" without re-running every
    # baseline in results/.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                           save_best_only=True, save_weights_only=True),
    ]
    if args.lr_schedule == "plateau":
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=max(2, args.patience // 3),
            min_lr=1e-7))

    print(f"\n=== Stage 1: frozen feature extraction "
          f"(<= {args.epochs_frozen} epochs, lr={args.frozen_lr:g}, patience={args.patience}) ===")
    history_frozen = model.fit(
        train_ds, validation_data=valid_ds, epochs=args.epochs_frozen, callbacks=callbacks,
    )

    finetune_info = unfreeze_last_blocks(base, args.finetune_blocks)
    print(f"\n=== Stage 2: fine-tuning last {args.finetune_blocks} base blocks "
          f"(<= {args.epochs_finetune} epochs, lr={args.finetune_lr:g} "
          f"[{args.lr_schedule}], patience={args.patience}) ===")
    print(f"Unfroze from block {finetune_info['first_unfrozen_block']} onward: "
          f"{finetune_info['layers_unfrozen']}/{finetune_info['total_layers']} layers, "
          f"{finetune_info['n_blocks_unfrozen']}/{finetune_info['total_blocks']} blocks")
    model.compile(
        optimizer=_finetune_optimizer(args, steps_per_epoch),
        loss=loss,
        metrics=["accuracy"],
    )
    history_finetune = model.fit(
        train_ds, validation_data=valid_ds, epochs=args.epochs_finetune, callbacks=callbacks,
    )

    n_frozen = len(history_frozen.history["loss"])
    n_finetune = len(history_finetune.history["loss"])
    schedule_info = {
        "recipe_version": RECIPE_VERSION,
        "epochs_frozen_cap": args.epochs_frozen,
        "epochs_finetune_cap": args.epochs_finetune,
        "frozen_lr": args.frozen_lr,
        "finetune_lr": args.finetune_lr,
        "lr_schedule": args.lr_schedule,
        "early_stopping_patience": args.patience,
        "steps_per_epoch": steps_per_epoch,
        # The v1 failure was that these were always False -- the cap stopped every run
        # before it converged. If they are False here too, raise the caps and re-run.
        "converged_frozen": n_frozen < args.epochs_frozen,
        "converged_finetune": n_finetune < args.epochs_finetune,
    }
    return history_frozen, history_finetune, finetune_info, schedule_info
