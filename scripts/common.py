"""
Shared helpers for the revision-R1 orchestration scripts.

Provides:
  - Project paths and constants (W&B entity/project, model IDs)
  - The revision run matrix (seeds, LoRA sweep, QLoRA, new models, Stage 1 diag)
  - Skip logic: query W&B for finished runs AND check local output dirs,
    so every script is resumable and never re-runs a finished configuration
  - Best-LoRA-configuration resolution from W&B results (with the documented
    fallback r=16 / alpha=32 / lr=2e-4 / dropout=0.05 all-linear)

All revision W&B runs are tagged `revision-r1`.
"""

import os
import sys
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.env_loader import load_env
load_env()

from utils.model_registry import get_spec, short_name_for  # noqa: E402

WANDB_ENTITY = "omaratef3221"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "arabic-medical-llm")
REVISION_TAG = "revision-r1"

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

LLAMA = "meta-llama/Llama-3.1-8B"
JAIS = "inceptionai/Jais-2-8B-Chat"
EXISTING_MODELS = [LLAMA, JAIS]

FANAR = "QCRI/Fanar-1-9B-Instruct"
SILMA = "silma-ai/SILMA-9B-Instruct-v1.0"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
NEW_MODELS = [FANAR, SILMA, QWEN]

# Fallback LoRA configuration (Table 2 conventions) used whenever the Task 3
# sweep has not finished for a model. Any consumer must surface the note.
FALLBACK_LORA = {"r": 16, "alpha": 32, "lr": 2e-4, "dropout": 0.05,
                 "target_modules": None}  # None -> all-linear default

# Original experiment output directories (run.sh exp01-exp14), keyed by run name.
LEGACY_OUTPUT_DIRS = {
    "llama-3.1-8b_s1-lora_s2-lora":     "exp01_llama_lora_lora",
    "llama-3.1-8b_s1-full_s2-full":     "exp02_llama_full_full",
    "llama-3.1-8b_s1-lora_s2-full":     "exp03_llama_lora_full",
    "llama-3.1-8b_s1-full_s2-lora":     "exp04_llama_full_lora",
    "jais-2-8b-chat_s1-lora_s2-lora":   "exp05_jais_lora_lora",
    "jais-2-8b-chat_s1-full_s2-full":   "exp06_jais_full_full",
    "jais-2-8b-chat_s1-lora_s2-full":   "exp07_jais_lora_full",
    "jais-2-8b-chat_s1-full_s2-lora":   "exp08_jais_full_lora",
    "llama-3.1-8b_s1-none_s2-lora":     "exp09_llama_none_lora",
    "llama-3.1-8b_s1-none_s2-full":     "exp10_llama_none_full",
    "jais-2-8b-chat_s1-none_s2-lora":   "exp11_jais_none_lora",
    "jais-2-8b-chat_s1-none_s2-full":   "exp12_jais_none_full",
    "llama-3.1-8b_s1-none_s2-none":     "exp13_llama_zeroshot",
    "jais-2-8b-chat_s1-none_s2-none":   "exp14_jais_zeroshot",
}


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """One row of the revision run matrix = one `python main.py ...` call."""
    run_name: str
    model: str                    # HF model ID
    stage1_method: str            # "none" | "lora" | "full"
    stage2_method: str            # "none" | "lora" | "full"
    phase: str                    # seeds | sweep | newmodels | stage1diag
    train: bool = True            # False -> evaluation only (zero-shot)
    extra_args: list = field(default_factory=list)
    note: str = ""                # surfaced in the dry-run plan

    @property
    def output_dir(self) -> str:
        legacy = LEGACY_OUTPUT_DIRS.get(self.run_name)
        if legacy:
            return os.path.join(OUTPUTS_DIR, legacy)
        return os.path.join(OUTPUTS_DIR, self.run_name)

    def main_py_args(self) -> list:
        args = [
            "--model", self.model,
            "--stage1_method", self.stage1_method,
            "--stage2_method", self.stage2_method,
            "--output_dir", self.output_dir,
            "--run_name", self.run_name,
            "--wandb_tags", REVISION_TAG,
            "--do_eval",
        ]
        if self.train:
            args.append("--do_train")
        return args + list(self.extra_args)


# ---------------------------------------------------------------------------
# W&B queries (cached per process; failure-tolerant so offline resume works)
# ---------------------------------------------------------------------------

_WANDB_CACHE: dict | None = None


def fetch_wandb_runs(force: bool = False) -> dict:
    """
    Return {run_name: {"state": ..., "accuracy": ..., "macro_f1": ...}} for
    every run in the W&B project. When a name appears more than once, a
    finished run with metrics wins over failed/crashed duplicates.
    """
    global _WANDB_CACHE
    if _WANDB_CACHE is not None and not force:
        return _WANDB_CACHE

    runs_by_name: dict = {}
    try:
        import wandb
        api = wandb.Api(timeout=60)
        for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"):
            info = {
                "state": run.state,
                "accuracy": run.summary.get("eval/accuracy"),
                "macro_f1": run.summary.get("eval/macro_f1"),
                "tags": list(run.tags or []),
            }
            prev = runs_by_name.get(run.name)
            if prev is None or (info["state"] == "finished" and prev["state"] != "finished"):
                runs_by_name[run.name] = info
            elif (info["state"] == "finished" and prev["state"] == "finished"
                  and prev["accuracy"] is None and info["accuracy"] is not None):
                runs_by_name[run.name] = info
    except Exception as e:  # network down, bad key, ... -> local checks only
        print(f"[common] WARNING: could not query W&B ({e}). "
              f"Falling back to local checkpoint checks only.")
    _WANDB_CACHE = runs_by_name
    return runs_by_name


def wandb_finished(run_name: str) -> bool:
    info = fetch_wandb_runs().get(run_name)
    return bool(info and info["state"] == "finished")


def wandb_accuracy(run_name: str) -> float | None:
    info = fetch_wandb_runs().get(run_name)
    if info and info["state"] == "finished":
        return info["accuracy"]
    return None


# ---------------------------------------------------------------------------
# Local existence checks
# ---------------------------------------------------------------------------

def predictions_path(run_name: str) -> str:
    return os.path.join(PREDICTIONS_DIR, f"{run_name}.parquet")


def has_predictions(run_name: str) -> bool:
    return os.path.isfile(predictions_path(run_name))


def local_eval_done(run_name: str) -> bool:
    """A local eval/results.json marks the configuration as completed."""
    candidates = [os.path.join(OUTPUTS_DIR, run_name)]
    legacy = LEGACY_OUTPUT_DIRS.get(run_name)
    if legacy:
        candidates.append(os.path.join(OUTPUTS_DIR, legacy))
    return any(os.path.isfile(os.path.join(c, "eval", "results.json"))
               for c in candidates)


def resolve_checkpoint_dir(run_name: str, stage: str = "stage2") -> str | None:
    """Locate the stage1/stage2 checkpoint directory for a run, if present."""
    candidates = [os.path.join(OUTPUTS_DIR, run_name)]
    legacy = LEGACY_OUTPUT_DIRS.get(run_name)
    if legacy:
        candidates.append(os.path.join(OUTPUTS_DIR, legacy))
    for c in candidates:
        stage_dir = os.path.join(c, stage)
        if os.path.isdir(stage_dir) and any(
            os.path.isfile(os.path.join(stage_dir, f))
            for f in ("config.json", "adapter_config.json",
                      "adapter_model.safetensors", "model.safetensors.index.json")
        ):
            return stage_dir
    return None


def run_completed(run_name: str) -> tuple[bool, str]:
    """
    NEVER re-run an existing configuration: a run counts as completed when a
    finished W&B run with the same name exists OR a local eval/results.json
    exists for its output directory.
    """
    if wandb_finished(run_name):
        return True, "W&B finished run"
    if local_eval_done(run_name):
        return True, "local eval/results.json"
    return False, ""


# ---------------------------------------------------------------------------
# Best-LoRA-configuration resolution (Task 3 -> consumed by QLoRA and Task 4)
# ---------------------------------------------------------------------------

def _fmt_lr(lr: float) -> str:
    return {1e-4: "1e-4", 2e-4: "2e-4", 4e-4: "4e-4"}.get(lr, f"{lr:g}")


def rank_run_name(model: str, r: int) -> str:
    base = f"{short_name_for(model)}_s1-none_s2-lora"
    return base if r == 16 else f"{base}_r{r}"


def lr_run_name(model: str, lr: float) -> str:
    return f"{short_name_for(model)}_s1-none_s2-lora_lr{_fmt_lr(lr)}"


def dropout_run_name(model: str, do: float) -> str:
    return f"{short_name_for(model)}_s1-none_s2-lora_do{do}"


def resolve_best_rank(model: str) -> tuple[int, bool, str]:
    """
    Pick the best LoRA rank for a model from finished W&B rank-sweep runs
    (r in {8, 16, 32, 64}; the r=16 run is the pre-existing baseline).

    Returns (rank, resolved, note). resolved=False -> fallback r=16 was used
    because at least one rank run has not finished yet.
    """
    accs = {r: wandb_accuracy(rank_run_name(model, r)) for r in (8, 16, 32, 64)}
    missing = [r for r, a in accs.items() if a is None]
    if missing:
        return 16, False, (f"rank sweep incomplete (missing r={missing}); "
                           f"falling back to r=16")
    best = max(accs, key=accs.get)
    return best, True, f"best rank r={best} (acc={accs[best]:.4f})"


def resolve_best_lora_config(model: str) -> dict:
    """
    Resolve the best LoRA configuration for a model from the Task 3 sweep:
    rank first, then LR and dropout (one factor at a time, vs. the best-rank
    baseline). Falls back to Table 2 defaults wherever runs are missing.

    Returns {"r", "alpha", "lr", "dropout", "resolved", "notes"}.
    """
    notes = []
    r, rank_resolved, rank_note = resolve_best_rank(model)
    notes.append(rank_note)

    baseline_acc = wandb_accuracy(rank_run_name(model, r))

    lr, lr_resolved = 2e-4, True
    lr_accs = {2e-4: baseline_acc}
    for cand in (1e-4, 4e-4):
        lr_accs[cand] = wandb_accuracy(lr_run_name(model, cand))
    if any(a is None for a in lr_accs.values()):
        lr_resolved = False
        notes.append("LR sweep incomplete; falling back to lr=2e-4")
    else:
        lr = max(lr_accs, key=lr_accs.get)
        notes.append(f"best lr={_fmt_lr(lr)}")

    dropout, do_resolved = 0.05, True
    do_accs = {0.05: baseline_acc}
    for cand in (0.0, 0.1):
        do_accs[cand] = wandb_accuracy(dropout_run_name(model, cand))
    if any(a is None for a in do_accs.values()):
        do_resolved = False
        notes.append("dropout sweep incomplete; falling back to dropout=0.05")
    else:
        dropout = max(do_accs, key=do_accs.get)
        notes.append(f"best dropout={dropout}")

    resolved = rank_resolved and lr_resolved and do_resolved
    if not resolved:
        notes.append("NOTE: using fallback Table 2 values for unresolved factors")
    return {
        "r": r, "alpha": 2 * r, "lr": lr, "dropout": dropout,
        "resolved": resolved, "notes": "; ".join(notes),
    }


def lora_override_args(cfg: dict) -> list:
    """Translate a resolved LoRA config into main.py CLI overrides."""
    args = ["--lora_r", str(cfg["r"]), "--lora_alpha", str(cfg["alpha"]),
            "--learning_rate", str(cfg["lr"]), "--lora_dropout", str(cfg["dropout"])]
    if cfg.get("target_modules"):
        args += ["--lora_target_modules", cfg["target_modules"]]
    return args


# ---------------------------------------------------------------------------
# Revision run matrix builders
# ---------------------------------------------------------------------------

def seeds_matrix() -> list:
    """Task 2: seeds 1337 and 2024 for the four Stage-2-only configurations."""
    specs = []
    for model in EXISTING_MODELS:
        short = short_name_for(model)
        for method in ("lora", "full"):
            for seed in (1337, 2024):
                specs.append(RunSpec(
                    run_name=f"{short}_s1-none_s2-{method}_seed{seed}",
                    model=model, stage1_method="none", stage2_method=method,
                    phase="seeds",
                    extra_args=["--seed", str(seed)],
                    note=f"multi-seed rerun (seed {seed})",
                ))
    return specs


def sweep_rank_matrix() -> list:
    """Task 3.1: LoRA rank sweep r in {8, 32, 64}, alpha = 2r (r=16 exists)."""
    specs = []
    for model in EXISTING_MODELS:
        for r in (8, 32, 64):
            specs.append(RunSpec(
                run_name=rank_run_name(model, r),
                model=model, stage1_method="none", stage2_method="lora",
                phase="sweep",
                extra_args=["--lora_r", str(r), "--lora_alpha", str(2 * r)],
                note=f"rank sweep r={r}, alpha={2 * r}",
            ))
    return specs


def sweep_lr_dropout_matrix() -> list:
    """
    Task 3.2: at the best rank per model, LR in {1e-4, 4e-4} and dropout in
    {0.0, 0.1}, one factor at a time. Best rank is re-resolved from W&B at
    call time (falls back to r=16 with a note when the rank sweep is missing).
    """
    specs = []
    for model in EXISTING_MODELS:
        r, resolved, rank_note = resolve_best_rank(model)
        alpha = 2 * r
        base_extra = ["--lora_r", str(r), "--lora_alpha", str(alpha)]
        for lr in (1e-4, 4e-4):
            specs.append(RunSpec(
                run_name=lr_run_name(model, lr),
                model=model, stage1_method="none", stage2_method="lora",
                phase="sweep",
                extra_args=base_extra + ["--learning_rate", str(lr)],
                note=f"LR ablation at {rank_note}",
            ))
        for do in (0.0, 0.1):
            specs.append(RunSpec(
                run_name=dropout_run_name(model, do),
                model=model, stage1_method="none", stage2_method="lora",
                phase="sweep",
                extra_args=base_extra + ["--lora_dropout", str(do)],
                note=f"dropout ablation at {rank_note}",
            ))
    return specs


def sweep_attnonly_matrix() -> list:
    """Task 3.3: attention-only target-module ablation at the best rank."""
    specs = []
    for model in EXISTING_MODELS:
        r, resolved, rank_note = resolve_best_rank(model)
        specs.append(RunSpec(
            run_name=f"{short_name_for(model)}_s1-none_s2-lora_attnonly",
            model=model, stage1_method="none", stage2_method="lora",
            phase="sweep",
            extra_args=["--lora_r", str(r), "--lora_alpha", str(2 * r),
                        "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj"],
            note=f"attention-only ablation at {rank_note}",
        ))
    return specs


def qlora_matrix() -> list:
    """Task 3.4: QLoRA (NF4 + double quant + paged AdamW) at best LoRA config."""
    specs = []
    for model in EXISTING_MODELS:
        cfg = resolve_best_lora_config(model)
        specs.append(RunSpec(
            run_name=f"{short_name_for(model)}_s1-none_s2-qlora",
            model=model, stage1_method="none", stage2_method="lora",
            phase="sweep",
            extra_args=["--qlora"] + lora_override_args(cfg),
            note=f"QLoRA; {cfg['notes']}",
        ))
    return specs


def newmodels_matrix() -> list:
    """Task 4: zero-shot, Stage-2 LoRA (best sweep config), Stage-2 Full."""
    specs = []
    for model in NEW_MODELS:
        short = short_name_for(model)
        specs.append(RunSpec(
            run_name=f"{short}_s1-none_s2-none",
            model=model, stage1_method="none", stage2_method="none",
            phase="newmodels", train=False,
            note="zero-shot baseline",
        ))
        # Best LoRA config comes from the Task 3 sweep on the EXISTING models.
        # Use the sweep winner of the model family average; when unresolved,
        # fall back to Table 2 defaults and say so.
        cfg = best_sweep_config_for_new_models()
        specs.append(RunSpec(
            run_name=f"{short}_s1-none_s2-lora",
            model=model, stage1_method="none", stage2_method="lora",
            phase="newmodels",
            extra_args=lora_override_args(cfg),
            note=f"Stage-2 LoRA; {cfg['notes']}",
        ))
        specs.append(RunSpec(
            run_name=f"{short}_s1-none_s2-full",
            model=model, stage1_method="none", stage2_method="full",
            phase="newmodels",
            note="Stage-2 full fine-tuning (Table 2 conventions)",
        ))
    return specs


def best_sweep_config_for_new_models() -> dict:
    """
    Choose the LoRA configuration to transfer to the new base models: the
    best resolved sweep configuration, preferring the model whose best-rank
    run scored higher. Unresolved -> documented Table 2 fallback.
    """
    candidates = []
    for model in EXISTING_MODELS:
        cfg = resolve_best_lora_config(model)
        if cfg["resolved"]:
            acc = wandb_accuracy(rank_run_name(model, cfg["r"]))
            candidates.append((acc or 0.0, model, cfg))
    if candidates:
        acc, model, cfg = max(candidates, key=lambda t: t[0])
        cfg = dict(cfg)
        cfg["notes"] = (f"transferred from {short_name_for(model)} sweep "
                        f"({cfg['notes']})")
        return cfg
    return {
        **FALLBACK_LORA,
        "resolved": False,
        "notes": ("Task 3 sweep not finished -> fallback "
                  "r=16/alpha=32/lr=2e-4/dropout=0.05 all-linear (noted)"),
    }


def stage1diag_matrix() -> list:
    """Task 5.2: Jais Stage 1 Full for 3 epochs, then Stage 2 Full."""
    return [RunSpec(
        run_name=f"{short_name_for(JAIS)}_s1-full3ep_s2-full",
        model=JAIS, stage1_method="full", stage2_method="full",
        phase="stage1diag",
        extra_args=["--stage1_epochs", "3"],
        note="multi-epoch Stage 1 ablation (3 epochs vs existing 1)",
    )]


def existing_conditions_matrix() -> list:
    """
    The 14 pre-revision conditions (7 per existing model, incl. zero-shot),
    used by scripts/reinfer_all.py for inference-only prediction dumps.
    """
    specs = []
    for model in EXISTING_MODELS:
        short = short_name_for(model)
        for s1, s2 in [("none", "none"),
                       ("none", "lora"), ("none", "full"),
                       ("lora", "lora"), ("lora", "full"),
                       ("full", "lora"), ("full", "full")]:
            specs.append(RunSpec(
                run_name=f"{short}_s1-{s1}_s2-{s2}",
                model=model, stage1_method=s1, stage2_method=s2,
                phase="reinfer", train=False,
                note="existing condition (inference-only re-eval)",
            ))
    return specs
