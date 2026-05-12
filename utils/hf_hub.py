"""
HuggingFace Hub upload utilities.

Uploads trained model checkpoints (LoRA adapters or full models) to the Hub
after each training stage, with a structured repo name and model card.

The repo name is fully derived from the experiment parameters (model name,
stage, fine-tuning methods) plus the authenticated user's HF username.
No prefix needs to be configured manually.
"""

import os
from huggingface_hub import HfApi, create_repo, upload_folder


def _build_repo_name(model_name: str, stage: str, s1_method: str, s2_method: str | None = None) -> str:
    """
    Build the repo name (without owner) from experiment parameters.

    Examples:
      llama-3.1-8b-s1-lora-aramed
      llama-3.1-8b-s1-lora-s2-full-medarabench
      jais-2-8b-chat-s1-none-s2-lora-medarabench
    """
    short_name = model_name.split("/")[-1].lower().replace("_", "-")

    if stage == "stage1":
        return f"{short_name}-s1-{s1_method}-aramed"
    else:
        return f"{short_name}-s1-{s1_method or 'none'}-s2-{s2_method}-medarabench"


def upload_checkpoint_to_hub(
    checkpoint_dir: str,
    model_name: str,
    stage: str,
    s1_method: str,
    s2_method: str | None = None,
    hf_token: str | None = None,
    private: bool = False,
    extra_metadata: dict | None = None,
) -> str:
    """
    Upload a model checkpoint directory to HuggingFace Hub.

    The repo owner is resolved automatically from the HF token (whoami).
    The repo name is derived from the experiment parameters — no manual
    prefix configuration needed.

    Args:
        checkpoint_dir: local path to the saved checkpoint
        model_name:     base model HF ID (used to build the repo name)
        stage:          "stage1" or "stage2"
        s1_method:      Stage 1 method ("lora", "full", or "none")
        s2_method:      Stage 2 method ("lora", "full", or None for stage1 uploads)
        hf_token:       HF API token (falls back to HF_TOKEN env var)
        private:        whether the repo should be private
        extra_metadata: dict of extra info to embed in the model card

    Returns:
        The full repo ID, e.g. "your-hf-username/llama-3.1-8b-s1-lora-s2-full-medarabench"
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "No HuggingFace token found. Set HF_TOKEN in your .env file or environment."
        )

    api = HfApi(token=token)

    # Resolve the owner from the token — no manual prefix needed
    user_info = api.whoami()
    owner = user_info["name"]

    repo_name = _build_repo_name(model_name, stage, s1_method, s2_method)
    repo_id = f"{owner}/{repo_name}"

    print(f"\nUploading {stage} checkpoint to HF Hub: {repo_id}  (owner resolved from token)")

    # Create repo if it doesn't exist
    create_repo(repo_id, token=token, private=private, exist_ok=True)

    # Always overwrite the README — PEFT auto-writes one with a local path
    # in `base_model:` which HF rejects with a 400 BadRequest. Our card uses
    # the proper HF model ID.
    card_path = os.path.join(checkpoint_dir, "README.md")
    _write_model_card(card_path, repo_id, model_name, stage, s1_method, s2_method, extra_metadata)

    upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        token=token,
        commit_message=f"Upload {stage} checkpoint ({s1_method}/{s2_method or '-'})",
        ignore_patterns=["_merged_stage1/*", "*.tmp", "__pycache__/*"],
    )

    hub_url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded to: {hub_url}")
    return repo_id


def _write_model_card(
    path: str,
    repo_id: str,
    model_name: str,
    stage: str,
    s1_method: str,
    s2_method: str | None,
    extra_metadata: dict | None,
):
    stage_label = "Stage 1 — Domain Adaptation (AraMed)" if stage == "stage1" \
        else "Stage 2 — Task Fine-tuning (MedAraBench)"

    lines = [
        "---",
        "language:",
        "- ar",
        "license: llama3",
        f"base_model: {model_name}",
        "tags:",
        "- arabic",
        "- medical",
        "- question-answering",
        "- fine-tuned",
        f"- {s1_method}-lora" if "lora" in (s1_method or "") else f"- {s1_method}",
        "---",
        "",
        f"# {repo_id}",
        "",
        f"**Base model:** `{model_name}`  ",
        f"**Training stage:** {stage_label}  ",
        f"**Stage 1 method:** {s1_method or 'none'}  ",
        f"**Stage 2 method:** {s2_method or 'N/A'}  ",
        "",
        "## Paper",
        "",
        "_LoRA vs. Full Fine-Tuning for Arabic Medical Question Answering: "
        "A Systematic Comparison Across General-Purpose and Arabic-Centric Large Language Models_",
        "",
        "## Training data",
        "",
        "| Stage | Dataset | Samples |",
        "|-------|---------|---------|",
        "| Stage 1 | AraMed (open-ended Arabic medical QA) | ~110K |",
        "| Stage 2 | MedAraBench (Arabic MCQ) | ~17.6K (cleaned) |",
        "",
        "## Evaluation",
        "",
        "Evaluated on MedAraBench test set (4,959 MCQ samples) using log-probability selection.",
        "Metrics: **Accuracy** and **Macro F1** across answer classes A–E.",
        "",
    ]

    if extra_metadata:
        lines += ["## Experiment metadata", "", "```json"]
        import json
        lines.append(json.dumps(extra_metadata, indent=2, ensure_ascii=False))
        lines += ["```", ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
