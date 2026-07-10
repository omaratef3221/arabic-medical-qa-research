import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

from utils.model_registry import get_spec, configure_tokenizer as _registry_configure_tokenizer


def _build_lora_config(lora_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        task_type=TaskType.CAUSAL_LM,
        bias=lora_cfg.get("bias", "none"),
    )


def _configure_tokenizer(tokenizer, model_name: str | None = None):
    """Ensure tokenizer has a pad token and right-padding for training."""
    spec = get_spec(model_name or tokenizer.name_or_path)
    return _registry_configure_tokenizer(tokenizer, spec)


def _nf4_config() -> BitsAndBytesConfig:
    """QLoRA quantization: NF4 4-bit with double quantization (Dettmers 2023)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _from_pretrained_kwargs(model_name: str, load_in_4bit: bool) -> dict:
    """Assemble AutoModelForCausalLM.from_pretrained kwargs from the registry."""
    spec = get_spec(model_name)
    kwargs = {
        "device_map": "auto",
        "trust_remote_code": spec.trust_remote_code,
    }
    if spec.attn_implementation:
        # e.g. Gemma-2 (SILMA): logit soft-capping is skipped by flash/sdpa
        kwargs["attn_implementation"] = spec.attn_implementation
    if load_in_4bit:
        kwargs["quantization_config"] = _nf4_config()
    else:
        kwargs["dtype"] = torch.bfloat16
    return kwargs


def count_trainable_parameters(model) -> tuple[int, int, float]:
    """
    Return (trainable, total, percent) with correct handling of 4-bit
    quantized weights, whose Params4bit tensors report packed (halved)
    element counts. PEFT's counter compensates for this; fall back to a
    manual count that applies the same correction.
    """
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable, total = 0, 0
        for p in model.parameters():
            numel = p.numel()
            if p.__class__.__name__ == "Params4bit":
                numel = numel * 2  # packed 4-bit: 2 weights per int8 element
            total += numel
            if p.requires_grad:
                trainable += numel
    pct = 100.0 * trainable / max(total, 1)
    return trainable, total, pct


def load_model_and_tokenizer(
    model_name: str,
    method: str = "lora",
    lora_config: dict | None = None,
    load_in_4bit: bool = False,
):
    """
    Load a base causal LM and its tokenizer.

    Args:
        model_name:   HuggingFace model identifier, e.g.
                      "meta-llama/Llama-3.1-8B" or "inceptionai/Jais-2-8B-Chat"
        method:       "lora"  → apply PEFT LoRA adapter after loading
                      "full"  → load base model with no adapter
        lora_config:  dict of LoRA hyperparameters (from lora.yaml['lora']).
                      Only used when method="lora".
        load_in_4bit: Use BitsAndBytesConfig for QLoRA (required for 70B models).

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=get_spec(model_name).trust_remote_code
    )
    tokenizer = _configure_tokenizer(tokenizer, model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, **_from_pretrained_kwargs(model_name, load_in_4bit)
    )

    if method == "lora":
        if load_in_4bit:
            # QLoRA: cast norms/embeddings, enable input grads for checkpointing
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        cfg = lora_config or {}
        peft_cfg = _build_lora_config(cfg)
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
    elif method == "full":
        # Ensure all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"method must be 'lora' or 'full', got {method!r}")

    return model, tokenizer


def load_from_checkpoint(
    checkpoint_path: str,
    base_model_name: str,
    method: str = "lora",
    load_in_4bit: bool = False,
):
    """
    Load a model from a saved checkpoint.

    For LoRA checkpoints: loads base model + merges/attaches adapter.
    For full FT checkpoints: loads model directly from checkpoint path.

    Args:
        checkpoint_path: path to saved checkpoint directory
        base_model_name: HuggingFace model name for the base model
                         (needed for LoRA checkpoints)
        method:          "lora" or "full"
        load_in_4bit:    enable QLoRA quantization

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    tokenizer = _configure_tokenizer(tokenizer, base_model_name)

    if method == "lora":
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, **_from_pretrained_kwargs(base_model_name, load_in_4bit)
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
    elif method == "full":
        # Full checkpoints inherit the base model's loading quirks
        # (e.g. eager attention for Gemma-2) via the registry lookup.
        kwargs = _from_pretrained_kwargs(base_model_name, load_in_4bit=False)
        kwargs["trust_remote_code"] = True  # checkpoint may carry remote code
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **kwargs)
    else:
        raise ValueError(f"method must be 'lora' or 'full', got {method!r}")

    return model, tokenizer


def merge_lora_and_save(model, save_path: str, tokenizer=None):
    """
    Merge a LoRA adapter into the base model weights and save to disk.
    Used for mixed-method experiments (LoRA Stage 1 → Full Stage 2).

    Args:
        model:     PeftModel with LoRA adapter
        save_path: directory to save the merged model
        tokenizer: if provided, also save the tokenizer
    """
    merged = model.merge_and_unload()
    merged.save_pretrained(save_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    print(f"Merged model saved to: {save_path}")
    return merged
