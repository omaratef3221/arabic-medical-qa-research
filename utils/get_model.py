import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)


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


def _configure_tokenizer(tokenizer):
    """Ensure tokenizer has a pad token and right-padding for training."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = _configure_tokenizer(tokenizer)

    # Quantization config for QLoRA
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

    if method == "lora":
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
    tokenizer = _configure_tokenizer(tokenizer)

    if method == "lora":
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        model = PeftModel.from_pretrained(base, checkpoint_path)
    elif method == "full":
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
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
