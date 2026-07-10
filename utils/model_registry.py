"""
Model registry: one place for every base model's loading quirks.

Each entry records the HuggingFace model ID, tokenizer/pad-token handling,
trust_remote_code, chat-vs-base flavour, attention implementation, and how
the answer-letter tokens (A-E) are resolved for log-probability evaluation.

Adding a new base model = adding one ModelSpec here. Everything else
(training, evaluation, orchestration) works off the registry.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str                      # HuggingFace model identifier
    short_name: str                 # used in run names: {short_name}_s1-..._s2-...
    is_chat: bool                   # chat/instruct model vs plain base LM
    trust_remote_code: bool = False
    attn_implementation: str | None = None   # None → transformers default (sdpa)
    pad_token_strategy: str = "eos"          # "eos" → pad=eos; "native" → keep tokenizer's own pad token
    # Letter variants to try, in order, when resolving answer-token IDs.
    # Each must encode to exactly ONE token for the resolution to be accepted.
    answer_letter_variants: tuple = ("{letter}", " {letter}")
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec):
    REGISTRY[spec.hf_id] = spec
    return spec


# ---------------------------------------------------------------------------
# Existing models
# ---------------------------------------------------------------------------

_register(ModelSpec(
    hf_id="meta-llama/Llama-3.1-8B",
    short_name="llama-3.1-8b",
    is_chat=False,
    trust_remote_code=False,
))

_register(ModelSpec(
    hf_id="inceptionai/Jais-2-8B-Chat",
    short_name="jais-2-8b-chat",
    is_chat=True,
    trust_remote_code=True,
))

# ---------------------------------------------------------------------------
# Revision R1: new base models
# ---------------------------------------------------------------------------

_register(ModelSpec(
    hf_id="QCRI/Fanar-1-9B-Instruct",
    short_name="fanar-1-9b-instruct",
    is_chat=True,
    trust_remote_code=False,
    attn_implementation="eager",
    pad_token_strategy="native",
    notes=(
        "Arabic-centric, Gemma2ForCausalLM (verified from config.json: "
        "sliding_window=4096, logit soft-capping 30/50) - same constraints "
        "as SILMA: eager attention, native <pad> token."
    ),
))

_register(ModelSpec(
    hf_id="silma-ai/SILMA-9B-Instruct-v1.0",
    short_name="silma-9b-instruct-v1.0",
    is_chat=True,
    trust_remote_code=False,
    attn_implementation="eager",
    pad_token_strategy="native",
    notes=(
        "Gemma-2 architecture: sliding-window attention + logit soft-capping. "
        "Soft-capping is silently skipped by flash-attn/sdpa kernels, so eager "
        "attention is forced. Tokenizer ships its own <pad> token - keep it. "
        "Full FT shards across all 4 GPUs via device_map=auto like the 8B models."
    ),
))

_register(ModelSpec(
    hf_id="Qwen/Qwen2.5-7B-Instruct",
    short_name="qwen2.5-7b-instruct",
    is_chat=True,
    trust_remote_code=False,
    notes="General-purpose multilingual. eos=<|im_end|>; pad falls back to eos.",
))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_spec(model_name: str) -> ModelSpec:
    """
    Return the ModelSpec for a HF model ID. Unknown models get a permissive
    default spec (trust_remote_code=True, pad=eos) so legacy paths keep working.
    """
    if model_name in REGISTRY:
        return REGISTRY[model_name]
    # Case-insensitive match on the repo name
    for hf_id, spec in REGISTRY.items():
        if hf_id.lower() == model_name.lower():
            return spec
    return ModelSpec(
        hf_id=model_name,
        short_name=model_name.split("/")[-1].lower(),
        is_chat=False,
        trust_remote_code=True,
        notes="Unregistered model - permissive defaults.",
    )


def short_name_for(model_name: str) -> str:
    return get_spec(model_name).short_name


def configure_tokenizer(tokenizer, spec: ModelSpec):
    """Apply the registry's pad-token strategy. Right-padding for training."""
    if spec.pad_token_strategy == "native" and tokenizer.pad_token is not None:
        pass  # tokenizer ships a real pad token (e.g. Gemma-2 <pad>)
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def resolve_answer_token_ids(tokenizer, spec: ModelSpec | None = None) -> dict[str, int]:
    """
    Resolve the token ID for each answer letter A-E and VERIFY that each
    resolves to a single token. Logs the resolved IDs (and to the active
    W&B run's config if one exists).

    Raises ValueError if any letter cannot be encoded as a single token
    under any registered variant - that would silently corrupt the
    log-probability evaluation.
    """
    variants = spec.answer_letter_variants if spec else ("{letter}", " {letter}")
    letters = ["A", "B", "C", "D", "E"]
    token_ids: dict[str, int] = {}
    resolved_variant: dict[str, str] = {}

    for letter in letters:
        for template in variants:
            text = template.format(letter=letter)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                token_ids[letter] = ids[0]
                resolved_variant[letter] = repr(text)
                break
        else:
            tried = {template.format(letter=letter):
                     tokenizer.encode(template.format(letter=letter), add_special_tokens=False)
                     for template in variants}
            raise ValueError(
                f"Answer letter {letter!r} does not encode to a single token for "
                f"tokenizer {tokenizer.name_or_path!r}. Tried: {tried}. "
                f"Register a custom answer_letter_variants for this model."
            )

    # All five letters must map to distinct token IDs
    if len(set(token_ids.values())) != len(letters):
        raise ValueError(
            f"Answer token IDs are not distinct for {tokenizer.name_or_path!r}: {token_ids}"
        )

    print(f"Answer token IDs ({tokenizer.name_or_path}):")
    for letter in letters:
        tid = token_ids[letter]
        print(f"  {letter} -> id={tid}  variant={resolved_variant[letter]}  "
              f"decoded={tokenizer.decode([tid])!r}")

    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update(
                {f"answer_token_id_{l}": token_ids[l] for l in letters},
                allow_val_change=True,
            )
    except ImportError:
        pass

    return token_ids
