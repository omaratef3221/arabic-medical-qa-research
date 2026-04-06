"""
.env file loader for credentials.

Reads key=value pairs from a .env file and injects them into os.environ
without overwriting variables that are already set (env > .env file).

Supports:
  - Comments (#)
  - Quoted values ("value" or 'value')
  - Inline comments (KEY=value  # comment)
  - Empty lines

Credentials loaded (set these in your .env file):
  WANDB_API_KEY    — Weights & Biases API key
  WANDB_PROJECT    — W&B project name (default: arabic-medical-llm)
  HF_TOKEN         — HuggingFace Hub token (for model uploads and gated models)
  HF_REPO_PREFIX   — HF username/org for uploaded repos, e.g. "your-username"
"""

import os
import re


def load_env(env_path: str | None = None) -> dict[str, str]:
    """
    Load environment variables from a .env file.

    Search order for the .env file:
      1. ``env_path`` argument (if provided)
      2. .env in the same directory as this file (utils/.env)  ← unlikely
      3. .env in the project script root (script/.env)         ← standard
      4. .env two levels up from this file                     ← fallback

    Variables already present in ``os.environ`` are never overwritten.

    Args:
        env_path: explicit path to a .env file (optional)

    Returns:
        dict of key→value pairs that were newly added to the environment
    """
    candidates = []
    if env_path:
        candidates.append(env_path)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(this_dir)       # utils/ → script/
    project_dir = os.path.dirname(script_dir)    # script/ → project root

    candidates += [
        os.path.join(script_dir, ".env"),
        os.path.join(project_dir, ".env"),
        os.path.join(this_dir, ".env"),
    ]

    dotenv_path = None
    for candidate in candidates:
        if os.path.isfile(candidate):
            dotenv_path = candidate
            break

    if dotenv_path is None:
        return {}

    loaded: dict[str, str] = {}

    with open(dotenv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Must contain '='
            if "=" not in line:
                continue

            key, _, raw_value = line.partition("=")
            key = key.strip()
            if not key:
                continue

            # Strip inline comments (only outside quotes)
            raw_value = raw_value.strip()
            raw_value = _strip_inline_comment(raw_value)

            # Strip surrounding quotes
            value = _unquote(raw_value)

            # Only set if not already in environment
            if key not in os.environ:
                os.environ[key] = value
                loaded[key] = value

    if loaded:
        print(f"Loaded {len(loaded)} credential(s) from: {dotenv_path}")

    return loaded


def _strip_inline_comment(s: str) -> str:
    """
    Remove trailing # comment from a value.

    For quoted values ("foo" # comment) the comment is outside the closing
    quote, so we strip after the closing quote.
    For unquoted values (foo # comment) we strip after the first ' #'.
    """
    if not s:
        return s

    if s[0] in ('"', "'"):
        quote_char = s[0]
        # Find the matching closing quote
        close = s.find(quote_char, 1)
        if close != -1:
            # Everything after the closing quote is discarded (comments etc.)
            return s[: close + 1]
        return s

    # Unquoted: strip ' #' and everything after
    match = re.search(r"\s+#.*$", s)
    if match:
        return s[: match.start()]
    return s


def _unquote(s: str) -> str:
    """Remove surrounding single or double quotes from a value."""
    if len(s) >= 2:
        if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            return s[1:-1]
    return s
