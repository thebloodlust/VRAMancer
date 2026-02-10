# core/tokenizer.py
"""
Wrapper leger autour de transformers.AutoTokenizer.

Cache le tokenizer en memoire afin d'eviter de le re-telecharger.
"""
from __future__ import annotations
from typing import Optional, Any

try:
    import transformers  # type: ignore
    _HAS_TRANSFORMERS = True
except ImportError:
    transformers = None  # type: ignore
    _HAS_TRANSFORMERS = False

_tokenizer_cache: dict[str, Any] = {}

def get_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Retourne un tokenizer a partir d'un nom de modele (ex. 'gpt2').

    Raises ImportError si transformers n'est pas installe.
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers requis : pip install transformers")

    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, **kwargs
    )
    _tokenizer_cache[model_name] = tokenizer
    return tokenizer
