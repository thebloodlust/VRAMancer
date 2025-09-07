# core/tokenizer.py
"""
Wrapper léger autour de transformers.AutoTokenizer.

Il cache le tokenizer dans un cache mémoire afin
d’éviter de le re‑télécharger à chaque lancement.
"""

import transformers
from typing import Optional

# On garde un cache global dans le module
_tokenizer_cache = {}

def get_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> transformers.PreTrainedTokenizer:
    """
    Retourne un tokenizer à partir d’un nom de modèle (ex. 'gpt2').

    Le tokenizer est mis en cache dans `_tokenizer_cache`
    pour éviter de le re‑télécharger sur le disque.

    Parameters
    ----------
    model_name : str
        Nom du modèle HuggingFace (ex. 'gpt2', 'EleutherAI/gpt-neo-125M', etc.).
    cache_dir : str | None
        Répertoire où télécharger le tokenizer (par défaut, HuggingFace cache).
    **kwargs
        Autres arguments passés à `AutoTokenizer.from_pretrained`.

    Returns
    -------
    transformers.PreTrainedTokenizer
        Le tokenizer chargé.
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, **kwargs
    )
    _tokenizer_cache[model_name] = tokenizer
    return tokenizer
