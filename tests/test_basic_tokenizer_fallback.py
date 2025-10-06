import os

def test_basic_tokenizer_forced():
    os.environ['VRM_FORCE_BASIC_TOKENIZER'] = '1'
    from core.utils import get_tokenizer
    tok = get_tokenizer('dummy-model')
    ids = tok.encode("Hello World")  # type: ignore
    assert isinstance(ids, list) and len(ids) == 2
    dec = tok.decode(ids)  # type: ignore
    assert 'hello' in dec
    # cleanup
    os.environ.pop('VRM_FORCE_BASIC_TOKENIZER')
