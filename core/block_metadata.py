# core/block_metadata.py

def get_block_metadata(index: int) -> dict:
    metadata = {
        0: {"estimated_size_mb": 800, "importance": "critical"},
        1: {"estimated_size_mb": 300, "importance": "normal"},
        2: {"estimated_size_mb": 1200, "importance": "low"},
        3: {"estimated_size_mb": 500, "importance": "normal"},
    }
    return metadata.get(index, {"estimated_size_mb": 500, "importance": "normal"})
