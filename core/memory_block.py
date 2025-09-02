import uuid

class MemoryBlock:
    def __init__(self, size_mb, gpu_id, status="free"):
        self.id = str(uuid.uuid4())
        self.size_mb = size_mb
        self.gpu_id = gpu_id
        self.status = status  # "free", "reserved", "allocated"

    def reserve(self):
        if self.status == "free":
            self.status = "reserved"
        else:
            raise RuntimeError(f"Block {self.id} is not free.")

    def allocate(self):
        if self.status == "reserved":
            self.status = "allocated"
        else:
            raise RuntimeError(f"Block {self.id} must be reserved before allocation.")

    def release(self):
        self.status = "free"

    def __repr__(self):
        return f"<Block {self.id[:8]} | {self.size_mb}MB | GPU {self.gpu_id} | {self.status}>"
