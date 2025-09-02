from core.logger import Logger

class Scheduler:
    def __init__(self, gpu_list, strategy="balanced", verbose=True):
        self.gpu_list = gpu_list
        self.strategy = strategy
        self.allocations = {gpu["id"]: [] for gpu in gpu_list}
        self.logger = Logger(verbose=verbose)
        self.logger.info(f"Scheduler initialisé avec stratégie '{strategy}'")
