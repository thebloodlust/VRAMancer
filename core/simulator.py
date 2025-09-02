from core.gpu_interface import get_available_gpus
from core.scheduler import Scheduler
from core.logger import Logger

def simulate_model_loading(model_name="FakeLLM", total_size_mb=4096, block_size_mb=512):
    logger = Logger(verbose=True)
    logger.info(f"Simulation du chargement du modèle '{model_name}' ({total_size_mb}MB)")

    gpus = get_available_gpus()
    if not gpus:
        logger.error("Aucun GPU détecté.")
        return

    scheduler = Scheduler(gpu_list=gpus, strategy="balanced", verbose=True)

    blocks_needed = total_size_mb // block_size_mb
    logger.info(f"Découpage en {blocks_needed} blocs de {block_size_mb}MB")

    for i in range(blocks_needed):
        try:
            block = scheduler.allocate_block(block_size_mb)
            logger.debug(f"Bloc {i+1}/{blocks_needed} alloué : {block}")
        except Exception as e:
            logger.error(f"Échec d’allocation du bloc {i+1}: {e}")
            break

    scheduler.show_allocations()
    logger.info("✅ Simulation terminée.")

if __name__ == "__main__":
    simulate_model_loading()
