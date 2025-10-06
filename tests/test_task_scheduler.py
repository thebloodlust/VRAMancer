from core.task_scheduler import OpportunisticScheduler
import time

def test_scheduler_basic():
    sched = OpportunisticScheduler(poll_interval=0.1, max_threads=2)
    counter = {"v":0}
    def job():
        counter["v"] += 1
        time.sleep(0.05)
    for _ in range(5):
        sched.submit(job)
    time.sleep(1.0)
    assert counter["v"] >= 2  # au moins quelques jobs ont tourné
    sched.stop()
