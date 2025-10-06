from core.task_scheduler import OpportunisticScheduler
import time

def test_batch_and_cancel():
    sched = OpportunisticScheduler(poll_interval=0.05, max_threads=1)
    ran = {"a":0, "b":0}
    def job_a():
        ran["a"] += 1; time.sleep(0.1)
    def job_b():
        ran["b"] += 1; time.sleep(0.2)
    ids = sched.submit_batch([
        {"fn": job_a, "priority":"HIGH", "tags":["test"], "est_runtime_s":0.1},
        {"fn": job_b, "priority":"NORMAL", "tags":["test"], "est_runtime_s":0.2},
    ])
    assert len(ids)==2
    # Cancel second quickly
    sched.cancel(ids[1])
    time.sleep(0.6)
    # a devrait avoir tourné >=1, b peut être annulé
    assert ran["a"] >= 1
    # b peut être 0 si annulé avant exécution
    sched.stop()
