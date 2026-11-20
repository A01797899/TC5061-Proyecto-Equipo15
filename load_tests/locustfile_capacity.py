"""Prueba de capacidad por escalones sobre POST /predict.

Uso:
    locust -f load_tests/locustfile_capacity.py --host http://127.0.0.1:8000 --headless
"""

from locust import LoadTestShape

from locustfile import MiBiciUser  # noqa: F401

STEP_USERS = [50, 100, 150, 200, 250, 300]
STEP_SECONDS = 60
SPAWN_RATE = 10


class StepLoadShape(LoadTestShape):
    """Aumenta los usuarios en escalones fijos y termina después del último."""

    def tick(self):
        run_time = self.get_run_time()
        step = int(run_time // STEP_SECONDS)
        if step >= len(STEP_USERS):
            return None
        return STEP_USERS[step], SPAWN_RATE
