from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator, EstimatorOptions, Session
from scipy.optimize import minimize
import numpy as np
import logging
import sys
from datetime import datetime
from config import *

from utils.data import load_mnist_0_vs_1_resized
from utils.circuit import create_ansatz
from utils.optimize import compute_cost

date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
log_filename = f"log/simulator_{date_str}.txt"

logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
sys.stdout = open(log_filename, "a")
sys.stderr = sys.stdout

logging.getLogger('qiskit').setLevel(logging.ERROR)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist_0_vs_1_resized(size=8)
    ansatz, params = create_ansatz(num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS)
    backend = AerSimulator()

    with Session(backend=backend) as session:
        options = EstimatorOptions(resilience_level=1)
        estimator = Estimator(mode=session) 

        indices = np.random.choice(len(X_train), NUM_SAMPLES, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        def total_cost(params_values: np.ndarray) -> float:
            costs = []
            for x, y in zip(X_batch, y_batch):
                try:
                    cost = compute_cost(params_values, x, y, ansatz, params, estimator, backend)
                    costs.append(cost)
                except Exception as e:
                    print(f"[!] Cost calculation error: {e}")
                    continue
            return np.mean(costs) if costs else float('inf')

        params_init = np.random.uniform(0, 2 * np.pi, len(params))

        result = minimize(total_cost, params_init, method="COBYLA", 
                        options={
                            "maxiter": MAX_ITER,
                            "disp": True,
                            "rhobeg": np.pi/4 # TODO: DOCUMENT THIS AND WHY
                        })
        
        print(f"Final cost: {result.fun:.4f}")
        print(f"Optimizer status: {result.message}")
        np.save(f"trained_params/simulator_{date_str}.npy", result.x)