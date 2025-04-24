from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import Estimator, EstimatorOptions, Session, QiskitRuntimeService
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
import numpy as np
import logging
import sys
from datetime import datetime
from config import *
import os

from utils.data import load_mnist_0_vs_1_resized
from utils.circuit import create_ansatz

date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
log_filename = f"log/hardware_{date_str}.txt"

logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
sys.stdout = open(log_filename, "a")
sys.stderr = sys.stdout

logging.getLogger('qiskit').setLevel(logging.ERROR)

API_TOKEN = os.getenv("API_TOKEN")
service = QiskitRuntimeService(channel='ibm_quantum', token=API_TOKEN, instance='usc/phys513/phys513p')
backend = service.least_busy()
session = Session(backend=backend)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist_0_vs_1_resized(size=8)
    ansatz, params = create_ansatz(num_qubits=NUM_QUBITS, num_layers=NUM_LAYERS)

    with Session(backend=backend) as session:
        options = EstimatorOptions(resilience_level=1)
        estimator = Estimator(mode=session) 

        indices = np.random.choice(len(X_train), NUM_SAMPLES, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        def compute_cost(params_values, x, y):
            try:
                target = 1 if y == 0 else -1

                # Assign parameters
                assigned_ansatz = ansatz.assign_parameters(dict(zip(params, params_values)))

                # Amplitude encoding
                x = np.asarray(x, dtype=np.complex128).flatten()
                x = x / np.linalg.norm(x)
                stateprep = StatePreparation(x)
                prep_circuit = QuantumCircuit(NUM_QUBITS)
                prep_circuit.append(stateprep, range(NUM_QUBITS))
                prep_circuit = prep_circuit.decompose(reps=10)

                # Combine and transpile
                full_circuit = prep_circuit.compose(assigned_ansatz)            
                full_circuit = transpile(full_circuit, backend=backend, optimization_level=1)

                # Observable and layout mapping
                observable = SparsePauliOp.from_list([("ZZZZZZ", 1.0)]) 
                observable_isa = observable.apply_layout(layout=full_circuit.layout)

                result = estimator.run([(full_circuit, observable_isa)]).result()
                output = result[0].data.evs.item()

                cost = (output - target) ** 2
                logging.info(f"Sample cost: {cost:.4f} (target={target}, output={output:.4f})")
                return cost

            except Exception as e:
                logging.error(f"Compute cost error: {e}")
                return float('inf')

        def total_cost(params_values):
            costs = [compute_cost(params_values, x, y) for x, y in zip(X_batch, y_batch)]
            return np.mean(costs) if costs else float('inf')

        # Initialize and train
        params_init = np.random.uniform(0, 2 * np.pi, len(params))
        result = minimize(total_cost, params_init, method="COBYLA", options={"maxiter": MAX_ITER, "disp": True})
        trained_params = result.x
        
        print(f"Final cost: {result.fun:.4f}")
        print(f"Optimizer status: {result.message}")
        np.save(f"trained_params/hardware_{date_str}.npy", result.x)