
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Estimator, Session
from typing import List

from .circuit import build_full_circuit

def compute_cost(
    params_values: np.ndarray,
    x: np.ndarray,
    y: float,
    ansatz: QuantumCircuit,
    param_list: List,
    estimator: Estimator,
    backend: Session
) -> float:
    """
    Runs the full circuit and computes the cost for one (x, y) pair.

    Args:
        params_values: current ansatz parameter values
        x: input vector (length 64)
        y: ground truth label (0 or 1)
        ansatz: parameterized circuit
        param_list: list of parameters in the ansatz
        estimator: Qiskit Estimator object

    Returns:
        Squared error between prediction and target
    """
    try:
        # Map y = 0 → +1, y = 1 → -1
        target = 1 if y == 0 else -1

        # Build full quantum circuit with assigned parameters
        ansatz_with_params = ansatz.assign_parameters(dict(zip(param_list, params_values)))
        full_circuit = build_full_circuit(x, ansatz_with_params, backend)

        # averaging out qubit measurements
        # TODO: check sources if this is used in practice
        observable = SparsePauliOp.from_list([("ZZZZZZ", 1.0)]) 

        # Run quantum computation
        job = estimator.run([(full_circuit, observable)])
        result = job.result()
        pub_result = result[0]
        output = pub_result.data.evs.item()

        # Mean squared error
        cost = (output - target) ** 2
        # hinge loss -> we punish if its too close to 0 
        cost = max(0, 1 - target * output)
        import logging
        logging.info(f"Sample cost: {cost:.4f} (target={target}, output={output:.4f})")
        return cost

    except Exception as e:
        print(f"Compute cost error: {str(e)}")
        return float('inf')  # Return high cost for failed evaluations
