from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit import transpile
from qiskit.circuit.library import StatePreparation
import numpy as np


def create_ansatz(num_qubits: int, num_layers: int = 1) -> tuple:
    """
    Creates a variational ansatz with optional entanglement.
    
    Each layer:
    - Applies Ry and Rz rotations with trainable parameters to each qubit
    - Optionally applies entangling CNOTs
    
    Returns:
        - QuantumCircuit object
        - List of parameters used
    """
    qc = QuantumCircuit(num_qubits)
    param_count = 2 * num_qubits * num_layers
    params = ParameterVector("theta", length=param_count)

    p = 0  # param index
    for _ in range(num_layers):
        for q in range(num_qubits):
            qc.ry(params[p], q)
            p += 1
            qc.rz(params[p], q)
            p += 1

        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

    return qc, list(params)


def create_amplitude_encoding_circuit(x: np.ndarray) -> QuantumCircuit:
    """Create a circuit that encodes the input vector into quantum amplitudes."""
    num_qubits = 6
    
    # Ensure perfect normalization
    x = np.asarray(x, dtype=np.complex128).flatten()
    x = x / np.linalg.norm(x)
    
    # Create and verify statevector
    stateprep = StatePreparation(x)
    prep_circuit = QuantumCircuit(num_qubits)
    prep_circuit.append(stateprep, range(num_qubits))

    prep_circuit = prep_circuit.decompose(reps=10) # Decompose to basic gates
  
    return prep_circuit

def build_full_circuit(x: np.ndarray, ansatz: QuantumCircuit, backend) -> QuantumCircuit:
    input_circuit = create_amplitude_encoding_circuit(x)
    full_circuit = input_circuit.compose(ansatz)
    
    transpiled = transpile(full_circuit, backend=backend, optimization_level=1)

    assert transpiled.num_qubits == 6, f"Transpiled circuit has {transpiled.num_qubits} qubits instead of 6"
    return transpiled