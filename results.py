from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator, Session
from qiskit.quantum_info import SparsePauliOp
import matplotlib.pyplot as plt
import numpy as np

from utils.data import load_mnist_0_vs_1_resized
from utils.circuit import create_ansatz, build_full_circuit
from config import *

def visualize(path: str):
    
    log_data = ""

    with open(path, 'r') as file:
        for line in file:
            if "Sample cost:" in line:
                log_data += line.strip() + "\n"

    target_pos = []
    target_neg = []

    for line in log_data.splitlines():
        if "Sample cost:" in line:
            parts = line.split("Sample cost:")[1].strip()
            cost_str, target_output_str = parts.split(" (")
            output_str = target_output_str.replace("target=", "").replace("output=", "").replace(")", "")
            target_val, output_val = output_str.split(", ")
            target_val = int(target_val)
            output_val = float(output_val)

            if target_val == 1:
                target_pos.append(output_val)
            else:
                target_neg.append(output_val)

    mean_pos = sum(target_pos) / len(target_pos) if target_pos else 0
    mean_neg = sum(target_neg) / len(target_neg) if target_neg else 0
    threshold = (mean_pos + mean_neg) / 2

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(target_pos, 'o-', label='Target = 1 (Positive)', markersize=5)
    plt.plot(target_neg, 'o-', label='Target = -1 (Negative)', markersize=5)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ≈ {threshold:.3f}')
    plt.title('Model Outputs Grouped by Target')
    plt.xlabel('Sample Index')
    plt.ylabel('Output Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def sample(file: str):
    _, X_test, _, y_test = load_mnist_0_vs_1_resized(size=8)
    trained_params = np.load(file)
    ansatz, _ = create_ansatz(num_qubits=6, num_layers=5)

    test_indices = np.random.choice(len(X_test), size=NUM_TEST_SAMPLES, replace=False)
    backend = AerSimulator()

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session) 

        correct = 0
        for i, idx in enumerate(test_indices):
            x = X_test[idx]
            y = y_test[idx]
            
            # Your normal predict flow
            full_circuit = build_full_circuit(x, ansatz, backend=backend)
            observable = SparsePauliOp.from_list([("ZZZZZZ", 1.0)])

            result = estimator.run([(full_circuit, observable, trained_params)]).result()
            output = result[0].data.evs.item()

            prediction = 0 if output >= 0 else 1
            print(f"Test {i+1}: Predicted = {prediction}, Actual = {int(y)}, Output = {output:.4f}")
            correct += (prediction == int(y))

    print(f"\nAccuracy on {NUM_TEST_SAMPLES} random test samples: {correct}/{NUM_TEST_SAMPLES}")

if __name__ == "__main__":

    log_path = "log\zzzzzz_err_mit.txt"
    params_path = "trained_params\zzzzzz_err_mit.npy"

    visualize(log_path)
    sample(params_path)