from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import numpy as np
import matplotlib.pyplot as plt

# function to perform quantum addition
def quantum_addition(qc, a, b, c):
    # apply the approximate Fourier transform to the first register
    for i in range(len(a)):
        qc.h(a[i])
        for j in range(i+1, len(a)):
            qc.cp(2 * np.pi / (2 ** (j-i)), a[j], a[i])

    # apply the controlled phase shift gates to the second register
    for i in range(len(b)):
        qc.cp(2 * np.pi * 2 ** i, a[len(a) - i - 1], b[i])

    # apply the inverse approximate Fourier transform to the first register
    for i in reversed(range(len(a))):
        for j in reversed(range(i+1, len(a))):
            qc.cp(-2 * np.pi / (2 ** (j-i)), a[j], a[i])
        qc.h(a[i])

    # copy the result from the first register to the third register
    qc.cx(a, c)

# create three quantum registers
a = QuantumRegister(3, 'a')  # first input register
b = QuantumRegister(3, 'b')  # second input register
c = QuantumRegister(3, 'c')  # output register

# create a classical register to measure the output register
m = ClassicalRegister(3, 'm')

# create a quantum circuit
qc = QuantumCircuit(a, b, c, m)

# set the inputs
input_values = ['000', '001', '010', '011', '100', '101', '110', '111']
input_counts = [0] * len(input_values)

# loop over all possible input values
for i in range(len(input_values)):
    input_value = input_values[i]

    # set the inputs
    for j in range(len(input_value)):
        if input_value[j] == '1':
            qc.x(a[j])

    # perform quantum addition
    quantum_addition(qc, a, b, c)

    # measure the output register
    qc.measure(c, m)

    # run the quantum circuit on the simulator
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1024)
    counts = job.result().get_counts()

    # print the results
    input_counts[i] = counts.get(input_value, 0)

# plot the results
plt.figure()
plt.bar(input_values, input_counts)
plt.title('Quantum Addition Results')
plt.xlabel('Input Values')
plt.ylabel('Counts')
plt.show()



