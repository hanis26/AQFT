#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from math import gcd # greatest common divisor
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
print("Imports Successful")


# In[2]:


def two_mod_21(power):
    U = QuantumCircuit(5)
    for iteration in range(power):
        U.cswap(0,3,4)
        U.cswap(0,1,2)
        U.cx(4,2)
        U.cx(4,0)
        U.swap(3,4)
        U.swap(0,3)
        U.swap(3,2)
        U.swap(2,1)
    U = U.to_gate()
    U.name = "2^%i mod 32" % (power)
    c_U = U.control()
    return c_U


# In[3]:


def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc


# In[4]:


# Specify variables
n_count = 5  # number of counting qubits
# Create QuantumCircuit with n_count counting qubits
# plus 5 qubits for U to act on
qc = QuantumCircuit(n_count + 5, n_count)

# Initialize counting qubits
# in state |+>
for q in range(n_count):
    qc.h(q)

# And auxiliary register in state |1>
qc.x(n_count)

# Do controlled-U operations
for q in range(n_count):
    qc.append(two_mod_21(2**q), 
              [q] + [i+n_count for i in range(5)])

# Do inverse-QFT
qc.append(qft_dagger(n_count), range(n_count))

# Measure circuit
qc.measure(range(n_count), range(n_count))
qc.draw(fold=-1)  # -1 means 'do not fold'


# In[12]:


# Example error probabilities
p_gate1 = 0.00007

# QuantumError objects
error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_model = NoiseModel()

noise_model.add_all_qubit_quantum_error(error_gate2, "cx")
                                        
sim_noise = AerSimulator(noise_model = noise_model)
t_qc = transpile(qc, sim_noise)


# aer_sim = Aer.get_backend('aer_simulator')
# t_qc = transpile(qc, aer_sim)
# qobj = assemble(t_qc)
# results = aer_sim.run(qobj).result()
# counts = results.get_counts()

counts = sim_noise.run(t_qc).result().get_counts()

# Define the threshold here
threshold = 20

# Filter out the counts less than the threshold

counts = {state: count for state, count in counts.items() if count >= threshold}

print(counts)
plot_histogram(counts)


# In[14]:


rows, measured_phases = [], []
for output in counts:
    decimal = int(output, 2)  # Convert (base 2) string to decimal
    phase = decimal/(2**n_count)  # Find corresponding eigenvalue
    measured_phases.append(phase)
    # Add these values to the rows in our table:
    rows.append([f"{output}(bin) = {decimal:>3}(dec)", 
                 f"{decimal}/{2**n_count} = {phase:.2f}"])
# Print the rows in a table
headers=["Register Output", "Phase"]
df = pd.DataFrame(rows, columns=headers)
print(df)


# In[16]:


a = 2
N = 21
FACTOR_FOUND = 0
ATTEMPT = 0
for output in counts:
    if FACTOR_FOUND == 2: break
    else: FACTOR_FOUND = 0
    ATTEMPT += 1
    print(f"\nATTEMPT {ATTEMPT}:")
    phase = int(output, 2)/(2**n_count)
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    print(f"Phase: {phase} Result: r = {r}")
    if phase != 0:
        # Guesses for factors are gcd(x^{r/2} ±1 , 15)
        guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
        print(f"Guessed Factors: {guesses[0]} and {guesses[1]}")
        for guess in guesses:
            if guess not in [1,N] and (N % guess) == 0:
                # Guess is a factor!
                print("*** Non-trivial factor found: "+str(guess)+" ***")
                FACTOR_FOUND = FACTOR_FOUND + 1


# In[ ]:




