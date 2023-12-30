#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
print("Imports Successful")


# In[6]:


def qft_rotations(circuit, n):
    if n == 0: # Exit function if circuit is empty
        return circuit
    n -= 1 # Indexes start from 0
    circuit.h(n) # Apply the H-gate to the most significant qubit
    for qubit in range(n):
        # For each less significant qubit, we need to do a
        # smaller-angled controlled rotation: 
        circuit.cp(pi/2**(n-qubit), qubit, n)
        # At the end of our function, we call the same function again on
        # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)


# In[7]:


qc = QuantumCircuit(4)
qft_rotations(qc,4)
qc.draw()


# In[8]:


from qiskit_textbook.widgets import scalable_circuit
scalable_circuit(qft_rotations)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit



# Let's see how it looks:
qc = QuantumCircuit(4)
qft(qc,4)
qc.draw()


# In[9]:


# Let us now proceed to carry out approximate QFT. In order to do that, we modify the code qft rotation.We do that
# by the following prescription. Suppose that the rotation caused by CROT is denoted by pi / 2^{k-j} where k is the 
#control qubit and j is the qubit on which the rotation acts. we create a function to take in another parameter m 
#such that rotation on the qubits is caused only if the contraint (pi / 2^{k - j} ) > (pi / 2^{m}) is satisfied

def approx_qft_rotations(circuit, n, m):
    if n == 0: # Exit function if circuit is empty
        return circuit
    n -= 1 # Indexes start from 0
    circuit.h(n) # Apply the H-gate to the most significant qubit
    for qubit in range(n):
        # For each less significant qubit, check if the controlled rotation should be applied
        if pi/2**(n -qubit) > pi/2**m:
            # If the angle of the controlled rotation satisfies the constraint, apply the rotation
            circuit.cp(pi/2**(n-qubit), qubit, n)
        # At the end of our function, we call the same function again on
        # the next qubits (we reduced n by one earlier in the function)
    approx_qft_rotations(circuit, n, m)
    
    
#We now define approx_qft in terms of the above function approx_qft_rotations
def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def approx_qft(circuit, n,m):
    """QFT on the first n qubits in circuit"""
    approx_qft_rotations(circuit, n,m)
    swap_registers(circuit, n)
    return circuit



# In[10]:




