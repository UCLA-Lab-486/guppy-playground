# Implement Steane Code

## Implement the code

The paper fault-tolerant-execution.pdf explains a Steane code, which is an error
correcting code for quantum circuits.

The file main.py currently holds a simple example circuit. You need to implement
a Steane code and run the example circuit on the logical qubits. Do this in a
separate file, steane.py.

The quantum computing language is guppy, whose documentation is here:
https://docs.quantinuum.com/guppy/language_guide/language_guide_index.html

Make the design of your Steane code modular, so I can easily see the logical
gates that are operating on the logical qubits.

## Faulty qubit testing framework

Design a testing framework so we can run steane.py on emulated faulty physical
qubits. Make this parameterizable by failure rate. Check if guppy already has
tools for this.

## Run and verify

Finally, run steane.py using emulated faulty physical qubits, and compare
results with main.py.
