import matplotlib.pyplot as plt
import numpy as np
import scienceplots


DGMS = []

x = np.linspace(7, 17, 201)
benchmark = 79.20
benchmarks = np.ones_like(x)*benchmark

with plt.style.context(['science']):
    plt.plot(x, benchmarks)
    plt.show()