"""
This script models the thermodynamic arrow of time by linking redshift (z) to entropy evolution, 
temperature, and energy density using Planck 2018 cosmological parameters. It calculates how 
entropy density and total entropy evolve as the universe expands, with redshift serving as a 
cosmic clock proxy.

In Thermal Relativity (TR), this visualization supports the principle that the 
thermodynamic arrow of time aligns with Absolute Time (T_A), and that entropy grows as a 
function of cosmic redshift. This reinforces the concept that cosmic evolution is a regulated 
thermodynamic process tied to temperature and expansion.

Outputs:
- CSV file: 'arrow_of_time.csv'
- Plot: 'absolute_time_arrow_of_time.png'

Author:
T. Matthew Ressler 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import Planck18 as cosmo

# Generate redshift values from high z (early universe) to now
z = np.logspace(np.log10(1100), np.log10(0.01), 1000)

# Absolute Time in seconds (converted from Gyr)
t_A = cosmo.age(z).value  # in Gyr
T_A = t_A * 3.154e16 * 1e9  # Convert Gyr to seconds

# Scale factor
a = 1 / (1 + z)

# Planck-calibrated constants
T_CMB = 2.72548  # Kelvin
rho0 = 1.0       # Arbitrary normalized energy density
T0 = T_CMB       # Initial temperature

# Temperature and energy density
T = T0 * (1 + z)
rho = rho0 * (1 + z)**3

# Entropy density and total entropy
s = rho / T
V = a**3
S = s * V

# Save output to CSV
df = pd.DataFrame({
    'Redshift z': z,
    'Absolute Time (s)': T_A,
    'Scale Factor a(z)': a,
    'Temperature (K)': T,
    'Energy Density': rho,
    'Entropy Density s(z)': s,
    'Total Entropy S(z)': S
})
df.to_csv("arrow_of_time.csv", index=False)

# Combined Plot: Temperature, Energy Density, and Entropy vs Redshift
plt.figure(figsize=(14, 6))

# Subplot 1: Redshift vs Temperature and Energy Density
plt.subplot(1, 2, 1)
plt.plot(z, T, label='Temperature (K)')
plt.plot(z, rho, label='Energy Density')
plt.xscale('log')
plt.xlabel('Redshift z (log scale)')
plt.ylabel('Temperature (K) and Energy Density (arb. units)')
plt.title('Temperature and Energy Density vs Redshift')
plt.legend()
plt.grid(True)

# Subplot 2: Redshift vs Total Entropy
plt.subplot(1, 2, 2)
plt.plot(z, S, label='Total Entropy', color='crimson')
plt.xscale('log')
plt.xlabel('Redshift z (log scale)')
plt.ylabel('Total Entropy (arbitrary units)')
plt.title('Entropy Evolution vs Redshift')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("absolute_time_arrow_of_time.png", dpi=300, bbox_inches='tight')
plt.show()
