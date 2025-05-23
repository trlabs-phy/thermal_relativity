"""
This script computes and visualizes the evolution of redshift, temperature, energy density, 
and entropy in terms of Absolute Time (T_A), using Planck 2018 cosmological parameters via Astropy. 
It spans from the early universe (z ~ 1100) to the present, linking redshift (z) to the progression 
of cosmic time and entropy growth.

Within Thermal Relativity (TR), this script supports the concept that redshift is 
thermodynamically governed and mapped onto a measurable absolute time framework. It tracks 
entropy production, energy density scaling, and thermal history as the universe evolves.

Outputs:
- CSV file: 'absolute_time_redshift.csv'
- Plot: 'redshift_over_absolute_time.png'

Author:
T. Matthew Ressler 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import Planck18 as cosmo

# Generate redshift values from high z (early universe) to now
z = np.logspace(np.log10(1100), np.log10(0.01), 1000)

# Age of universe at each redshift (in Gyr)
t_A = cosmo.age(z).value  # Absolute Time in Gyr
T_A = t_A * 3.154e16 * 1e9  # Convert to seconds

# Scale factor and redshift
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

# Compute age of the universe at each redshift (in Gyr)
t_A = cosmo.age(z).value  # Absolute time in Gyr

# Plot Redshift vs Absolute Time
plt.figure(figsize=(10, 6))
plt.plot(t_A, z, color='red')
plt.xlabel('Absolute Time (Gyr)')
plt.ylabel('Redshift z')
plt.title('Redshift Over Absolute Time (Planck 2018)')
plt.grid(True)
plt.tight_layout()
plt.savefig("redshift_over_absolute_time.png", dpi=300, bbox_inches='tight')
plt.show()

# Save data to CSV
df = pd.DataFrame({
    'Redshift z': z,
    'Absolute Time (s)': T_A,
    'Total Entropy (S)': S,
    'Scale Factor a(T)': a,
    'Temperature (K)': T,
    'Energy Density (rho)': rho,
    'Entropy Density (s)': s
})
df.to_csv("absolute_time_redshift.csv", index=False)
