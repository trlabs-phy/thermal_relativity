"""
This script visualizes two natural cosmological clocks—redshift and entropy—mapped against 
Absolute Time (T_A), using Planck 2018 data via Astropy. It constructs a synthetic entropy function 
S(T_A) and compares it to the observational redshift curve over the history of the universe.

In Thermal Relativity (TR), this supports the idea that entropy and redshift both serve 
as timekeepers within a thermodynamically evolving universe. Redshift reflects cosmic expansion, 
while entropy captures the irreversible arrow of thermodynamic time. Together, they reinforce the 
framework’s core claim that time is a measurable quantity, governed by the interaction between 
Absolute Time (ATF) and the universe’s thermal history.

Outputs:
- CSV file: 'planck_entropy_clock.csv'
- Plot: '2_clocks.png'

Author:
T. Matthew Ressler 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import Planck18 as cosmo

# === Redshift range ===
z = np.logspace(np.log10(1100), np.log10(0.01), 1000)

# === Get Absolute Time T_A at each redshift from Planck18 (in Gyr) ===
t_A_gyr = cosmo.age(z).value  # Age of the universe at each redshift
T_A = t_A_gyr * 3.154e16 * 1e9  # Convert Gyr to seconds

# === Simulate entropy growth S(T_A) ===
S = np.log1p((T_A / T_A.max()) * (np.exp(30) - 1))

# === Create DataFrame ===
df_astropy = pd.DataFrame({
    'Redshift z': z,
    'Absolute Time T_A (s)': T_A,
    'Entropy S(T_A)': S
})

# === Export to CSV ===
df_astropy.to_csv("planck_entropy_clock.csv", index=False)

# === Plotting ===
plt.figure(figsize=(12, 5))

# Redshift as a clock
plt.subplot(1, 2, 1)
plt.plot(T_A / (3.154e16 * 1e9), z)
plt.xlabel("Absolute Time T_A (Gyr)")
plt.ylabel("Redshift z")
plt.title("Redshift as a Clock (Planck 2018 via Astropy)")
plt.grid(True)

# Entropy as a clock
plt.subplot(1, 2, 2)
plt.plot(T_A / (3.154e16 * 1e9), S)
plt.xlabel("Absolute Time T_A (Gyr)")
plt.ylabel("Entropy S")
plt.title("Entropy as a Clock (Planck 2018 via Astropy)")
plt.grid(True)

plt.tight_layout()
plt.savefig("2_clocks.png", dpi=300, bbox_inches='tight')
plt.show()
