"""
This script models the thermodynamic evolution of the universe by connecting absolute time (T_A),
proper time (τ), and entropy through ΛCDM cosmology using Planck 2018 parameters. It tracks the 
scale factor a(T), energy density ρ(T), temperature T(T), entropy density s(T), and total entropy S(T) 
as functions of time. The purpose is to illustrate how entropy and temperature evolve in a 
self-governing universe, as proposed in Thermal Relativity (TR), where time is treated 
as an absolute measurable quantity influenced by local gravitational potential.

Outputs:
- CSV file: 'time_entropy_connection.csv'
- Plots: 'time_proof.png', 'entropy_vs_density.png'

Author:
T. Matthew Ressler 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Planck 2018 cosmological parameters
T_CMB = 2.72548  # K, current CMB temperature
H0_km_s_Mpc = 67.4  # Hubble constant in km/s/Mpc
H0 = H0_km_s_Mpc * 1000 / (3.086e22)  # Convert H0 to 1/s
Omega_m = 0.315
Omega_Lambda = 0.685
c = 3e8  # Speed of light in m/s
GM_r = 1e-6  # Local gravitational potential approximation

# Time array in seconds
T_A = np.linspace(1e15, 4.35e17, 1000)  # ~from 0.03 Myr to 13.8 Gyr

# Scale factor for a flat ΛCDM universe approximation
a = (Omega_m * (T_A / T_A[-1])**(2/3) + Omega_Lambda)**(1/3)

# Proper time based on local gravity
tau = T_A * np.sqrt(1 - GM_r)

# Energy density and temperature evolution
rho0 = 1.0  # Normalized
T0 = T_CMB
rho = rho0 / a**3
T = T0 / a

# Entropy density and total entropy
s = rho / T
V = a**3
S = s * V

# Create a dataframe for export
df = pd.DataFrame({
    'Absolute Time (s)': T_A,
    'Proper Time (s)': tau,
    'Scale Factor a(T)': a,
    'Energy Density': rho,
    'Temperature (K)': T,
    'Entropy Density': s,
    'Total Entropy': S
})

print(df.head())  # Show the first few rows of the data
df.to_csv("time_entropy_connection.csv", index=False)

# Plot 1: Scale Factor, Local Energy Density, and Total Entropy over Time
plt.figure(figsize=(8, 6))
plt.plot(T_A / (3.154e16 * 1e9), a, label='Scale Factor a(T)')
plt.plot(T_A / (3.154e16 * 1e9), rho, label='Local Energy Density')
plt.plot(T_A / (3.154e16 * 1e9), S, label='Total Entropy')
plt.xlabel('Absolute Time (Gyr)')
plt.ylabel('Value')
plt.title('Evolution of a(T), ρ(T), and S(T)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time_proof.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Entropy comparison
plt.figure(figsize=(8, 6))
plt.plot(s, S)
plt.xlabel('Local Entropy Density s(T)')
plt.ylabel('Total Entropy S(T)')
plt.title('Total Entropy vs Local Entropy Density')
plt.grid(True)
plt.tight_layout()
plt.savefig("entropy_vs_density.png", dpi=300, bbox_inches='tight')
plt.show()
