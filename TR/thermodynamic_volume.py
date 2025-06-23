"""
thermodyanmic_volume.py

This script calculates the thermodynamic volume of the observable universe (governed by the Proper Time Field, PTF),
the extended volume of the Absolute Time Field (ATF), and the volume of the intervening ATF Shell (ATF-only region).
It uses the time difference between Absolute Time (T_A) and Proper Time (τ), attributed to the delay of gravity's 
activation in the Planck epoch be 5 billion years, to derive the radial extension of the ATF Shell and its 
corresponding volume.

These calculations support the Thermal Relativity framework by quantifying the measurable thermodynamic boundary
beyond the PTF, interpreted as the ATF Shell—a region where entropy and gravitational dynamics
cease. Results are expressed in gigalight-years (Gly) and cubic gigalight-years (Gly³).

Author: T. Matthew Ressler
"""
import numpy as np
import pandas as pd

# Set display options to avoid scientific notation
pd.set_option('display.float_format', '{:.12f}'.format)

# Constants
c_km_s = 299792.458  # Speed of light in km/s
seconds_per_year = 3.154e7
gly_to_km = 9.461e17  # 1 Gly in km

# Inputs
R_Gly = 46.5  # Radius of observable universe in Gly
delta_T_years = 5e9  # Absolute Time minus Proper Time in years
delta_T_sec = delta_T_years * seconds_per_year

# Convert distances to km
R_km = R_Gly * gly_to_km
delta_R_km = c_km_s * delta_T_sec
R_total_km = R_km + delta_R_km

# Compute volumes in km^3
V_ptf_km3 = (4 / 3) * np.pi * R_km**3
V_total_km3 = (4 / 3) * np.pi * R_total_km**3
V_atfshell_km3 = V_total_km3 - V_ptf_km3

# Convert volumes to Gly³
km3_to_gly3 = (1 / gly_to_km)**3
V_ptf_gly3 = V_ptf_km3 * km3_to_gly3
V_total_gly3 = V_total_km3 * km3_to_gly3
V_atfshell_gly3 = V_atfshell_km3 * km3_to_gly3

# Compile results into DataFrame
df_cleaned = pd.DataFrame({
    'Region': ['Physical Universe (PTF)', 'ATF Shell (ATF-only)', 'Total (PTF + ATF Shell)'],
    'Radius (Gly)': [R_Gly, delta_R_km / gly_to_km, R_total_km / gly_to_km],
    'Volume (Gly³)': [V_ptf_gly3, V_atfshell_gly3, V_total_gly3]
})

# Print
print(df_cleaned)

# Export to CSV
df_cleaned.to_csv("thermodynamic_volumes.csv", index=False)