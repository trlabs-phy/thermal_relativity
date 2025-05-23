"""
This script calculates the proper time difference between a clock on Earth's surface and an ideal 
clock in a gravity-free frame (Absolute Time), using general relativistic time dilation due to 
Earth’s gravitational potential. It computes both the instantaneous and cumulative time difference 
over the age of the Earth and the universe, based on the factor GM/rc².

Within the Thermal Relativity (TR), this quantifies the deviation of Proper Time (τ) from 
Absolute Time (T_A), reinforcing the concept that gravitational fields locally affect the 
perception of time. The script supports the ToU assertion that time is a measurable quantity, 
with τ corrected by the ATF through gravitational influences.

Outputs include:
- Proper time rate per second
- Time deviation per nanosecond and per year
- Total time lost relative to Absolute Time over geological and cosmological timescales

Author:
T. Matthew Ressler 2025
"""

import numpy as np

# Planck-based constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
M_earth = 5.972e24  # mass of Earth (kg)
R_earth = 6.371e6  # radius of Earth (m)
c = 3e8  # speed of light (m/s)

# Calculate GM/rc^2 for Earth
GM_rc2 = (G * M_earth) / (R_earth * c**2)

# Time values
seconds_per_year = 3.154e7
age_of_earth_yrs = 4.54e9
age_of_universe_yrs = 13.8e9

# Convert to seconds
age_of_earth_sec = age_of_earth_yrs * seconds_per_year
age_of_universe_sec = age_of_universe_yrs * seconds_per_year

# Proper Time calculation
tau_per_second = np.sqrt(1 - 2 * GM_rc2)
delta_per_second = 1 - tau_per_second

# Derived time differences
delta_per_nanosecond = delta_per_second * 1e9
delta_per_year_ms = delta_per_second * seconds_per_year * 1e3

# Cumulative difference
delta_earth_age = delta_per_second * age_of_earth_sec
delta_universe_age = delta_per_second * age_of_universe_sec

# Time difference per year in milliseconds
delta_per_year_ms = delta_per_second * seconds_per_year * 1e3

# Output results
results = {
    "GM/rc^2 (Earth)": GM_rc2,
    "Tau per second": tau_per_second,
    "Time difference per second (ns)": delta_per_nanosecond,
    "Time difference per year (ms)": delta_per_year_ms,
    "Time lost over Earth's age (years)": delta_earth_age / seconds_per_year,
    "Time lost over Universe's age (years)": delta_universe_age / seconds_per_year
}

for key, value in results.items():
    print(f"{key}: {value}")
