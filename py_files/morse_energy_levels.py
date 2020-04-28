# Generates figure detailing the hierarchy of Hamiltonian approach applied to
#   the Morse potential

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def V_plus(x, A):
    """Super symmetric representation of the Morse potential"""
    return np.exp(-2 * x) - ((2 * A) - 1) * np.exp(-x) + A ** 2

def V_minus(x, A):
    """Super symmetric representation of the Morse potential"""
    return np.exp(-2 * x) - ((2 * A) + 1) * np.exp(-x) + A ** 2

A = 5
x_data = np.linspace(-3, 5, num=500)

V_0 = V_minus(x_data, A)
V_1 = V_plus(x_data, A)
V_2 = V_plus(x_data, A-1) + 2 * (A - 1) + 1
V_3 = V_plus(x_data, A-2) + 2 * (A - 1) + 1 + 2 * (A - 2) + 1
V_4 = V_plus(x_data, A-3) + 2 * (A - 1) + 1 + 2 * (A - 2) + 1 + 2 * (A - 3) + 1

pots = [V_0, V_1, V_2, V_3, V_4]

# Generating figure
plt.figure(figsize=(13,10))
plt.ylim(bottom=-8, top=35)
lw = 3          #line width

# Potentials
c1 = '#2ca02c'  # color for V_1 - V_4
z1 = 1          # z order
plt.plot(x_data, V_4,linewidth=lw, c=c1, zorder=z1)
plt.plot(x_data, V_3,linewidth=lw, c=c1, zorder=z1)
plt.plot(x_data, V_2,linewidth=lw, c=c1, zorder=z1)
plt.plot(x_data, V_1, label='$V_{i}$',linewidth=lw, c=c1, zorder=z1)
plt.plot(x_data, V_0, label='$V_{0}$',linewidth=lw, c='#1f77b4', zorder=z1)

# Energy Levels
c2 = '#d62728'  # color for E_1 - E_4
z2 = 2          # z order
plt.hlines(y=0, xmin=-2.05, xmax=-1.18, colors=c2, linewidth=lw, zorder=z2)
plt.hlines(y=9, xmin=-2.24, xmax=-0.55, colors=c2, linewidth=lw, zorder=z2)
plt.hlines(y=16, xmin=-2.3, xmax=0.1, colors=c2, linewidth=lw, zorder=z2)
plt.hlines(y=21, xmin=-2.38, xmax=0.94, colors=c2,linewidth=lw, zorder=z2)
plt.hlines(y=24, xmin=-2.4, xmax=2.3, colors=c2, linewidth=lw, zorder=z2)

# Energy level labels
format_dict = {'color': 'k', 'fontsize': 36}
plt.text(-2.6, -1, r'$E_{0}$', format_dict)
plt.text(-2.75, 8, r'$E_{1}$', format_dict)
plt.text(-2.85, 15, r'$E_{2}$', format_dict)
plt.text(-2.9, 19.8, r'$E_{3}$', format_dict)
plt.text(-2.95, 23.3, r'$E_{4}$', format_dict)

# Plot formatting
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
sns.despine()
plt.legend(fontsize=36, loc='right')
plt.ylabel('Energy', size=36)
plt.xlabel('x', size=36)
plt.show();
