import collections
import pickle
import numpy as np
import pandas as pd
import collipy as cp
from collipy.helper import function_timer
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)
fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw=dict(hspace=0, height_ratios=[3, 1]))
fig.suptitle('Sine Wave Title')
fig.supxlabel('$m[MeV]$')
ax1.set_ylabel('$E[GeV]$')
ax1.plot(x, y)
ax1.axvspan(0, 1, color='C6')
ax1.grid()
ax2.set_ylabel('$residuals$')
ax2.plot(x + 1, -y)
ax2.axhline(0, ax2.get_xlim()[0], ax2.get_xlim()[1], color='C3', linewidth=1.1)
ax2.grid()