import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('loadgen/results.csv')

plt.plot(sorted(data.iloc[1:, 2]))
plt.savefig('latencies_sorted.png', dpi=150, bbox_inches='tight')

plt.clf()
plt.plot(data.iloc[1:, 2])
plt.savefig('latencies.png', dpi=150, bbox_inches='tight')


plt.close()