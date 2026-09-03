import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('loadgen/results.csv')

sorted_lat = sorted(data.iloc[1:, 2])
plt.plot(sorted_lat)
plt.savefig('latencies_sorted.png', dpi=150, bbox_inches='tight')
total = len(sorted_lat)
print(f'P50 = {sorted_lat[total//2]}')
print(f'P99 = {sorted_lat[total*99//100]}')
print(f'P99.9 = {sorted_lat[total*999//1000]}')

plt.clf()
plt.plot(data.iloc[1:, 2])
plt.savefig('latencies.png', dpi=150, bbox_inches='tight')


plt.close()