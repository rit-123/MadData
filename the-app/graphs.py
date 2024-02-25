import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('lunges5.csv')
columns = df.columns

fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.flatten()

for i, ax in enumerate(axs):
    if i < len(columns):
        ax.plot(df[columns[i]])
        ax.set_title(f'Graph {i+1}: {columns[i]}')
        # ax.set_xlabel('Column Number')
        # ax.set_ylabel('Column Values')

plt.tight_layout()
plt.show()

