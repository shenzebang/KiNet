import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter_2d(X, mins=np.array([-10, -10]), maxs=np.array([10, 10])):
    T, N, D = X.shape
    # Reshape and create DataFrame
    X_reshaped = X.reshape(-1, D)
    df = pd.DataFrame(X_reshaped)

    # Add time and particle index
    df["time"] = np.repeat(np.arange(T), N)
    df["particle"] = np.tile(np.arange(N), T)

    # Plot
    for t in range(T):
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df[df["time"] == t], x=0, y=1, hue="particle", palette="viridis")
        plt.ylim(mins[0], maxs[0])
        plt.xlim(mins[1], maxs[1])
        plt.title(f"Scatter plot at time {t}")
        plt.show()
