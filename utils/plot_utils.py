import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp

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
        # only the first two dimensions are used, even in the Kinetic case
        plt.ylim(mins[0], maxs[0])
        plt.xlim(mins[1], maxs[1])
        plt.title(f"Scatter plot at time {t}")
        plt.show()

def plot_density_2d(f, config=None):
    # Sample data
    if config is None:
        side = jnp.linspace(-10, 10, 256)
        X, Y = jnp.meshgrid(side, side)
    else:
        mins = config["mins"]
        maxs = config["maxs"]
        side_x = jnp.linspace(mins[0], maxs[0], 256)
        side_y = jnp.linspace(mins[1], maxs[1], 256)
        X, Y = jnp.meshgrid(side_x, side_y)

    XY = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z = f(XY)
    Z = Z.reshape(X.shape)

    # Plot the density map using nearest-neighbor interpolation
    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(X, Y, Z)
    fig.colorbar(pcm, ax=ax)
    plt.show()