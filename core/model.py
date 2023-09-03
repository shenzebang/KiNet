import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Tuple
from core.normalizing_flow import TimeEmbedding
from example_problems.euler_poisson_with_drift import ground_truth_op_vmapx, ground_truth_op_uniform

class MLP(nn.Module):
    output_dim: int = 1
    hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    # hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    scaling: float = 1.
    use_normalization: bool = False
    time_embedding_dim: int = 0
    append_time: bool = True

    def setup(self):
        self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.layers_time = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.normalizations = [nn.LayerNorm() for _ in list(self.hidden_dims) + [self.output_dim]]
        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=20,)
        # self.act = jax.nn.sigmoid
        # self.act = jax.nn.relu
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
        # if t.ndim != 2 or t.shape[1] != 1:
        #     raise ValueError("t should be a 2D array and the second dimension should be 1")
        #
        # if x.ndim != 2:
        #     raise ValueError("x should be a 2D array")

        if z.ndim == 2 and z.shape[0] != t.shape[0]:
            raise ValueError("x.shape[0] should agree with t.shape[0]")

        assert t.ndim == 1 and z.ndim == 1
        # tx = jnp.concatenate([t, x], axis=-1)
        # y = tx
        if self.time_embedding_dim > 0:
            t = self.time_embedding(t)
        if self.append_time:
            y = jnp.concatenate([t, z], axis=-1)
        else:
            y = z
        for i, (layer, layer_t, normalization) in enumerate(zip(self.layers, self.layers_time, self.normalizations)):
            y = layer(y)
            if i < len(self.layers) - 1:
                if not self.append_time and self.time_embedding_dim > 0:
                    time_t = layer_t(t)
                    y = y + time_t
                if self.use_normalization:
                    y = self.act(normalization(y))
                else:
                    y = self.act(y)

        # return jax.nn.relu(y)
        return jnp.exp(y) * self.scaling

    # def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
    #     # if t.ndim != 2 or t.shape[1] != 1:
    #     #     raise ValueError("t should be a 2D array and the second dimension should be 1")
    #     #
    #     # if x.ndim != 2:
    #     #     raise ValueError("x should be a 2D array")
    #
    #     if x.ndim == 2 and x.shape[0] != t.shape[0]:
    #         raise ValueError("x.shape[0] should agree with t.shape[0]")
    #
    #     tx = jnp.concatenate([t, x], axis=-1)
    #     y = tx
    #     for i, layer in enumerate(self.layers):
    #         y = layer(y)
    #         if i < len(self.layers) - 1:
    #             y = self.act(y)
    #
    #     return jax.nn.relu(y)

class KiNet(nn.Module):
    time_embedding_dim: int = 0
    output_dim: int = 2
    time_dim: int = 10
    # hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    # hidden_dims: Tuple[int] = (50, 50, 50, 50, 50, 50, 50, 50,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100, 100, 100, 100, 100,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100,)
    def setup(self):
        self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=self.time_embedding_dim, )
        self.layer_time = nn.Dense(self.time_dim)
        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
        if t.ndim == 0:
            t = jnp.ones(1) * t

        if self.time_embedding_dim > 0:
            t = self.time_embedding(t)

        # t = self.layer_time(t)
        if z.ndim == 2:
            t = jnp.broadcast_to(t, (z.shape[0], t.shape[0]))
        z = jnp.concatenate([t, z], axis=-1)

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = self.act(z)

        return z


class KiNet_Debug_2(nn.Module):
    time_embedding_dim: int = 0
    output_dim: int = 2
    time_dim: int = 10
    # hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    # hidden_dims: Tuple[int] = (50, 50, 50, 50, 50, 50, 50, 50,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100, 100, 100, 100, 100,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100,)
    def setup(self):
        self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [1]]
        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=self.time_embedding_dim, )
        self.layer_time = nn.Dense(self.time_dim)
        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
        if t.ndim == 0:
            t = jnp.ones(1) * t

        if self.time_embedding_dim > 0:
            t = self.time_embedding(t)

        # t = self.layer_time(t)
        if z.ndim == 2:
            t = jnp.broadcast_to(t, (z.shape[0], t.shape[0]))
        # z = jnp.concatenate([t, z], axis=-1)

        for i, layer in enumerate(self.layers):
            t = layer(t)
            if i < len(self.layers) - 1:
                t = self.act(t)

        return z * t

class KiNet_Debug(nn.Module):
    time_embedding_dim: int = 0
    output_dim: int = 2
    time_dim: int = 10
    # hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    # hidden_dims: Tuple[int] = (50, 50, 50, 50, 50, 50, 50, 50,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100, 100, 100, 100, 100,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100,)
    def setup(self):
        self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=self.time_embedding_dim, )
        self.layer_time = nn.Dense(self.time_dim)
        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
        assert t.ndim == 0 or t.ndim == 1
        if t.ndim == 1:
            t = t[0]
        # x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        if x.ndim == 2:
            return ground_truth_op_vmapx(t, x)
        elif x.ndim == 1:
            return ground_truth_op_uniform(t, x)
        else:
            raise ValueError("xs should be of size either (batch, dim) or (dim).")
