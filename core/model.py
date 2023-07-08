import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Tuple
from core.normalizing_flow import TimeEmbedding


class MLP(nn.Module):
    output_dim: int = 1
    hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    scaling: float = 1.

    def setup(self):
        self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.layers_time = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.normalizations = [nn.LayerNorm() for _ in list(self.hidden_dims) + [self.output_dim]]
        self.time_embedding = TimeEmbedding(dim=20,)
        # self.act = jax.nn.sigmoid
        # self.act = jax.nn.relu
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
        # if t.ndim != 2 or t.shape[1] != 1:
        #     raise ValueError("t should be a 2D array and the second dimension should be 1")
        #
        # if x.ndim != 2:
        #     raise ValueError("x should be a 2D array")

        if x.ndim == 2 and x.shape[0] != t.shape[0]:
            raise ValueError("x.shape[0] should agree with t.shape[0]")

        assert t.ndim == 1 and x.ndim == 1
        # tx = jnp.concatenate([t, x], axis=-1)
        # y = tx
        time_embedding_t = self.time_embedding(t)
        y = x
        for i, (layer, layer_t, normalization) in enumerate(zip(self.layers, self.layers_time, self.normalizations)):
            y = layer(y)
            if i < len(self.layers) - 1:
                time_t = layer_t(time_embedding_t)
                y = y + time_t
                y = self.act(normalization(y))

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
    output_dim: int = 2
    hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    # hidden_dims: Tuple[int] = (50, 50, 50, 50, 50, 50, 50, 50,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100, 100, 100, 100, 100,)
    def setup(self):
        self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.layers_time = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.time_embedding = TimeEmbedding(dim=20, )
        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
        # if t.ndim != 2 or t.shape[1] != 1:
        #     raise ValueError("t should be a 2D array and the second dimension should be 1")
        #
        # if x.ndim != 2:
        #     raise ValueError("x should be a 2D array")
        if type(t) is float or type(t) is int:
            t = jnp.ones(1) * t
        elif t.ndim == 0:
            t = jnp.ones(1) * t

        # if z.ndim == 2:
        #     t = jnp.ones([z.shape[0], 1]) * t



        # if z.ndim == 2 and z.shape[0] != t.shape[0]:
        #     raise ValueError("x.shape[0] should agree with t.shape[0]")

        # tz = jnp.concatenate([t, z], axis=-1)
        time_embedding_t = self.time_embedding(t)

        y = z
        for i, (layer, layer_t) in enumerate(zip(self.layers, self.layers_time)):
            y = layer(y)
            if i < len(self.layers) - 1:
                time_t = layer_t(time_embedding_t)
                y = y + time_t
                y = self.act(y)

        return y


# class KiNet(nn.Module):
#     output_dim: int = 2
#     hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
#     # hidden_dims: Tuple[int] = (50, 50, 50, 50, 50, 50, 50, 50,)
#     # hidden_dims: Tuple[int] = (100, 100, 100, 100, 100, 100, 100, 100,)
#     def setup(self):
#         self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
#         # self.act = jax.nn.sigmoid
#         self.act = jax.nn.tanh
#
#     def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
#         # if t.ndim != 2 or t.shape[1] != 1:
#         #     raise ValueError("t should be a 2D array and the second dimension should be 1")
#         #
#         # if x.ndim != 2:
#         #     raise ValueError("x should be a 2D array")
#         if type(t) is float or type(t) is int:
#             t = jnp.ones(1) * t
#         elif t.ndim == 0:
#             t = jnp.ones(1) * t
#
#         if z.ndim == 2:
#             t = jnp.ones([z.shape[0], 1]) * t
#
#
#
#         if z.ndim == 2 and z.shape[0] != t.shape[0]:
#             raise ValueError("x.shape[0] should agree with t.shape[0]")
#
#         tz = jnp.concatenate([t, z], axis=-1)
#         y = tz
#         for i, layer in enumerate(self.layers):
#             y = layer(y)
#             if i < len(self.layers) - 1:
#                 y = self.act(y)
#
#         return y