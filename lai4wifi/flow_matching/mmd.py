import jax
import jax.numpy as jnp


@jax.jit
def sq_distances(x, y):
    x_sqnorms = jnp.sum(x ** 2, axis=-1, keepdims=True)
    y_sqnorms = jnp.sum(y ** 2, axis=-1, keepdims=True)
    return x_sqnorms + y_sqnorms.T - 2 * jnp.dot(x, y.T)


@jax.jit
def rbf_kernel(x, y, gamma):
    sq_dists = sq_distances(x, y)
    return jnp.exp(-gamma * sq_dists)


@jax.jit
def multiscale_rbf_kernel(x, y, gammas):
    Ks = jax.vmap(rbf_kernel, in_axes=(None, None, 0))(x, y, gammas)
    return Ks.mean(axis=0)


@jax.jit
def median_heuristic(x, y=None):
    if y is None:
        y = x

    sq_dists = sq_distances(x, y)
    mask = jnp.triu(jnp.ones_like(sq_dists), k=1)
    dists = jnp.sqrt(jnp.maximum(sq_dists, 0.0))
    dists_with_nans = jnp.where(mask > 0, dists, jnp.nan)
    return jnp.nanmedian(dists_with_nans)


@jax.jit
def compute_mmd(x, y, gammas=None):
    n, m = x.shape[0], y.shape[0]

    if gammas is None:
        combined = jnp.concatenate([x, y], axis=0)
        median_dist = median_heuristic(combined)
        base_gamma = 1.0 / (2.0 * jnp.maximum(median_dist ** 2, 1e-10))
        scales = jnp.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        gammas = base_gamma * scales

    K_xx = multiscale_rbf_kernel(x, x, gammas)
    K_yy = multiscale_rbf_kernel(y, y, gammas)
    K_xy = multiscale_rbf_kernel(x, y, gammas)

    sum_K_xx = jnp.sum(K_xx) - jnp.trace(K_xx)
    sum_K_yy = jnp.sum(K_yy) - jnp.trace(K_yy)
    sum_K_xy = jnp.sum(K_xy)

    mmd_sq = sum_K_xx / (n * (n - 1)) + sum_K_yy / (m * (m - 1)) - 2 * sum_K_xy / (n * m)
    return jnp.sqrt(jnp.maximum(mmd_sq, 0.0))
