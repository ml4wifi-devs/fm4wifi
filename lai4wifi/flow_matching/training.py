import jax
import jax.numpy as jnp
import optax

from lai4wifi.flow_matching.model import FMUnet


def loss_fn(model, params, key, x, cond, schedule='uniform'):
    z_key, t_key = jax.random.split(key)

    if schedule == 'log-normal':
        t = jax.random.normal(t_key, (x.shape[0],))
        t = 1 / (1 + jnp.exp(-t))
    elif schedule == 'uniform':
        t = jax.random.uniform(t_key, (x.shape[0],), minval=0.0, maxval=0.99)
    else:
        raise ValueError(f'Invalid schedule: {schedule}')

    z = jax.random.normal(z_key, x.shape)
    t_b = t.reshape((-1,) + (1,) * (len(x.shape) - 1))

    x_t = (1 - t_b) * z + t_b * x
    v_t = x - z

    v_pred = model.apply(params, x_t, cond, t)
    return optax.squared_error(v_pred, v_t).mean()


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    data = load()
    sample = next(data)

    model = FMUnet()
    params = model.init({'params': init_key}, *sample)

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    print(model.tabulate(jax.random.PRNGKey(0), *sample))
    generate_fn = jax.jit(lambda params, key, *x: model.apply(params, *x, rngs={'lai': key}, method='gen'))

    ...
