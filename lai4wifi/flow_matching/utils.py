import os
os.environ['JAX_ENABLE_X64'] = 'True'

from functools import partial

import jax
import jax.numpy as jnp


def gen_sample(graph):
    edges, nodes, globals = jax.tree.map(lambda x: x[:, :-1], (graph.edges, graph.nodes, graph.globals))
    return graph._replace(edges=edges, nodes=nodes, globals=globals)


@partial(jax.jit, static_argnames=['axis', 'total_repeat_length'])
def jax_repeat(a, repeats, axis, total_repeat_length):
    return jnp.repeat(a, repeats, axis, total_repeat_length=total_repeat_length)
