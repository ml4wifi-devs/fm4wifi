import enum
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class Connection:
    active: jax.Array
    mcs: jax.Array
    rate: jax.Array
    rssi: jax.Array
    selected: jax.Array
    tx_power: jax.Array
    type: jax.Array


class JaxEnum(enum.Enum):
    def __jax_array__(self):
        return jax.nn.one_hot(jnp.asarray(self.value), len(type(self)) - 1)


class EdgeType(JaxEnum):
    AP = 0
    STA = 1
    NAN = 2


class IsActive(JaxEnum):
    FALSE = 0
    TRUE = 1
    NAN = 2


class IsSelected(JaxEnum):
    FALSE = 0
    TRUE = 1
    NAN = 2


class McsValue(JaxEnum):
    MCS_0 = 0
    MCS_1 = 1
    MCS_2 = 2
    MCS_3 = 3
    MCS_4 = 4
    MCS_5 = 5
    MCS_6 = 6
    MCS_7 = 7
    MCS_8 = 8
    MCS_9 = 9
    MCS_10 = 10
    MCS_11 = 11
    NAN = 12


class TxPowerValue(JaxEnum):
    LVL_1 = 0
    LVL_2 = 1
    LVL_3 = 2
    LVL_4 = 3
    NAN = 4
