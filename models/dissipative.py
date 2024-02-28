import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import NamedTuple
from flowlenia.flowlenia_params import (
    FlowLeniaParams as FLP, 
    Config as FLPConfig, 
    State as FLPState,
    beam_mutation)

def death_beam(state, key, sz=20):
    A, P = state.A, state.P
    loc = jr.randint(key, (3,), minval=0, maxval=int(P.shape[0])-sz).at[-1].set(0)
    zeroA = jnp.zeros((sz,sz,A.shape[-1]))
    A = jax.lax.dynamic_update_slice(A, zeroA, loc)
    zeroP = jnp.zeros((sz,sz,P.shape[-1]))
    P = jax.lax.dynamic_update_slice(P, zeroP, loc)
    return state._replace(A=A, P=P)

def birth_beam(state, key, q, sz=20):
    kA, kP, kloc = jr.split(key, 3)
    A, P = state.A, state.P
    loc = jr.randint(kloc, (3,), minval=0, maxval=P.shape[0]/5-sz).at[-1].set(0)
    #loc = jnp.array([10, 10, 0], dtype=int)
    p = jnp.ones((sz,sz,P.shape[-1])) * jr.normal(kP, (1,1,P.shape[-1]))
    c = q / (sz**2 * 0.5)
    a = jr.uniform(kA, (sz,sz,A.shape[-1])) * c
    dA = jax.lax.dynamic_update_slice(jnp.zeros_like(A), a, loc)
    A = A + dA
    P = jax.lax.dynamic_update_slice(P, p, loc)
    return state._replace(A=A, P=P)

class Config(NamedTuple):
    flp_cfg: FLPConfig
    # --- Init ---
    n_init_species: int = 100
    init_patch_size: int = 20
    # --- Mutations ---
    mutation_rate: float = 0.001
    beam_size: int = 20
    # --- Death/Birth ---
    beam_prob: float = 0.01

class DissipativeFLP(eqx.Module):
    
    """
    """
    #-------------------------------------------------------------------
    # Parameters:
    flp: FLP
    # Statics
    cfg: Config
    #-------------------------------------------------------------------

    def __init__(self, cfg: Config, key: jax.Array):
        
        self.flp = FLP(cfg.flp_cfg, key=key)
        self.cfg = cfg

    #-------------------------------------------------------------------

    def initialize(self, key: jax.Array)->FLPState:
        s = self.flp.initialize(key)
        A, P = s.A, s.P
        key_locs, key_A, key_P = jr.split(key, 3)
        locs = jr.randint(key_locs, 
                          (self.cfg.n_init_species,3), 
                          minval=0, 
                          maxval=self.cfg.flp_cfg.X-self.cfg.init_patch_size).at[:,-1].set(0.)
        a = jr.uniform(key_A, (self.cfg.n_init_species, self.cfg.init_patch_size, self.cfg.init_patch_size, self.cfg.flp_cfg.C))
        A = jax.vmap(jax.lax.dynamic_update_slice, in_axes=(None,0,0))(A, a, locs).sum(0)
        p = jr.normal(key_P, (self.cfg.n_init_species, 1, 1, self.cfg.flp_cfg.k)) \
            * jnp.ones((self.cfg.n_init_species, self.cfg.init_patch_size, self.cfg.init_patch_size, self.cfg.flp_cfg.k))
        P = jax.vmap(jax.lax.dynamic_update_slice, in_axes=(None,0,0))(P, p, locs).sum(0)
        return s._replace(A=A, P=P)

    #-------------------------------------------------------------------

    def __call__(self, state: FLPState, key: jax.Array)->FLPState:
        assert key is not None
        key_step, key_rm, key_add = jr.split(key, 3)
        state = self.flp(state, key_step)
        state = beam_mutation(state, key, sz=self.cfg.beam_size, p=self.cfg.mutation_rate)

        def remove_and_add(state, key):
            kr, ka = jr.split(key)
            new_state = death_beam(s, kr)
            delta = (state.A - new_state.A).sum((0,1))
            state = birth_beam(new_state, ka, delta)
            return state

        # --- Remove and add ---
        k1, k2 = jr.split(key_rm)
        state = jax.lax.cond(
            jr.uniform(k1) < self.cfg.beam_prob,
            lambda s, k: remove_and_add(s, k),
            lambda s, k: s,
            state, k2
        )

        return state

if __name__ == '__main__':
    import numpy as np
    from flowlenia.utils import conn_from_matrix
    from flowlenia.flowlenia_params import Config as FLPConfig
    from flowlenia.vizutils import display_flp
    import matplotlib.pyplot as plt

    M = np.array([[5, 5, 5],
              [5, 5, 5],
              [5, 5, 5]], dtype = int)
    C = M.shape[0]
    k = int(jnp.sum(M))
    c0, c1 = conn_from_matrix(M)

    flp_cfg = FLPConfig(
        X=64,
        Y=64,
        C=C,
        k=k,
        c0=c0,
        c1=c1,
        mix_rule="stoch"
    )

    cfg = Config(
        flp_cfg = flp_cfg,
        n_init_species=12,
        mutation_rate=0.01,
        beam_prob=0.1
    )

    mdl = DissipativeFLP(cfg, jr.key(1))

    s = mdl.initialize(jr.key(2))
    def step(c, x):
        s, k = c
        k, _k = jr.split(k)
        return [mdl(s, _k), k], s
    _, S = jax.lax.scan(step, [s, jr.key(1)], None, 128)
    display_flp(S)
    m = S.A.sum((1,2,3))
    plt.plot(m); plt.show()









