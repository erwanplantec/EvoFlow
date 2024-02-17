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
    A, P = state.P, state.A
    loc = jr.randint(key, (3,), minval=0, maxval=int(P.shape[0]/5)-sz).at[-1].set(0)
    zeroA = jnp.zeros((sz,sz,A.shape[-1]))
    A = jax.lax.dynamic_update_slice(A, zeroA, loc)
    zeroP = jnp.zeros((sz,sz,P.shape[-1]))
    P = jax.lax.dynamic_update_slice(P, zeroP, loc)
    return state._replace(A=A, P=P)

def birth_beam(state, key, sz=20):
    kA, kP, kloc = jr.split(key, 3)
    A, P = state.P, state.A
    loc = jr.randint(kloc, (3,), minval=0, maxval=P.shape[0]/5-sz).at[-1].set(0)
    p = jr.normal(kP, (sz,sz,P.shape[-1]))
    a = jr.uniform(kA, (sz,sz,A.shape[-1]))
    A = jax.lax.dynamic_update_slice(A, a, loc)
    P = jax.lax.dynamic_update_slice(P, p, loc)
    return state._replace(A=A, P=A)

class Config(NamedTuple):
    flp_cfg: FLPConfig
    # --- Init ---
    n_init_species: int = 100
    init_patch_size: int = 20
    # --- Mutations ---
    mutation_rate: float = 0.001
    beam_size: int = 20
    # --- Death/Birth ---
    death_prob: float = 0.15
    birth_prob: float = 0.1

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
        # --- Remove ---
        k1, k2 = jr.split(key_rm)
        state = jax.lax.cond(
            jr.uniform(k1) < self.cfg.death_prob,
            lambda s, k: death_beam(s, k),
            lambda s, k: s,
            state, k2
        )
        # --- Add ---
        k1, k2 = jr.split(key_add)
        state = jax.lax.cond(
            jr.uniform(k1) < self.cfg.death_prob,
            lambda s, k: birth_beam(s, k),
            lambda s, k: s,
            state, k2
        )

        return state







