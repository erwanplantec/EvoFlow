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


class Config(NamedTuple):
    flp_cfg: FLPConfig
    # --- Init ---
    n_init_species: int = 100
    init_patch_size: int = 20
    # --- Mutations ---
    mutation_rate: float = 0.001
    beam_size: int = 20

class SimpleFLP(eqx.Module):
    
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
        key_step, key_mut = jr.split(key)
        state = self.flp(state, key_step)
        state = beam_mutation(state, key_mut, sz=self.cfg.beam_size, p=self.cfg.mutation_rate)
        return state