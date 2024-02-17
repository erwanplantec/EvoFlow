import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import NamedTuple, Union
from jaxtyping import Float, Array
from flowlenia.flowlenia_params import (
    FlowLeniaParams as FLP, 
    Config as FLPConfig, 
    State as FLPState,
    beam_mutation)

class State(NamedTuple):
    A: Float[Array, "X Y C"] #Cells activations
    P: Float[Array, "X Y K"] #Embedded parameters
    F: Float[Array, "X Y"]
    fK: jax.Array             #Kernels fft

    def to_flp_state(self)->FLPState:
        return FLPState(A=self.A, P=self.P, fK=self.fK)

class Config(NamedTuple):
    flp_cfg: FLPConfig=FLPConfig()
    # --- Init ---
    n_init_species: int = 100
    init_patch_size: int = 20
    # --- Mutations ---
    mutation_rate: float = 0.001
    beam_size: int = 20
    # --- Food and decay ---
    food_birth_rate: float=.1
    food_growth_rate: float=.1
    decay_rate: float=.005
    digest_rate: float=.01
    food_to_matter_ratio: float=1.
    digesting_channel: Union[int, list[int]]=0

def food_beam(state, key, sz=5):
    loc = jr.randint(key, (2,), minval=0, maxval=state.A.shape[0]-sz)
    f = jnp.ones((sz,sz))
    F = jax.lax.dynamic_update_slice(state.F, f, loc)
    return state._replace(F=F)

class FoodFLP(eqx.Module):
    
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

    def initialize(self, key: jax.Array)->State:
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
        F = jnp.zeros((self.cfg.flp_cfg.X, self.cfg.flp_cfg.Y))
        return State(A=A, P=P, fK=s.fK, F=F)

    #-------------------------------------------------------------------

    def __call__(self, state: State, key: jax.Array)->State:
        key_step, key_mut, key_food = jr.split(key, 3)
        flp_state = state.to_flp_state()
        flp_state = self.flp(flp_state, key_step)
        state = state._replace(A=flp_state.A, P=flp_state.P)
        state = beam_mutation(state, key_mut, sz=self.cfg.beam_size, p=self.cfg.mutation_rate)
        # --- Eating and decay ---
        Adig = state.A[self.cfg.digesting_channel]
        if isinstance(self.cfg.digesting_channel, list):
            Adig = Adig.sum(-1)
        dF = jnp.clip(Adig * self.cfg.digest_rate, 0., state.F)
        dA = dF * self.cfg.food_to_matter_ratio
        F = state.F - dF
        A = state.A + dA[...,None]
        state = state._replace(A=A, F=F)
        # --- Food add ---
        state = jax.lax.cond(
            jr.uniform(key_food) < self.cfg.food_birth_rate,
            lambda s, k: food_beam(s, k),
            lambda s, k: s,
            state, key_food
        )
        return state