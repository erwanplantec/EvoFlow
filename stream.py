from twitchstream.outputvideo import TwitchBufferedOutputStream
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import numpy as np
from jaxtyping import PyTree

class TwitchStreamerSimulator:

	#-------------------------------------------------------------------

	def __init__(
		self,
		model: PyTree,
		stream_key: str,
		width: int=256,
		height: int=256,
		fps=30.,
		verbose: bool=True):

		self.mdl = model
		self.stream_key = stream_key
		self.width = width
		self.height = height
		self.fps = fps
		self.verbose = verbose

	#-------------------------------------------------------------------

	def simulate_and_stream(self, key):

		key, init_key = jr.split(key)
		s = self.mdl.initialize(init_key)
		locs = jnp.arange(20) + (cfg.X//2-10)
		A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(init_key, (20, 20, 1)))
		s = s._replace(A=A)

		with TwitchBufferedOutputStream(twitch_stream_key=self.stream_key, 
			width=self.width, height=self.height, fps=self.fps, 
			verbose=self.verbose) as videostream:
			while True:
				if videostream.get_video_frame_buffer_state() < 30:
					key, k_ = jr.split(key)
					s = self.mdl(s, k_)
					im = jnp.ones_like(s.A) * s.A
					videostream.send_video_frame(np.array(im))




if __name__ == '__main__':
	from flowlenia.flowlenia import FlowLenia, Config
	from flowlenia.utils import conn_from_matrix
	# --- mdl ---
	cfg = Config(X=64, Y=64, C=3, k=9)
	M = np.array([[2, 1, 0],
				  [0, 2, 1],
				  [1, 0, 2]])
	c0, c1 = conn_from_matrix(M)
	cfg = cfg._replace(c0=c0, c1=c1)
	fl = FlowLenia(cfg, key=jr.key(101))

	stream_key = "live_1033351530_Ev8lE3jCu6b58aqQdbo6897Mhv60de"

	streamer = TwitchStreamerSimulator(fl, stream_key=stream_key, width=cfg.X, height=cfg.Y)

	streamer.simulate_and_stream(jr.key(1))