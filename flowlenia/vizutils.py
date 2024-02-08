import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import display, clear_output
from IPython.core.display import HTML
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def display_fl(states):
    ims = []
    fig, ax = plt.subplots()
    for i in range(100):
        A = states.A[i]
        C = A.shape[-1]
        if C==1:
            img = A
        if C==2:
            img=jnp.dstack([A[...,0], A[...,0], A[...,1]])
        else:
            img = A[...,:3]
        im = ax.imshow(img, animated=True)
        ims.append([im])
    _ = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()

def display_flp(states):
    ims = []
    fig, ax = plt.subplots()
    for i in range(100):
        A, P = states.A[i], states.P[i]
        im = ax.imshow(P[..., :3] * A.sum(-1, keepdims=True), animated=True)
        ims.append([im])
    _ = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    plt.show()

class VideoWriter:
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)
    
    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
              img = np.uint8(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)
    
    def close(self):
        if self.writer:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *kw):
        self.close()
    
    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))