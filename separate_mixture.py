import sys
from pathlib import Path
from typing import *
import torch

DEVICE = torch.device("cpu")
SAMPLE_RATE = 22050 # < IMPORTANT: do not change
STEMS = ["speech", "non_speech"] # < IMPORTANT: do not change
# ROOT_PATH = Path("..").resolve().absolute()
CKPT_PATH =   "ckpts"
DATA_PATH = "data"

# sys.path.append(str(ROOT_PATH))
# %load_ext autoreload
# %autoreload 2

from main.module_base import Model

# Load model
model = Model.load_from_checkpoint("ckpts/run-4am/epoch=22-valid_loss=0.075.ckpt").to(DEVICE)
denoise_fn = model.model.diffusion.denoise_fn

import soundfile as sf
import torch

audio, sr = sf.read('data/dummy_speech_n_speech/mixture_1.wav')
audio = torch.from_numpy(audio.transpose(1,0).reshape(1,2,-1)).float() # seq_len, 2 -> 2, seq_len -> 1, 2, seq_len ( batch, stereo, seq_len)
print(audio.shape, sr) # If the audio's sampling rate is not 22050, you should adjust your audio file to match the target sampling rate.


from main.separation import separate_mixture
from audio_diffusion_pytorch import KarrasSchedule
# Generation hyper-parameters
s_churn = 20.0
num_steps = 150
num_resamples = 2

# Define timestep schedule
schedule = KarrasSchedule(sigma_min=1e-4, sigma_max=20.0, rho=7)(num_steps, DEVICE)

start_idx = 0
sources = audio[:,:, start_idx:start_idx + 262144].to(DEVICE)
sources = ((sources[:,0:1] + sources[:,1:2])/2).float() # Stereo to mono

separated = separate_mixture(
    mixture= sources,
    denoise_fn= denoise_fn,
    sigmas=schedule,
    noises= torch.randn(1, 2, 262144).to(DEVICE),
    s_churn=s_churn, # > 0 to add randomness
    num_resamples= num_resamples,
)
separated.shape

import numpy as np
import soundfile as sf
separated = separated.detach().cpu().numpy().squeeze(0)

for i, stem in enumerate(STEMS):
    sf.write(
        f"output/separations/mixture_sep/{stem}.wav",
        separated[i],
        22050,
        format="WAV"
    )