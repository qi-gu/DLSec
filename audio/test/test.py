import argparse
import posixpath

import torch
import torchaudio

from audio.EvalBox.Attack import ImperceptibleCWAttacker
from audio.Models.pytorch_model import PyTorchAudioModel
from audio.Models.utils import load_decoder, load_model
from audio.utils.misc import AITESTING_DOMAIN, download_if_needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O parameters
    parser.add_argument("--goal", type=str, default="HELLO", help="Please use uppercase")

    # plot parameters
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "librispeech_pretrained_v3.pth.tar"
    model_path = download_if_needed(posixpath.join(AITESTING_DOMAIN, "ckpts", model_name))
    model = load_model(model_path=model_path)
    decoder = load_decoder(labels=model.labels)
    audio_model = PyTorchAudioModel(model, decoder, device)

    wav_name = "7176-92135-0024.wav"
    wav_path = download_if_needed(posixpath.join(AITESTING_DOMAIN, "data", wav_name))
    sound, sample_rate = torchaudio.load(wav_path)
    sound = sound.to(device)
    goal = args.goal.upper()
    print(audio_model(sound, decode=True))
    # attacker = FGSMAttacker(model=audio_model, device=device)
    # attacker = PGDAttacker(model=audio_model, device=device)
    # attacker = CWAttacker(model=audio_model, device=device)
    if device == "cpu":
        attacker = ImperceptibleCWAttacker(model=audio_model, device=device, max_iter_2=200)
    else:
        attacker = ImperceptibleCWAttacker(model=audio_model, device=device)
    # attacker = GeneticAttacker(model=audio_model, device=device)
    adv = attacker.generate(sound, goal)
    print(audio_model(adv, decode=True))
    print((adv - sound).abs().max())
    torchaudio.save("output.wav", adv.cpu(), sample_rate=sample_rate)
