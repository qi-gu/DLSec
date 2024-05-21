#!/usr/bin/python3
# -*- coding: utf-8 -*-

import itertools
import posixpath
import re
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from audio.EvalBox.Attack import CWAttacker, FGSMAttacker, GeneticAttacker, ImperceptibleCWAttacker, PGDAttacker
from audio.Models.pytorch_model import PyTorchAudioModel
from audio.Models.utils import load_decoder, load_model
from audio.utils.misc import AITESTING_DOMAIN, download_if_needed

_MODULE_DIR = Path(__file__).resolve().parent   # E:\Projects\Pycharm\DLSec\audio

_DATA_DIR = _MODULE_DIR / "data"
_DATA_DIR.mkdir(exist_ok=True)

_BUILTIN_MODELS = [
    "librispeech_pretrained_v3",
    "an4_pretrained_v3",
    "ted_pretrained_v3",
]

_BUILTIN_AUDIO_DIR = _MODULE_DIR / "Datasets"
_BUILTIN_AUDIO_FILES = sorted(_BUILTIN_AUDIO_DIR.glob("*.wav"))

_REMOTE_AUDIO_FILES = [
    "librispeech_ft_clean.tar.gz",
    "an4.tar.gz",
    "action_test.zip",
]

_AUDIO_FILE_SIZES = {
    "builtin": len(_BUILTIN_AUDIO_FILES),
    "librispeech_ft_clean.tar.gz": 114,
    "an4.tar.gz": 1078,
    "action_test.zip": 2,
}

_BUILTIN_RECIPES = {
    "fgsm": FGSMAttacker,
    "pgd": PGDAttacker,
    "genetic": GeneticAttacker,
    "cw": CWAttacker,
    "icw": ImperceptibleCWAttacker,
}

_DEFAULT_RECIPES = ["cw", "pgd"]

_DEFAULT_ROBUST_THR = 0.6

from AudioConfig import config

def audio_test(config):
    res = {}
    config_bak = {k: v if v is not None else "Default" for k, v in config.items()}
    # get attack recipes
    recipes = config["recipes"].split(",")
    input_dir = config.get("input_dir", _BUILTIN_AUDIO_DIR)
    if input_dir in _REMOTE_AUDIO_FILES:
        config_bak["num_examples"] = _AUDIO_FILE_SIZES[input_dir] * len(recipes)
    elif input_dir == _BUILTIN_AUDIO_DIR:
        config_bak["num_examples"] = _AUDIO_FILE_SIZES["builtin"] * len(recipes)
    else:
        config_bak["num_examples"] = "Custom"

    goal = config["goal"].upper()

    recipe_cls = []
    for recipe in recipes:
        recipe_cls.append(_BUILTIN_RECIPES[recipe.lower()])

    model = config["model"]
    if model in _BUILTIN_MODELS:
        model_path = download_if_needed(posixpath.join(AITESTING_DOMAIN, "ckpts", model + ".pth.tar"))
        model_path = Path(model_path).resolve()
    else:
        # 展示仅支持自带的模型
        return {'error': '暂不支持自定义模型'}
        model_path = Path(model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file `{str(model_path)}` not found")

    # input_dir = config["input_dir"] or _REMOTE_AUDIO_FILES[0]
    input_dir = config.get("input_dir", _BUILTIN_AUDIO_DIR)
    if input_dir in _REMOTE_AUDIO_FILES:
        input_dir = download_if_needed(posixpath.join(AITESTING_DOMAIN, "data", input_dir), _DATA_DIR, extract=True)
        input_dir = Path(input_dir).resolve()
        input_is_builtin = False
    elif Path(input_dir).resolve() == _BUILTIN_AUDIO_DIR:
        input_dir = Path(input_dir).resolve()
        input_is_builtin = True
    else:
        input_dir = Path(input_dir).resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory `{str(input_dir)}` not found")
        input_is_builtin = False

    output_dir = config.get("output_dir", None)
    if output_dir is None:
        if input_is_builtin:
            output_dir = _DATA_DIR / f"{input_dir.stem}-AdvGen"
        else:
            output_dir = input_dir.parent / f"{input_dir.stem}-AdvGen"
        if output_dir.exists():
            tmp = [
                item
                for item in output_dir.parent.glob(f"{output_dir.stem}-*")
                if re.match(f"^{output_dir.stem}\\-\\d+$", item.stem)
            ]
            if len(tmp) == 0:
                num = 1
            else:
                num = max(int(f.stem.split("-")[-1]) for f in tmp) + 1
            output_dir = output_dir.parent / f"{output_dir.stem}-{num}"
    else:
        if Path(output_dir).parent == Path("."):
            if input_is_builtin:
                output_dir = _DATA_DIR / output_dir
            else:
                output_dir = input_dir.parent / output_dir
        else:
            output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    early_stop = config.get("early_stop", True)
    verbose = config.get("verbose", True)
    query_budget = config.get("query_budget", None)

    # load model
    model = load_model(model_path=model_path).to(device)
    decoder = load_decoder(labels=model.labels)
    audio_model = PyTorchAudioModel(model, decoder, device)

    input_files = sorted(itertools.chain(input_dir.glob("*.wav"), input_dir.glob("*.flac")))
    testing_results = {
        "success": [],
        "perturbation": [],
    }

    for r_cls in recipe_cls:
        with tqdm(
            input_files,
            dynamic_ncols=True,
            mininterval=1.0,
            desc=r_cls.__name__.replace("Attacker", ""),
        ) as pbar:
            for input_file_path in pbar:
                try:
                    # load input audio file
                    sound, sample_rate = torchaudio.load(input_file_path)
                    sound = sound.to(device)
                    recipe_args = dict(
                        model=audio_model,
                        device=device,
                        query_budget=query_budget,
                        early_stop=early_stop,
                        verbose=verbose,
                    )
                    if device == "cpu":
                        recipe_args.update({"max_iter_2": 200})

                    attacker = r_cls(**recipe_args)
                    adv = attacker.generate(sound, goal)
                    decoded_adv = audio_model(adv, decode=True)[0][0][0]

                    testing_results["success"].append(decoded_adv == goal)
                    testing_results["perturbation"].append((adv - sound).abs().max().item())

                    output_file_path = output_dir / f"{input_file_path.stem}-{int(time.time())}.wav"

                    # save to output file
                    torchaudio.save(str(output_file_path), adv.detach().cpu(), sample_rate=sample_rate)

                    pbar.set_postfix_str(
                        "Accumulated [Success / Total: "
                        f"""{sum(testing_results["success"])} / {len(testing_results["success"])}]"""
                    )
                except KeyboardInterrupt:
                    if len(testing_results["success"]) >= 10:
                        print("\n\nTesting terminated by user before completion\n\n")
                        break
                    else:
                        print("\n\nTesting cancelled by user\n\n")
                        return

    summary_rows = [[k, v] for k, v in config_bak.items()]
    log_summary_rows(summary_rows, "Testing Args")

    success_rate = np.mean(testing_results["success"]).item()
    if success_rate >= config_bak.get("robust_threshold", _DEFAULT_ROBUST_THR):
        conclusion = f"The model \042{config_bak['model']}\042 is NOT robust"
    else:
        conclusion = f"The model \042{config_bak['model']}\042 is robust"

    summary_rows = [
        [
            "Average Perturbation",
            f"""{np.mean(testing_results["perturbation"]).item():.4f}""",
        ],
        ["Number of successful attacks", f"""{sum(testing_results["success"])}"""],
        [
            "Number of failed attacks",
            f"""{len(testing_results["success"]) - sum(testing_results["success"])}""",
        ],
        ["Adversarial Attack Success Rate", f"{100 * success_rate:.2f}%"],
    ]
    # 将summary_rows嵌入到res中，作为一个字典，summary_rows[i][0]作为key，summary_rows[i][1]作为value
    for i in range(len(summary_rows)):
        if summary_rows[i][0] != "":
            res[summary_rows[i][0]] = summary_rows[i][1]

    summary_rows.append([conclusion, ""])
    log_summary_rows(summary_rows, "Testing Results")

    return res


def log_summary_rows(rows, title, align_center=False):
    width, fillchar = 90, "#"
    title = title.center(len(title) + 10, " ")
    title = title.center(width, fillchar)
    msg = "\n" + title + "\n\n"
    if len(rows) == 0:
        print(msg)
        return
    max_len = max([len(row[0]) for row in rows if row[1] != ""])
    if align_center:
        rows = [[row[0].rjust(max_len), row[1]] for row in rows]
    else:
        rows = [[row[0].ljust(max_len), row[1]] for row in rows]
    msg += "\n".join([f"{row[0]}  {row[1]}" for row in rows]) + "\n\n" + fillchar * width
    print(msg)


if __name__ == "__main__":
    audio_test(config)
