import warnings

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from audio.EvalBox.Attack.attack import Attacker
from audio.EvalBox.Attack.utils import target_sentence_to_label


class PGDAttacker(Attacker):
    def __init__(self, model, device, **kwargs):
        super(PGDAttacker, self).__init__(model, device)
        self._parse_params(**kwargs)
        self.criterion = nn.CTCLoss()

    def _parse_params(self, **kwargs):
        self.eps = kwargs.get("eps", 0.025)
        self.iterations = kwargs.get("query_budget", None)
        if self.iterations is None:
            self.iterations = kwargs.get("iterations", 100)
        elif kwargs.get("iterations", None) is not None:
            warnings.warn("`query_budget` is set, `iterations` will be ignored", RuntimeWarning)
        self.alpha = kwargs.get("alpha", 1e-3)
        self.early_stop = kwargs.get("early_stop", True)
        self.verbose = kwargs.get("verbose", True)

    def generate(self, sounds, targets):
        raw_targets = targets
        targets = target_sentence_to_label(targets)
        targets = targets.view(1, -1).to(self.device).detach()
        target_lengths = torch.IntTensor([targets.shape[1]]).view(1, -1)
        advs = sounds.clone().detach().to(self.device).requires_grad_(True)
        print_sep = "#" * 3
        with torch.backends.cudnn.flags(enabled=False):
            for i in tqdm(
                range(self.iterations),
                dynamic_ncols=True,
                mininterval=1.0,
                disable=not self.verbose,
            ):
                self.model.zero_grad()
                out, output_sizes = self.model(advs)
                out = out.transpose(0, 1).log()
                loss = self.criterion(out, targets, output_sizes, target_lengths)
                loss.backward()
                # data_grad = advs.grad.data.nan_to_num(nan=0)
                data_grad = advs.grad.data.masked_fill(advs.grad.data.isnan(), 0)
                advs = advs - self.alpha * data_grad.sign()
                noise = torch.clamp(advs - sounds.data, min=-self.eps, max=self.eps)
                advs = sounds + noise
                advs = torch.clamp(advs, min=-1, max=1)
                advs = advs.detach().requires_grad_(True)
                decode_out, out, output_sizes = self.model(advs, decode=True)
                decode_out = [x[0] for x in decode_out]
                if self.verbose:
                    print(
                        f"{print_sep} loss: {loss.item():.5f} {print_sep}",
                        f"advs perturb: {(advs - sounds).abs().max().item():.5f} {print_sep}",
                    )
                    print(
                        f"{print_sep} decode output: {decode_out} {print_sep}",
                        f"raw targets: {raw_targets} {print_sep}",
                        f"success: {decode_out[0] == raw_targets} {print_sep}",
                    )
                if self.early_stop and all(x == raw_targets for x in decode_out):
                    break
        return advs
