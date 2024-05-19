import warnings

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .attack import Attacker
from .utils import target_sentence_to_label


class CWAttacker(Attacker):
    def __init__(self, model, device, **kwargs):
        super(CWAttacker, self).__init__(model, device)
        self._parse_params(**kwargs)
        self.criterion = nn.CTCLoss()

    def _parse_params(self, **kwargs):
        self.eps = kwargs.get("eps", 0.1)
        self.lambd = kwargs.get("lambd", 0)
        self.iterations = kwargs.get("query_budget", None)
        if self.iterations is None:
            self.iterations = kwargs.get("iterations", 180)
        elif kwargs.get("iterations", None) is not None:
            warnings.warn("`query_budget` is set, `iterations` will be ignored", RuntimeWarning)
        self.lr = kwargs.get("learning_rate", 1e-3)
        self.decrease_factor = kwargs.get("decrease_factor", 0.8)
        self.num_iter_decrease_eps = kwargs.get("num_iter_decrease_eps", 10)
        self.early_stop = kwargs.get("early_stop", True)
        self.verbose = kwargs.get("verbose", True)

    def generate(self, sounds, targets):
        raw_targets = targets
        targets = target_sentence_to_label(targets)
        targets = targets.view(1, -1).to(self.device).detach()
        target_lengths = torch.IntTensor([targets.shape[1]]).view(1, -1)
        advs = sounds.clone().requires_grad_(True)
        eps = torch.ones((sounds.shape[0], 1)).to(self.device) * self.eps
        minx = torch.clamp(sounds - eps, min=-1)
        maxx = torch.clamp(sounds + eps, max=1)
        optimizer = torch.optim.Adam([advs], lr=self.lr)
        results = advs.clone().detach()
        with torch.backends.cudnn.flags(enabled=False):
            print_sep = "#" * 3
            for i in tqdm(
                range(self.iterations),
                dynamic_ncols=True,
                mininterval=1.0,
                disable=not self.verbose,
            ):
                optimizer.zero_grad()
                decode_out, out, output_sizes = self.model(advs, decode=True)
                decode_out = [x[0] for x in decode_out]
                out = out.transpose(0, 1).log()
                loss_CTC = self.criterion(out, targets, output_sizes, target_lengths)
                loss_norm = self.lambd * torch.mean((advs - sounds) ** 2)
                loss = loss_CTC + loss_norm
                loss.backward()
                # advs.grad.nan_to_num_(nan=0)
                advs.grad.data.masked_fill_(advs.grad.data.isnan(), 0)
                optimizer.step()
                # advs.data.clamp_(min=minx, max=maxx)
                advs.data = torch.max(torch.min(advs.data, maxx), torch.min(minx, maxx))
                if i % self.num_iter_decrease_eps == 0:
                    if self.verbose:
                        print(
                            f"{print_sep} loss: {loss.item():.5f} {print_sep}",
                            f"advs perturb: {(advs - sounds).abs().max().item():.5f} {print_sep}",
                            f"results perturb: {(results - sounds).abs().max().item():.5f} {print_sep}",
                        )
                        print(
                            f"{print_sep} decode output: {decode_out} {print_sep}",
                            f"raw targets: {raw_targets} {print_sep}",
                            f"success: {decode_out[0] == raw_targets} {print_sep}",
                        )
                    for j in range(len(decode_out)):
                        if decode_out[j] == raw_targets:
                            norm = (advs[j] - sounds[j]).abs().max()
                            if eps[j, 0] > norm:
                                eps[j, 0] = norm
                                results[j] = advs[j].clone()
                            eps[j, 0] *= self.decrease_factor
                    if self.early_stop and all(x == raw_targets for x in decode_out):
                        break
                minx = torch.clamp(sounds - eps, min=-1)
                maxx = torch.clamp(sounds + eps, max=1)
        return results
