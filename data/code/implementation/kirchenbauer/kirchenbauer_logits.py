import torch
from transformers import LogitsProcessor

from data.code.implementation.kirchenbauer.kirchenbauer_base import KirchenbauerBase


class KirchenbauerLogitsProcessor(KirchenbauerBase, LogitsProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: list) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores)
        for batch_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[batch_idx][greenlist_token_ids[batch_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        scores = torch.nn.functional.softmax(scores)
        return scores

    def _tip_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor) -> torch.Tensor:
        scores[~greenlist_mask] = 0
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for batch_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[batch_idx])
            batched_greenlist_ids[batch_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
        scores = self._bias_greenlist_logits(scores, green_tokens_mask, greenlist_bias=self.delta)
        return scores

    def set_attributes(self, gamma: float = None, delta: float = None):
        if gamma:
            self.set_gamma(gamma)
        if delta:
            self.set_delta(delta)
