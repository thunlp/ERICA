import torch
from .generic_pair_loss import GenericPairLoss

class NTXentLoss(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple
        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')
            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half())
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return torch.mean(-log_exp)
        return 0

class NTXentLoss_R(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        mask1 = a1.le(15)
        a1 = torch.masked_select(a1, mask1)
        p = torch.masked_select(p, mask1)

        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple
        mask1 = a1.le(15)
        a1 = torch.masked_select(a1, mask1)

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')
            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half())
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return torch.mean(-log_exp)
        return 0