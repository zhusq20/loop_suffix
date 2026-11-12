"""Appendix-D native-proposal correction, independent of LAD tree scheduling."""
from dataclasses import dataclass

import torch


@dataclass
class Proposal:
    token: int
    distribution: torch.Tensor
    depth: int


def probabilities(logits, temperature=1., top_p=1.):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError('Sampling requires positive temperature and 0 < top_p <= 1')
    p = (logits.float() / temperature).softmax(-1)
    if top_p < 1:
        values, order = p.sort(descending=True)
        keep = values.cumsum(-1) - values < top_p
        values = values * keep
        p = torch.zeros_like(p).scatter(-1, order, values / values.sum())
    return p


def residual(p, q):
    r = (p - q).clamp_min(0)
    mass = r.sum()
    if not bool(mass > 0):
        return None
    return r / mass


def draw(p, generator):
    return int(torch.multinomial(p, 1, generator=generator))


def second_proposal(first, q2_raw, depth, generator):
    if bool(q2_raw[first.token] >= first.distribution[first.token]):
        return None
    q2 = residual(q2_raw, first.distribution)
    if q2 is None:
        raise ArithmeticError('Decreased first-token probability requires positive residual mass')
    return Proposal(draw(q2, generator), q2, depth)


def verify(p, proposals, generator):
    current = p
    for index, proposal in enumerate(proposals):
        q, token = proposal.distribution, proposal.token
        if not bool(q[token] > 0):
            raise ValueError('A proposal must have positive probability under its recorded distribution')
        ratio = (current[token] / q[token]).clamp(max=1.)
        if bool(torch.rand((), device=p.device, generator=generator) < ratio):
            return token, index
        current = residual(current, q)
        if current is None:
            raise ArithmeticError('Impossible rejection with no remaining target mass')
    return draw(current, generator), None
