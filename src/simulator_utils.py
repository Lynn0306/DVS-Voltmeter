#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   simulator_utils.py
@Time    :   2022/7/13 01:27
@Author  :   Songnan Lin, Ye Ma
@Contact :   songnan.lin@ntu.edu.sg, my17@tsinghua.org.cn
@Note    :   Sample events by alternating between polarity selection and timestamp sampling, in Algorithm 1, Section 4
@inproceedings{lin2022dvsvoltmeter,
  title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle={ECCV},
  year={2022}
}
'''

import torch
import numpy as np
import random
import math


def sample_non_c_zero(ep, c_in, sigma_in):
    # 取inverse gaussian的分布
    # https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    pos_c_position = c_in > 0
    neg_c_position = c_in < 0

    X = torch.empty_like(ep, dtype=ep.dtype,
                         device=ep.device)

    # same direction
    X_Pos = torch.randn(size=(torch.sum(pos_c_position),), dtype=ep.dtype, device=ep.device)
    X[pos_c_position] = X_Pos

    # opposite direction  # why?
    ep_neg = ep[neg_c_position]
    c_neg = c_in[neg_c_position]
    sigma_in_neg = sigma_in[neg_c_position]
    x_max_thres = -1 * torch.sqrt(-4 * ep_neg *
                                  c_neg / torch.pow(sigma_in_neg, 2.0))
    sample_mean = torch.zeros_like(x_max_thres, device=x_max_thres.device)
    sample_sigma = torch.ones_like(x_max_thres, device=x_max_thres.device)
    sample_inf = sample_sigma * (-1) * np.inf
    x_truncated = sample_truncated_normal(
        sample_mean, sample_sigma, sample_inf, x_max_thres)
    X[neg_c_position] = x_truncated

    mean = ep / c_in
    lambda_ig = torch.pow(ep / sigma_in, 2.0)
    scale = lambda_ig
    mu = mean / 2 / scale

    Y = mean * X * X
    Z = (4*scale*Y + Y * Y)
    X = mean + mu * (Y - torch.sqrt(Z))

    U = torch.empty_like(ep, device=ep.device)
    U.uniform_()
    out = torch.where(U > mean/(mean + X), mean * mean / X, X)
    return out


def sample_levy(c, mu=0):
    # standard brown movement without drift
    # 分布服从c=(alpha/sigma)^2 的levy分布，详情请看
    # https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    # https://en.wikipedia.org/wiki/L%C3%A9vy_distribution#Random_sample_generation
    u = torch.empty_like(c, device=c.device)
    u.uniform_()
    ev = torch.erfinv(1-u)
    out = c / torch.pow(ev, 2) + mu
    return out


def sample_IG_torch(episilon: torch.Tensor, c: torch.Tensor, sigma: torch.Tensor):
    assert isinstance(episilon, torch.Tensor)
    assert isinstance(c, torch.Tensor)
    delta_t_tensor = torch.zeros_like(episilon, device=episilon.device)

    # 当有没有漂移的时候有不同分布，有漂移是inverse gaussian，没有是Levy
    # 以下是levy代码
    zeros_mask = c == 0.0
    ep_zeros_c = episilon[zeros_mask]
    sigma_zeros_c = sigma[zeros_mask]
    scale_levy = torch.pow(ep_zeros_c / sigma_zeros_c, 2.0)
    delta_t_c_zero = sample_levy(scale_levy).reshape(-1)

    # where_non_zeros = torch.nonzero(torch.logical_not(zeros_mask))
    where_non_zeros = ~zeros_mask
    ep_non_zeros = episilon[where_non_zeros]
    c_non_zeros = c[where_non_zeros]
    sigma_non_zeros = sigma[where_non_zeros]
    delta_t_c_non_zero = sample_non_c_zero(
        ep_non_zeros, c_non_zeros, sigma_non_zeros).reshape(-1)

    delta_t_tensor[zeros_mask] = delta_t_c_zero
    delta_t_tensor[where_non_zeros] = delta_t_c_non_zero

    return delta_t_tensor


def test_scipy(n: int, epsilon: np.ndarray, c: np.ndarray):
    # pdf = invgauss.rvs(c, size=n)
    # pdf = pdf
    # geninvgauss()
    mean = epsilon / c
    scale = epsilon ** 2
    mu = mean / 2 / scale
    x = np.random.normal(size=(n,))
    u = np.random.uniform(size=(n,))

    y = mean * x * x
    x = mean + mu * (y - np.sqrt(4*scale*y + y*y))
    print(np.sum(np.isnan(x)))
    b = np.where(u > mean/(mean + x), mean*mean/x, x)
    b_w = np.where(np.logical_not(np.isnan(b)))
    b = b[b_w]
    print(b.shape)
    return b


def sample_truncated_normal(mean: torch.Tensor, scale: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[summary]
    [1]https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/21
    [2]The Truncated Normal Distribution
    Args:
        mean (torch.Tensor): [description]
        scale (torch.Tensor): [description]
        a (torch.Tensor): [description]
        b (torch.Tensor): [description]

    Returns:
        torch.Tensor: [description]
    """
    assert mean.shape == scale.shape
    assert scale.shape == a.shape
    assert a.shape == b.shape

    def normal_cdf(v, mu, sigma):
        cdf = 0.5 * (1 + torch.erf((v - mu) *
                     sigma.reciprocal() / math.sqrt(2)))
        return cdf

    uni = torch.empty_like(mean).uniform_()
    alpha_cdf = normal_cdf(a, mean, scale)
    beta_cdf = normal_cdf(b, mean, scale)
    p = alpha_cdf + (beta_cdf - alpha_cdf) * uni
    v = (2 * p - 1).clamp(-1, 1)
    out = mean + scale * math.sqrt(2) * torch.erfinv(v)
    out = torch.where(out < a, a, out)
    out = torch.where(out > b, b, out)
    return out


def gaussian_cdf(x, mu=0.0, sigma=1.0):
    fai = (torch.erf((x - mu) / sigma / math.sqrt(2)) + 1) * 0.5
    return fai


def inverse_gaussian_cdf_diff(x1, x2, lamb, mu):
    # assume x1 < x2
    k = 2 * lamb / mu

    lx1 = torch.sqrt(lamb / x1)
    x1_1 = lx1 * (x1 / mu - 1)
    x1_2 = -1 * lx1 * (x1 / mu + 1)

    lx2 = torch.sqrt(lamb / x2)
    x2_1 = lx2 * (x2 / mu - 1)
    x2_2 = -1 * lx2 * (x2 / mu + 1)

    # log1 = torch.log(gaussian_cdf(x1))
    # log2 = torch.log(gaussian_cdf(x2)) + k

    # cdf = log1 + torch.log1p(torch.exp(log2-log1))
    # cdf = torch.exp(cdf)
    part1 = gaussian_cdf(x2_1) - gaussian_cdf(x1_1)
    part2 = gaussian_cdf(x2_2) - gaussian_cdf(x1_2)
    part2_log = torch.log(part2) + k
    part2 = torch.exp(part2_log)
    cdf = part1 + part2
    cdf = torch.where(x <= 0, torch.zeros_like(x), cdf)
    return cdf


def inverse_gaussian_pdf(x, lamb, mu):
    expin = -1 * lamb * torch.pow(x - mu,  2) / 2 / torch.pow(mu, 2) / x
    k = torch.sqrt(lamb / 2 / math.pi / torch.pow(x, 3))
    return k * torch.exp(expin)


def ig_prob_a_b_NC(a, b, lamb, mu, h: int = 4):
    # using Newton-Cotes equations
    if h == 1:
        para = [0.5, 0.5]
    elif h == 2:
        para = [1.0/6, 2.0/3, 1.0/6]
    elif h == 3:
        para = [0.125, 0.375, 0.375, 0.125]
    elif h == 4:
        para = [7/90.0, 16/45, 2/15, 16/45, 7/90]
    else:
        # if h != 4:
        raise NotImplementedError('h> 4 methods not implemented.')
    a = torch.clamp_min(a, 0)
    sub = b - a
    inter_tensors = [a]
    inter_tensors.extend([a + sub * i / h for i in range(h-1)])
    inter_tensors.append(b)

    assert len(inter_tensors) == len(para) == h+1

    out = torch.zeros_like(a)
    for para_single, inter_tensor_single in zip(para, inter_tensors):
        out = out + para_single * \
            inverse_gaussian_pdf(inter_tensor_single, lamb, mu)
    out = out * sub
    return out


def event_generation(ep_on, ep_off, c, sigma, delta_vd_legacy, start_t, end_t, x=None, y=None):
    assert ep_on.shape == ep_off.shape
    assert ep_on.shape == c.shape
    assert ep_on.shape == delta_vd_legacy.shape
    delta_vd_legacy = delta_vd_legacy.double()

    if x is None:
        assert y is None  # stands for 2 dim tensors
        assert len(ep_on.shape) == 2
        h, w = ep_on.shape
        h = torch.arange(h, device=ep_on.device)
        w = torch.arange(w, device=ep_on.device)
        yy, xx = torch.meshgrid(h, w)
        x = xx.reshape(-1)
        y = yy.reshape(-1)

    if len(ep_on.shape) == 2:
        ep_on = ep_on.reshape(-1)
        ep_off = ep_off.reshape(-1)
        c = c.reshape(-1)
        delta_vd_legacy = delta_vd_legacy.reshape(-1)
        start_t = start_t.reshape(-1).to(torch.float64)
        sigma = sigma.reshape(-1)

    # first on probs
    # 判断是先触发ON/OFF事件
    # ref: 随机过程
    ep_on_real = ep_on - delta_vd_legacy  # ep to trigger positive OFF
    ep_off_real = ep_off + delta_vd_legacy   # absolute ep to trigger negative ON
    sigma_squared = torch.pow(sigma, 2.0)
    exp_2uB = torch.exp(2 * c * ep_off_real / sigma_squared)
    exp_2uA = torch.exp(-2 * c * ep_on_real / sigma_squared)
    p_first_on = (exp_2uB - 1) / (exp_2uB - exp_2uA)   # trigger OFF(A) before ON(-B)
    p_first_on = torch.where(torch.isnan(p_first_on), torch.ones_like(
        p_first_on, device=p_first_on.device), p_first_on)
    p_first_on = torch.where(c == 0, torch.ones_like(
        p_first_on, device=p_first_on.device) * 0.5, p_first_on)

    u = torch.empty_like(p_first_on, device=p_first_on.device)
    u.uniform_()
    on_mask = u <= p_first_on  # 先触发Positive (OFF)事件的mask

    ep_input = torch.where(on_mask, ep_on_real, ep_off_real).to(torch.float64)

    c_input = torch.where(on_mask, c, -1 * c)
    delta_t_step = sample_IG_torch(ep_input, c_input, sigma)  # 采样 $\Delta t$
    t_end_ideal = delta_t_step.to(torch.float64) + start_t.to(torch.float64)

    delta_t_2_ends = t_end_ideal - end_t
    delta_vd_legacy_new = delta_vd_legacy

    still_possible_events = delta_t_2_ends < 0  # 没到end time， 仍有新event可能

    # last_event_positions = torch.logical_not(still_possible_events)
    # 超过end time，计算 $\Delta V_d_legacy$
    last_event_positions = ~still_possible_events
    # ep_ori = torch.where(on_mask, ep_on, -1 * ep_off)
    sign = on_mask.to(torch.float64) * 2 - 1
    ep_signed = sign * ep_input
    delta_vd_last = ep_signed * (end_t - start_t) / delta_t_step
    delta_vd_legacy_new[last_event_positions] = delta_vd_legacy_new[last_event_positions] + \
        delta_vd_last[last_event_positions]

    # collect events
    # this_event_positions = torch.nonzero(still_possible_events)
    this_event_positions = still_possible_events
    events_t = t_end_ideal[this_event_positions]
    events_x = x[this_event_positions]
    events_y = y[this_event_positions]
    events_p = on_mask[this_event_positions]

    # if len(this_event_positions[0]) > 0:
    if torch.sum(this_event_positions) > 0:
        ep_on_next = ep_on[this_event_positions]
        ep_off_next = ep_off[this_event_positions]
        c_next = c[this_event_positions]
        delta_vd_next = torch.zeros_like(ep_on_next, device=ep_on_next.device)
        # delta_vd_next_ideal = torch.zeros_like(
        #     ep_on_next, device=ep_on_next.device)
        sigma_next = sigma[this_event_positions]
        x_next = events_x
        y_next = events_y
        t_next = events_t
        e_t, e_x, e_y, e_p, e_delta_vd = event_generation(
            ep_on_next, ep_off_next, c_next, sigma_next, delta_vd_next, t_next, end_t, x_next, y_next)

        events_t = torch.cat([events_t, e_t])
        events_x = torch.cat([events_x, e_x])
        events_y = torch.cat([events_y, e_y])
        events_p = torch.cat([events_p, e_p])

        delta_vd_legacy_new[still_possible_events] = e_delta_vd

    return events_t, events_x, events_y, events_p, delta_vd_legacy_new
