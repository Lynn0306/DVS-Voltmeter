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
import math


def sample_IG(ep, mu_in, sigma_in):
    '''
    Notes:
        Time interval follows an inverse Gaussian distribution with non-zero drift parameter mu (Eq. (13))
        See url for details:
        https://https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Sampling_from_an_inverse-Gaussian_distribution
    Input:
        :param ep: target voltage change (-ep_on or ep_off)
        :param mu_in: drift parameter of Brownian motion with drift
        :param sigma_in: scale parameter of Brownian motion with drift
    :return:
    '''

    # Sampling from IG(mean, scale)
    mean = ep / mu_in
    scale = torch.pow(ep / sigma_in, 2.0)

    # ------------------
    # Generate a random variate from a normal distribution v ~ N(0,1)
    V = torch.empty_like(ep, dtype=ep.dtype, device=ep.device)
    pos_mean_position = mean > 0
    neg_mean_position = mean < 0
    # positive mean: random sampling
    V_pos_mean = torch.randn(size=(torch.sum(pos_mean_position),), dtype=ep.dtype, device=ep.device)
    V[pos_mean_position] = V_pos_mean
    # negative mean: truncated normal to avoid invalid sqrt operation
    ep_neg_mean = ep[neg_mean_position]
    mu_neg_mean = mu_in[neg_mean_position]
    sigma_neg_mean = sigma_in[neg_mean_position]
    v_max_thres = -1 * torch.sqrt(-4 * ep_neg_mean * mu_neg_mean / torch.pow(sigma_neg_mean, 2.0))
    sample_mean = torch.zeros_like(v_max_thres, device=v_max_thres.device)
    sample_sigma = torch.ones_like(v_max_thres, device=v_max_thres.device)
    sample_inf = sample_sigma * (-1) * np.inf
    v_truncated = sample_truncated_normal(sample_mean, sample_sigma, sample_inf, v_max_thres)
    V[neg_mean_position] = v_truncated

    # ------------------
    Y = mean * V * V
    Z = (4 * scale * Y + Y * Y)
    X = mean + mean / 2 / scale * (Y - torch.sqrt(Z))  # use the relation.

    # ------------------
    # Another random uniform-distribution variate to compare.
    U = torch.empty_like(ep, device=ep.device)
    U.uniform_()
    out = torch.where(U > mean / (mean + X), mean * mean / X, X)
    return out


def sample_levy(c, mu=0):
    '''
    Notes:
        Time interval follows a L ́evy distribution when mu = 0 (Eq. (14))
        See url for details
        https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
        https://en.wikipedia.org/wiki/L%C3%A9vy_distribution#Random_sample_generation
    Input:
        :param ep: target amplitude of voltage change
        :param mu: const = 0
    :return:
        out: time interval
    '''

    u = torch.empty_like(c, device=c.device)
    u.uniform_()
    ev = torch.erfinv(1-u)
    out = c / torch.pow(ev, 2) + mu
    return out


def sample_timestamp(episilon: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    '''
    Notes:
        Time interval follows an inverse Gaussian distribution with non-zero drift parameter mu (Eq. (13));
        Time interval follows a L ́evy distribution when mu = 0 (Eq. (14))
    Input:
        :param episilon: target voltage change (-ep_on or ep_off)
        :param mu: drift parameter of Brownian motion with drift
        :param sigma: scale parameter of Brownian motion with drift
    :return:
        delta_t_tensor: time interval
    '''
    assert isinstance(episilon, torch.Tensor)
    assert isinstance(mu, torch.Tensor)
    delta_t_tensor = torch.zeros_like(episilon, device=episilon.device)

    # ------------------
    # L ́evy distribution when mu = 0 (Eq. (14))
    zero_mu_mask = mu == 0.0
    ep_zero_mu = episilon[zero_mu_mask]
    sigma_zero_mu = sigma[zero_mu_mask]
    scale_levy = torch.pow(ep_zero_mu / sigma_zero_mu, 2.0)
    delta_t_zero_mu = sample_levy(scale_levy).reshape(-1)

    # ------------------
    # Inverse Gaussian distribution when mu != 0 (Eq. (13))
    non_zero_mu_mask = ~zero_mu_mask
    ep_non_zero_mu = episilon[non_zero_mu_mask]
    mu_non_zero_mu = mu[non_zero_mu_mask]
    sigma_non_zero_mu = sigma[non_zero_mu_mask]
    delta_t_non_zero_mu = sample_IG(ep_non_zero_mu, mu_non_zero_mu, sigma_non_zero_mu).reshape(-1)

    delta_t_tensor[zero_mu_mask] = delta_t_zero_mu
    delta_t_tensor[non_zero_mu_mask] = delta_t_non_zero_mu

    return delta_t_tensor


def sample_truncated_normal(mean: torch.Tensor, scale: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[summary]
    [1]https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/21
    [2]The Truncated Normal Distribution
    Args:
        mean (torch.Tensor): Truncated normal mean
        scale (torch.Tensor): Truncated normal scale
        a (torch.Tensor): Truncated normal lower bound
        b (torch.Tensor): Truncated normal upper bound

    Returns:
        torch.Tensor: Sampled truncated normal tensor
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


def event_generation(ep_on, ep_off, mu, sigma, delta_vd_res, start_t, end_t, x=None, y=None):
    '''
    Note:
    Input:
        :param ep_on: target voltage change of an ON event
        :param ep_off: target voltage change of an OFF event
        :param mu: drift parameter of Brownian motion with drift
        :param sigma: scale parameter of Brownian motion with drift
        :param delta_vd_res: residual voltage change (delta vd) of the simulation between the last two adjacent frames
        :param start_t: timestamp of the last event
        :param end_t: timestamp of the 2nd frame
        :param x: x position
        :param y: y position
    :return:
        events_t: us
        events_x:
        events_y:
        events_p: {0, 1}
        delta_vd_res_new: updated residual voltage change (delta vd)
    '''
    assert ep_on.shape == ep_off.shape
    assert ep_on.shape == mu.shape
    assert ep_on.shape == delta_vd_res.shape
    delta_vd_res = delta_vd_res.double()

    if x is None:
        assert y is None  # stands for 2 dim tensors
        assert len(ep_on.shape) == 2
        h, w = ep_on.shape
        h = torch.arange(h, device=ep_on.device)
        w = torch.arange(w, device=ep_on.device)
        yy, xx = torch.meshgrid(h, w)
        y = yy.reshape(-1)
        x = xx.reshape(-1)

    if len(ep_on.shape) == 2:
        ep_on = ep_on.reshape(-1)
        ep_off = ep_off.reshape(-1)
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        delta_vd_res = delta_vd_res.reshape(-1)
        start_t = start_t.reshape(-1).to(torch.float64)

    # ------------------
    # Polarity selection: the probability of delta V hitting - ep_on_real before ep_off_real.
    signed_ep_on_real  = - ep_on - delta_vd_res  # Neg for ON, pos for OFF;
    signed_ep_off_real =  ep_off - delta_vd_res
    sigma_squared = torch.pow(sigma, 2.0)
    exp_2uA = torch.exp(2 * mu * signed_ep_on_real / sigma_squared)
    exp_2uB = torch.exp(2 * mu * signed_ep_off_real / sigma_squared)
    p_first_on = (exp_2uA - 1) / (exp_2uA - exp_2uB)  # Eq. (12)
    p_first_on = torch.where(torch.isnan(p_first_on), torch.ones_like(
        p_first_on, device=p_first_on.device), p_first_on)
    p_first_on = torch.where(mu == 0, torch.ones_like(
        p_first_on, device=p_first_on.device) * 0.5, p_first_on)

    u = torch.empty_like(p_first_on, device=p_first_on.device)
    u.uniform_()
    on_mask = u <= p_first_on  # Uniform sampling compared with the probability.

    # ------------------
    # Timestamp sampling
    signed_ep_input = torch.where(on_mask, signed_ep_on_real, signed_ep_off_real).to(torch.float64)
    delta_t_step = sample_timestamp(signed_ep_input, mu, sigma)  # sampled time interval for next event

    nan_mask = torch.isnan(delta_t_step)
    delta_t_step = torch.nan_to_num(delta_t_step, nan=0.0)
    t_end_ideal = delta_t_step.to(torch.float64) + start_t.to(torch.float64)

    # valid event is triggered before timestamp of 2nd frame.
    no_end_mask = t_end_ideal < end_t
    valid_events_mask = torch.logical_and(no_end_mask, torch.logical_not(nan_mask))
    events_t = t_end_ideal[valid_events_mask]
    events_x = x[valid_events_mask]
    events_y = y[valid_events_mask]
    events_p = on_mask[valid_events_mask]

    # invalid event is triggered after timestamp of 2nd frame,
    # update $\Delta V_d^{res}$ for simulation during the next two adjacent frames.
    last_event_positions = ~no_end_mask
    delta_vd_res_new = delta_vd_res
    delta_vd_last = signed_ep_input * (end_t - start_t) / delta_t_step  # proportional
    delta_vd_res_new[last_event_positions] = delta_vd_res_new[last_event_positions] + delta_vd_last[last_event_positions]

    # As for valid events, there is an opportunity to trigger extra events at their positions.
    # Repeat the event sampling until no more extra valid events generated.
    if torch.sum(no_end_mask) > 0:
        ep_on_next = ep_on[no_end_mask]
        ep_off_next = ep_off[no_end_mask]
        mu_next = mu[no_end_mask]
        sigma_next = sigma[no_end_mask]
        delta_vd_res_next = torch.zeros_like(ep_on_next, device=ep_on_next.device)
        x_next = x[no_end_mask]
        y_next = y[no_end_mask]
        t_next = t_end_ideal[no_end_mask]
        e_t, e_x, e_y, e_p, e_delta_vd = event_generation(
            ep_on_next, ep_off_next, mu_next, sigma_next, delta_vd_res_next, t_next, end_t, x_next, y_next)

        events_t = torch.cat([events_t, e_t])
        events_x = torch.cat([events_x, e_x])
        events_y = torch.cat([events_y, e_y])
        events_p = torch.cat([events_p, e_p])

        delta_vd_res_new[no_end_mask] = e_delta_vd

    return events_t, events_x, events_y, events_p, delta_vd_res_new

