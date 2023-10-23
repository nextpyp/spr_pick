# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import stats
import numpy as np

from torch.autograd import Variable

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
    pred (batch x c x h x w)
    gt_regr (batch x c x h x w)
    '''

    # gt = gt.unsqueeze(0)
    # pos_inds = gt.eq(0.9).float()
    # print('gt', gt.shape)
    # print('pred', pred.shape)
    pos_inds = gt.eq(1).float()
    # gt[gt == -1] = 0
    neg_inds = gt.lt(1).float()
    # gt_0 = gt.gt(-1).float()
    # print('gt', gt.shape)
    # print('gt', gt)
    # print('hello')
    # neg_inds_update = gt_0 == neg_inds

    # neg_inds_update = neg_inds_update.float()
    # print('neg_inds_update', neg_inds_update)
    # neg_inds_update = neg_inds
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    
    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

def _bce_puloss(pred, gt, tau, beta, gamma):
    true_pos_inds = gt.gt(0).float()
    # print(true_pos_inds)
    unlabeled_inds = gt.lt(0).float()
    num_pos = true_pos_inds.float().sum()
    num_unlabeld = unlabeled_inds.float().sum()
    pos_loss = torch.log(pred) * true_pos_inds
    pos_loss_tot = -(pos_loss.sum())/ num_pos
    pos_risk = (pos_loss_tot) * tau 

    neg_pos_loss = torch.log(1-pred) * true_pos_inds 
    neg_poss_loss_tot = -(neg_pos_loss.sum())/num_pos 
    neg_pos_risk = neg_poss_loss_tot
    unlabeled_neg_loss = torch.log(1-pred)*unlabeled_inds
    unlabeled_loss = -(unlabeled_neg_loss).sum()
    unlabeled_risk = unlabeled_loss / num_unlabeld
    neg_risk_total = -tau * neg_pos_risk + unlabeled_risk

    if neg_risk_total < -beta:
        return -gamma*neg_risk_total
    else:
        return pos_risk + neg_risk_total


def _pu_neg_loss_mod(pred, gt, tau, beta, gamma):
    true_pos_inds = gt.gt(0.4).float()
    # print(true_pos_inds)
    unlabeled_inds = gt.lt(0).float()
    num_pos = true_pos_inds.float().sum()
    num_unlabeld = unlabeled_inds.float().sum()
    # print('num_unlabeld', num_unlabeld)
    pos_loss = torch.pow(1 - pred, 2)*torch.log(pred) * true_pos_inds
    # print(pos_loss)
    # pos_loss = torch.log(pred) * torch.pow(1 - pred, 1) * true_pos_inds
    pos_loss_tot = -(pos_loss.sum())/ num_pos

    # print('pos_loss', pos_loss_tot)
    pos_risk = (pos_loss_tot) * tau
    neg_pos_loss = torch.pow(pred, 2) * torch.log(1-pred) * true_pos_inds
    # neg_pos_loss = torch.log(1-pred) * torch.pow(pred, 1) * true_pos_inds
    neg_pos_loss_tot = -(neg_pos_loss.sum()) / num_pos
    neg_pos_risk = neg_pos_loss_tot
    # print('neg pos risk', neg_pos_risk)
    unlabeled_neg_loss = torch.pow(pred, 2) * torch.log(1 - pred) * unlabeled_inds
    # unlabeled_neg_loss = torch.pow(pred, 1) * torch.log(1 - pred) * unlabeled_inds
    unlabeled_loss = -(unlabeled_neg_loss).sum()
    unlabeled_risk = unlabeled_loss / num_unlabeld
    # print('unlabeled_risk', unlabeled_risk)
    neg_risk_total = -tau * neg_pos_risk + unlabeled_risk
    # print('unlabeled tot', neg_risk_total)
    if neg_risk_total < -beta:
        return pos_risk
    else:
        return pos_risk + neg_risk_total

def _pu_neg_loss(pred, gt, tau, beta, gamma):
    '''
    Positive Unlabeled Focal Loss 
    Arguments:
    pred (batch x c x h x w)
    gt (batch x c x h x w)
    
    '''
    # print('gt', gt.shape)
    # gt = gt.unsqueeze(0)
    # true_pos_inds = gt.eq(1).float()
    true_pos_inds = gt.gt(0.5).float()
    other_inds = gt.lt(1).float()
    labeled_inds = gt.gt(0.5).float()
    soft_pos_inds = labeled_inds == other_inds
    soft_pos_inds = soft_pos_inds.float()
    unlabeled_inds = gt.eq(-1).float()
    # gt[gt == -1] = 0

    # num_pos = true_pos_inds.float().sum() + soft_pos_inds.float().sum()
    num_pos = true_pos_inds.float().sum()
    # print('num_pos', num_pos)
    num_unlabeld = unlabeled_inds.float().sum()
    num_soft = soft_pos_inds.float().sum()
    soft_pow_weights = torch.pow(1 - gt, 4)
    # soft_pow_weights = 1
    soft_pow_neg_weights = torch.pow(gt, 4)
    # soft_pow_neg_weights = 1
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * true_pos_inds
    soft_pos_loss = torch.log(1 - pred) * torch.pow(pred, 2) * soft_pow_weights * soft_pos_inds
    # pos_loss_tot = 0.1*(-(pos_loss.sum())/ num_pos) - 0.3*((soft_pos_loss.sum())/num_soft)
    # pos_loss_tot = -(pos_loss + soft_pos_loss).sum()
    pos_loss_tot = -(pos_loss.sum())/ num_pos - (soft_pos_loss.sum())/num_soft
    # pos_loss_tot = -(pos_loss.sum()+soft_pos_loss.sum())/(num_pos+num_soft)
    # pos_loss_tot = -(pos_loss.sum()+soft_pos_loss.sum())/num_pos
    pos_risk = (pos_loss_tot) * tau 
    # pos_risk = pos_loss_tot
    neg_pos_loss = torch.log(1-pred) * torch.pow(pred, 2) * true_pos_inds
    neg_soft_pos_loss = torch.log(pred) * torch.pow(1-pred, 2) * soft_pow_neg_weights * soft_pos_inds
    # neg_pos_loss_tot = -(neg_pos_loss + neg_soft_pos_loss).sum()
    neg_pos_loss_tot = -(neg_pos_loss.sum()) / num_pos - (neg_soft_pos_loss.sum())/num_soft
    # neg_pos_loss_tot = -(neg_pos_loss.sum()+neg_soft_pos_loss.sum())/(num_pos+num_soft)
    # neg_pos_loss_tot = -(neg_pos_loss.sum()+neg_soft_pos_loss.sum())/num_pos
    # neg_pos_loss_tot = 0.1*(-(neg_pos_loss.sum()) / num_pos) - 0.3*((neg_soft_pos_loss.sum())/num_soft)
    neg_pos_risk = neg_pos_loss_tot 
    unlabeled_neg_loss = torch.pow(pred, 2) * torch.log(1 - pred) * unlabeled_inds
    unlabeled_loss = -(unlabeled_neg_loss).sum()
    unlabeled_risk = unlabeled_loss / num_unlabeld

    neg_risk_total = -tau * neg_pos_risk + unlabeled_risk
    # neg_risk_total = -neg_pos_risk + unlabeled_risk
    # print('neg_risk_total', neg_risk_total)
    # print('pos_risk', pos_risk)
    # print('neg_pos_risk', neg_pos_risk)

    if neg_risk_total < -beta:
        return pos_risk
    else:
        return pos_risk + neg_risk_total

class PULoss(nn.Module):

    def __init__(self, tau, beta = 0, gamma = 1):
        super(PULoss, self).__init__()
        self.tau = tau 

        self.gamma = gamma 
        self.beta = beta 
        self.puloss = _pu_neg_loss
        # self.puloss = _pu_neg_loss_mod
        # self.puloss = _bce_puloss
    def forward(self, pred, gt):
        return self.puloss(pred, gt, self.tau, self.beta, self.gamma)

def _kl_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # D_KL(P || Q)
    batch, chans, height, width = p.shape
    unsummed_kl = F.kl_div(
        q.reshape(batch * chans, height * width).log(),
        p.reshape(batch * chans, height * width),
        reduction='none',
    )
    kl_values = unsummed_kl.sum(-1).view(batch, chans)
    return kl_values


def _js_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)

# TODO: add this to the main module


def _reduce_loss(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'none':
        return losses
    return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)


def js_div_loss_2d(
        input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
):
    """Calculates the Jensen-Shannon divergence loss between heatmaps.

    Args:
        input (torch.Tensor): the input tensor with shape :math:`(B, N, H, W)`.
        target (torch.Tensor): the target tensor with shape :math:`(B, N, H, W)`.
        reduction (string, optional): Specifies the reduction to apply to the
          output: `none` | `mean` | `sum`. `none`: no reduction
          will be applied, `mean`: the sum of the output will be divided by
          the number of elements in the output, `sum`: the output will be
          summed. Default: `mean`.

    Examples:
        >>> input = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(input, input)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(_js_div_2d(target, input), reduction)



def modified_pu_loss(criteria, pi, out_score, y, l2=0, slack=1, entropy_penalty=0):
    out_score = out_score.unsqueeze(1)
    # print('out_score', out_score.shape)
    # print('y', y.shape)
    y_flatten = y.view(y.shape[0], y.shape[1], y.shape[2]*y.shape[3])
    y_sum = torch.sum(y_flatten, dim=2)
    y_sum = y_sum.view(-1)
    # print('y_sum', y_sum.shape)
    select = (y_sum.data > 0)
    # print('select',select.shape)
    out_score_class = out_score[select]
    y_class = y[select]
    # print('y_class', y_class[0][0][30:35, 30:35])
    # print('out_score-class', out_score_class[0][0][30:35, 30:35])
    classifier_loss = _neg_loss(out_score_class, y_class)
    # print('classifier_loss,', classifier_loss)
    select = (y_sum.data == 0)
    N = select.sum().item()
    # print('total_select', total_select)
    out_score_un = out_score[select]
    out_score_un_flatten = out_score_un.view(out_score_un.shape[0], out_score_un.shape[1], out_score_un.shape[2]*out_score_un.shape[3])
    # p_hat = out_score_un.view(-1)
    out_score_mean = torch.mean(out_score_un_flatten, dim=2)
    # print('out_score_mean', out_score_mean)
    p_hat = out_score_mean.view(-1)
    q_mu = p_hat.sum()
    q_var = torch.sum(p_hat * (1-p_hat))
    count_vector = torch.arange(0, N+1).float()
    count_vector = count_vector.to(q_mu.device)

    q_discrete = -0.5*(q_mu - count_vector)**2/(q_var + 1e-7)
    q_discrete = F.softmax(q_discrete, dim=0)

    log_binom = stats.binom.logpmf(np.arange(0, N+1), N, pi)
    log_binom = torch.from_numpy(log_binom).float()
    if q_var.is_cuda:
        log_binom = log_binom.cuda()
    log_binom = Variable(log_binom)
    ge_penalty = -torch.sum(log_binom*q_discrete) 

    if entropy_penalty > 0:
        q_entropy = 0.5 * (torch.log(q_var) + np.log(2*np.pi) + 1)
        ge_penalty = ge_penalty + q_entropy * entropy_penalty
    # print('ge_penalty', ge_penalty)
    ge_penalty = ge_penalty / N

    loss = classifier_loss + slack * ge_penalty

    return loss


def pu_loss(criteria, pi, out_score, y, l2=0, slack=5, entropy_penalty=0):
    out_score = out_score.view(-1)

    # print('y',y)
    y = y.view(-1)
    select = (y.data >= 0)
    if select.sum().item() > 0:
        classifier_loss = criteria(out_score[select], y[select])
    else:
        classifier_loss = 0
    # print('classifier_loss,', classifier_loss)
    select = (y.data == -1)
    N = select.sum().item()
    # p_hat = torch.sigmoid(out_score[select])
    p_hat = out_score[select]
    q_mu = p_hat.sum()
    q_var = torch.sum(p_hat*(1-p_hat))
    count_vector = torch.arange(0, N+1).float()
    count_vector = count_vector.to(q_mu.device)

    q_discrete = -0.5*(q_mu - count_vector)**2/(q_var + 1e-7)
    q_discrete = F.softmax(q_discrete, dim=0)

    log_binom = stats.binom.logpmf(np.arange(0, N+1), N, pi)
    log_binom = torch.from_numpy(log_binom).float()
    if q_var.is_cuda:
        log_binom = log_binom.cuda()
    log_binom = Variable(log_binom)
    # ge_penalty = -torch.mean(log_binom*q_discrete)
    ge_penalty = -torch.sum(log_binom*q_discrete)
    # print('ge ge_penalty', ge_penalty)
    if entropy_penalty > 0:
        q_entropy = 0.5 * (torch.log(q_var) + np.log(2*np.pi) + 1)
        ge_penalty = ge_penalty + q_entropy * entropy_penalty
    loss = classifier_loss + slack*ge_penalty
    # loss = loss.mean()
    # print('classifier_loss', classifier_loss)
    # print('ge_penalty', ge_penalty)
    return loss 

class PuLoss(nn.Module):
    def __init__(self):
        super(PuLoss, self).__init__()
    def forward(self, criteria, pi, out_score, y, l2=0, slack=4.0, entropy_penalty=0.0):
        # print('slack,', slack)
        loss = pu_loss(criteria, pi, out_score, y, l2, slack, entropy_penalty)
        return loss






