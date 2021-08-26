"""
Loss for topic model
"""

import torch


def topic_modeling_loss(inputs, topic_num, word_dists, prior_mean, prior_variance,
                        posterior_mean, posterior_variance, posterior_log_variance):
    # KL term
    # var division term
    var_division = torch.sum(posterior_variance / prior_variance, dim=1)
    # diff means term
    diff_means = prior_mean - posterior_mean
    diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)
    # logvar det division term
    logvar_det_division = prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
    # combine terms
    KL = 0.5 * (var_division + diff_term - topic_num + logvar_det_division)

    # Reconstruction term
    RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)

    loss = KL + RL

    # the losses are averaged across observations for each minibatch
    return loss.mean()
