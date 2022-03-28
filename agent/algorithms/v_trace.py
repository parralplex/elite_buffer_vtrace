"""

V-trace algorithm based on https://arxiv.org/abs/1802.01561
KL mask for gradient nullifying is from LASER algorithm https://arxiv.org/pdf/1909.11583.pdf
Additional policy-cloning and value-cloning loss functions are from CLEAR algorithm https://arxiv.org/pdf/1811.11682.pdf
"""
import torch
import torch.nn.functional as F


def v_trace(actions, behavior_logits, bootstrap_value, target_logits, target_values, not_done, rewards, behavior_values, flags):

    target_log_policy = F.log_softmax(target_logits, dim=-1)
    target_action_log_probs = target_log_policy.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    behavior_log_policy = F.log_softmax(behavior_logits, dim=-1)
    behavior_action_log_probs = behavior_log_policy.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    if flags.use_kl_mask or flags.use_policy_cloning_loss:
        beh_dist = torch.distributions.Categorical(logits=behavior_logits)
        tar_dist = torch.distributions.Categorical(logits=target_logits)
        kl_loss = torch.distributions.kl.kl_divergence(tar_dist, beh_dist)

    with torch.no_grad():
        if flags.use_kl_mask:
            kl_mask = (kl_loss.detach() < flags.kl_div_threshold)
        else:
            kl_mask = torch.ones_like(rewards)

        policy_rate = torch.exp(target_action_log_probs - behavior_action_log_probs)
        rhos = torch.clamp_max(policy_rate, max=flags.rho_const)
        coefs = torch.clamp_max(policy_rate, max=1.0)

        values_t_plus_1 = torch.cat([target_values[1:], bootstrap_value.unsqueeze(0)], dim=0)

        discounts = not_done.long() * flags.gamma

        deltas = rhos * (rewards + discounts * values_t_plus_1 - target_values)

        vs_minus_v_xs_current = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = []
        for t in reversed(range(flags.r_f_steps)):
            vs_minus_v_xs_current = deltas[t] + discounts[t] * coefs[t] * vs_minus_v_xs_current * kl_mask[t]
            vs_minus_v_xs.append(vs_minus_v_xs_current)

        vs_minus_v_xs.reverse()

        vs_minus_v_xs = torch.stack(vs_minus_v_xs)

        vs = vs_minus_v_xs + target_values

        policy_gradient_rhos = torch.clamp_max(policy_rate, max=flags.c_const)

        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat([vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0)

        policy_gradient_advantages = policy_gradient_rhos * (rewards + discounts * vs_t_plus_1 - target_values) * kl_mask

    policy = F.softmax(target_logits, dim=-1)
    log_policy = F.log_softmax(target_logits, dim=-1)
    entropy_loss = flags.entropy_loss_weight * torch.sum(policy * log_policy)

    value_loss = flags.value_loss_weight * (0.5 * torch.sum(torch.pow(((vs - target_values) * kl_mask), 2.0)))

    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(target_logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )

    cross_entropy = cross_entropy.view_as(policy_gradient_advantages)

    policy_loss = flags.policy_gradient_loss_weight * torch.sum(cross_entropy * policy_gradient_advantages)

    if flags.use_policy_cloning_loss:
        policy_loss += flags.policy_cloning_loss_weight * (kl_loss * kl_mask).sum()

    if flags.use_value_cloning_loss:
        value_loss += flags.value_cloning_loss_weight * torch.dist(behavior_values * kl_mask, target_values * kl_mask, p=2)

    return value_loss, entropy_loss, policy_loss

