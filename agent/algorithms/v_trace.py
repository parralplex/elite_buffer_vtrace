"""

V-trace algorithm based on https://arxiv.org/abs/1802.01561
"""
import torch
import torch.nn.functional as F


def v_trace(actions, behavior_logits, bootstrap_value, target_logits, target_values, not_done, rewards, flags):

    target_log_policy = F.log_softmax(target_logits, dim=-1)
    target_action_log_probs = target_log_policy.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    # TODO pricalculate behavior_action_log_probs in workers
    behavior_log_policy = F.log_softmax(behavior_logits, dim=-1)
    behavior_action_log_probs = behavior_log_policy.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        policy_rate = torch.exp(target_action_log_probs - behavior_action_log_probs)
        rhos = torch.clamp_max(policy_rate, max=flags.rho_const)
        coefs = torch.clamp(policy_rate, max=1.0)

        values_t_plus_1 = torch.cat([target_values[1:], bootstrap_value.unsqueeze(0)], dim=0)

        discounts = not_done.long() * flags.gamma

        deltas = rhos * (rewards + discounts * values_t_plus_1 - target_values)

        vs_minus_v_xs_current = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = []
        for t in reversed(range(flags.r_f_steps)):
            vs_minus_v_xs_current = deltas[t] + discounts[t] * coefs[t] * vs_minus_v_xs_current
            vs_minus_v_xs.append(vs_minus_v_xs_current)

        vs_minus_v_xs.reverse()
        vs_minus_v_xs = torch.stack(vs_minus_v_xs)

        vs = vs_minus_v_xs + target_values

        policy_gradient_rhos = torch.clamp_max(policy_rate, max=flags.c_const)

        vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)

        policy_gradient_advantages = policy_gradient_rhos * (rewards + discounts * vs_t_plus_1 - target_values)

    policy = F.softmax(target_logits, dim=-1)
    log_policy = F.log_softmax(target_logits, dim=-1)
    entropy_loss = flags.entropy_loss_coef * torch.sum(policy * log_policy)

    baseline_loss = flags.baseline_loss_coef * (0.5 * torch.sum(torch.pow(vs - target_values, 2.0)))

    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(target_logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(policy_gradient_advantages)

    # cannot use "target_action_log_probs" instead of cross_entropy because some gradients created during
    # target_action_log_probs calculation cannot be replicated deterministically during backprop and therefore reproducibility cannot be assured

    policy_loss = torch.sum(cross_entropy * policy_gradient_advantages)

    return baseline_loss, entropy_loss, policy_loss
