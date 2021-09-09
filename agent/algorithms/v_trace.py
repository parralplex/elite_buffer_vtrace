"""

V-trace algorithm based on https://arxiv.org/abs/1802.01561
"""
import torch
import torch.nn.functional as F


def v_trace(actions, beh_logits, bootstrap_value, current_logits, current_values, not_done, rewards, device, flags, batch_size):
    target_log_policy = F.log_softmax(current_logits[:-1], dim=-1)
    target_action_log_probs = target_log_policy.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    behavior_log_policy = F.log_softmax(beh_logits, dim=-1)
    behavior_action_log_probs = behavior_log_policy.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        policy_rate = torch.exp(target_action_log_probs - behavior_action_log_probs)
        rhos = torch.clamp(policy_rate, max=flags.rho_const)
        coefs = torch.clamp(policy_rate, max=flags.c_const)

        values_t_plus_1 = current_values[1:]

        discounts = not_done.long() * flags.gamma

        deltas = rhos * (rewards + discounts * values_t_plus_1 - current_values[:-1])

        vs = torch.zeros((flags.r_f_steps, batch_size)).to(device)

        vs_minus_v_xs = torch.zeros_like(bootstrap_value)
        for t in reversed(range(flags.r_f_steps)):
            vs_minus_v_xs = deltas[t] + discounts[t] * coefs[t] * vs_minus_v_xs
            vs[t] = current_values[t] + vs_minus_v_xs

        vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
        advantages_vt = rhos * (rewards + discounts * vs_t_plus_1 - current_values[:-1])

    policy = F.softmax(current_logits[:-1], dim=-1)
    log_policy = F.log_softmax(current_logits[:-1], dim=-1)
    entropy_loss = flags.entropy_loss_coef * torch.sum(policy * log_policy)

    baseline_loss = flags.baseline_loss_coef * torch.sum((vs - current_values[:-1]) ** 2)

    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(current_logits[:-1], 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages_vt)

    # cannot used "target_action_log_probs" instead of cross_entropy because some gradients created during
    # target_action_log_probs calculation cannot be replicated deterministically during backprop and therefore reproducibility cannot be assured

    policy_loss = torch.sum(cross_entropy * advantages_vt)

    return baseline_loss, entropy_loss, policy_loss
