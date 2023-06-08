# https://github.com/HanGuo97/soft-Q-learning-for-text-generation/blob/main/sql/losses.py

import warnings
import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
from typing import List, Tuple, Dict, Any, Optional, Union

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

# https://texar-pytorch.readthedocs.io/en/latest/_modules/texar/torch/utils/utils.html?highlight=sequence_mask#
def sequence_mask(lengths: Union[torch.LongTensor, List[int]],
                  max_len: Optional[int] = None,
                  dtype: Optional[torch.dtype] = None,
                  device: Optional[torch.device] = None) -> torch.ByteTensor:
    r"""Return a mask tensor representing the first N positions of each cell.

    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with

    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```

    Examples:

    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]

    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```

    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask

def mask_and_reduce(sequence, sequence_length, 
        average_across_timesteps=False, 
        sum_over_timesteps=True):
    assert average_across_timesteps is not sum_over_timesteps
    mask = sequence_mask(sequence_length)
    if average_across_timesteps:
        res = torch.masked_select(sequence, mask > 0).mean()
    else:
        res = mask * sequence
        res = res.sum(dim=1).mean()
    return res

# https://texar-pytorch.readthedocs.io/en/latest/_modules/texar/torch/losses/entropy.html?highlight=_get_entropy
def _get_entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.

    Args:
        logits: Unscaled log probabilities.

    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, -1)
    return entropy

# https://github.com/HanGuo97/soft-Q-learning-for-text-generation/blob/main/sql/utils.py
def add_prefix_to_dict_keys_inplace(
        d: Dict[str, Any],
        prefix: str,
        keys_to_exclude: Optional[List[str]] = None,
) -> None:

    # https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
    keys = list(d.keys())
    for key in keys:
        if keys_to_exclude is not None and key in keys_to_exclude:
            continue

        new_key = f"{prefix}{key}"
        d[new_key] = d.pop(key)

# https://github.com/HanGuo97/soft-Q-learning-for-text-generation/blob/main/sql/misc_utils.py
def unionize_dicts(dicts: List[Dict]) -> Dict:
    union_dict: Dict = {}
    for d in dicts:
        for k, v in d.items():
            if k in union_dict.keys():
                raise KeyError
            union_dict[k] = v

    return union_dict

def gather_2d_on_last_dim(
        tensor: FloatTensor,
        index: LongTensor,
        shape: torch.Size
) -> FloatTensor:
    """Simplified version of `tf.gather_nd` in PyTorch"""
    flattened_tensor = tensor.reshape(-1, tensor.shape[-1])
    flattened_index = index.reshape(-1)
    flattened_gathered_tensor = flattened_tensor[
        torch.arange(flattened_index.shape[0]),
        flattened_index]
    return flattened_gathered_tensor.reshape(shape)


def soft_q_loss_with_sparse_rewards(
        implementation: str,
        logits: FloatTensor,
        logits_: FloatTensor,
        logits_pi: Optional[FloatTensor],
        actions: LongTensor,
        sampled_actions: Optional[LongTensor],
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[FloatTensor, Dict[str, Any]]:
    """Soft Q Learning Loss Functions with Sparse Rewards

    Arguments:
        implementation: string, which loss function to use
        logits:          [batch_size, sequence_length, vocab_size]
        logits_:         [batch_size, sequence_length, vocab_size]
        logits_pi:       [batch_size, sequence_length, vocab_size]
        actions:         [batch_size, sequence_length]
        rewards:         [batch_size]
        sequence_length: [batch_size]
    """
    if implementation not in ["v0", "v1", "v2", "v3", "v2_v2r", "v3_v3r", "v2_v2r_v3_v3r", "sac"]:
        raise ValueError

    if implementation not in ["sac"]:
        if logits_pi is not None or sampled_actions is not None:
            raise ValueError("`logits_pi` should be None.")

    if implementation in ["sac"]:
        if logits_pi is None or sampled_actions is None:
            raise ValueError("`logits_pi` should not be None.")

    if not torch.is_tensor(rewards):
        raise TypeError

    if rewards.ndim != 1 or logits.shape[0] != rewards.shape[0]:
        raise ValueError

    if implementation == "v0":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_0

    if implementation == "v1":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_1

    if implementation == "v2":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_2

    if implementation == "v3":
        _sql_loss_func = soft_q_loss_with_sparse_rewards_3

    if implementation == "v2_v2r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_2_2_reversed,
            coefficient=coefficient,
            margin_constant=margin_constant,
            margin_coefficient=margin_coefficient)

    if implementation == "v3_v3r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_3_3_reversed,
            coefficient=coefficient)

    if implementation == "v2_v2r_v3_v3r":
        _sql_loss_func = partial(
            soft_q_loss_with_sparse_rewards_2_2_reversed_3_3_reversed,
            coefficient=coefficient)

    if implementation == "sac":
        def _sql_loss_func(
                logits: FloatTensor,
                logits_: FloatTensor,
                actions: LongTensor,
                rewards: FloatTensor,
                sequence_length: LongTensor,
        ) -> Tuple[FloatTensor, Dict[str, Any]]:
            if logits_pi is None:
                raise ValueError

            if logits.shape != logits_pi.shape:
                raise ValueError(
                    f"`logits.shape` = {logits.shape}, but "
                    f"`logits_pi.shape` = {logits_pi.shape}")

            if coefficient is None:
                raise ValueError

            return soft_actor_critic_loss(
                logits_pi=logits_pi,  # closure
                logits_Q=logits,
                logits_Q_=logits_,
                actions=actions,
                sampled_actions=sampled_actions,    # closure
                rewards=rewards,
                sequence_length=sequence_length,
                coefficient=coefficient,  # closure
            )

    if logits.shape != logits_.shape:
        raise ValueError(
            f"`logits.shape` = {logits.shape}, but "
            f"`logits_.shape` = {logits_.shape}")

    raw_losses, quantities_to_log = _sql_loss_func(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length
    )

    loss = mask_and_reduce(
        sequence=raw_losses,
        sequence_length=sequence_length)
    loss_log = {
        "loss": loss,
        "sequence_length": sequence_length.float().mean(),
        "loss-normalized": mask_and_reduce(
            sequence=raw_losses,
            sequence_length=sequence_length,
            average_across_timesteps=True,
            sum_over_timesteps=False),
    }

    for key, value in quantities_to_log.items():
        masked_mean, masked_min, masked_max = get_masked_mean_min_max(
            value, lengths=sequence_length)
        loss_log[f"{key}/min"] = masked_min
        loss_log[f"{key}/max"] = masked_max
        loss_log[f"{key}/mean"] = masked_mean

    return loss, loss_log


def soft_q_loss_with_sparse_rewards_1(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    Q = gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    # use `V` from the target if available
    V_ = logits_.logsumexp(dim=-1)

    # Build the target `= V_t+1 + r`
    # where we assume the rewards to be sparse
    # i.e., only comes at the final step
    Q_ = torch.zeros_like(Q)
    Q_[:, :-1] = V_[:, 1:]
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards

    raw_losses = F.mse_loss(Q, Q_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": logits.logsumexp(dim=-1),
        "Q_": Q_,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        _recover_mle: bool = False,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    Q = gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V = logits.logsumexp(dim=-1)
    A = Q - V

    # Target outputs
    Q_ = torch.zeros_like(Q)
    A_ = torch.zeros_like(Q)
    V_ = logits_.logsumexp(dim=-1)
    Q_[:, :-1] = V_[:, 1:]
    A_[:, :-1] = V_[:, 1:] - V_[:, :-1]
    # Terminal V-target is the last V-target before
    # the episode ends, thus depends on `sequence_length`
    terminal_V_ = V_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1]
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards
    A_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards - terminal_V_

    if _recover_mle is True:
        warnings.warn("Recover-MLE Mode")
        A_ = A.detach() + 1

    raw_losses = F.mse_loss(A, A_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "Q_": Q_,
        "V_": V_,
        "A_": A_,
        "H": _get_entropy(logits),
        "H_": _get_entropy(logits_),
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_3(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        freeze_future_steps: bool = False,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    Q = gather_2d_on_last_dim(
        tensor=logits,
        index=actions,
        shape=actions.shape)
    V = logits.logsumexp(dim=-1)
    A = Q - V

    # Target outputs
    V_ = logits_.logsumexp(dim=-1)

    A2 = masked_reverse_cumsum(
        A,
        lengths=sequence_length,
        dim=-1)

    if freeze_future_steps is True:
        # This line of code essentially
        # decompose `A` (with gradient)
        # and cumsum of future `A`
        # (without gradient)
        A2 = (A2 - A).detach() + A

    raw_losses = F.mse_loss(
        A2, rewards.view(-1, 1) - V_,
        reduction="none")

    quantities_to_log = {
        "Q": Q,
        "V": V,
        "A": A,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2_2_reversed(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: Optional[float] = None,
        margin_constant: Optional[float] = None,
        margin_coefficient: Optional[float] = None,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    raw_losses_2, quantities_to_log_2 = soft_q_loss_with_sparse_rewards_2(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length)

    add_prefix_to_dict_keys_inplace(
        quantities_to_log_2, prefix="0/")

    if coefficient is not None:
        raw_losses_2_r, quantities_to_log_2_r = soft_q_loss_with_sparse_rewards_2(
            logits=logits_,
            logits_=logits,
            actions=actions,
            rewards=rewards,
            sequence_length=sequence_length)

        raw_losses = (
            coefficient * raw_losses_2 +
            (1 - coefficient) * raw_losses_2_r)

        add_prefix_to_dict_keys_inplace(
            quantities_to_log_2_r, prefix="1/")

        quantities_to_log = unionize_dicts([
            quantities_to_log_2,
            quantities_to_log_2_r,
        ])

    else:
        raw_losses = raw_losses_2
        quantities_to_log = quantities_to_log_2

    if margin_constant is not None and margin_coefficient is not None:
        raw_losses_margin, quantities_to_log_margin = large_margin_classification_loss(
            logits=logits,
            expert_actions=actions,
            margin_constant=margin_constant)

        raw_losses = raw_losses + margin_coefficient * raw_losses_margin
        add_prefix_to_dict_keys_inplace(
            quantities_to_log_margin, prefix="margin/")
        quantities_to_log = unionize_dicts([
            quantities_to_log,
            quantities_to_log_margin,
        ])

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_3_3_reversed(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    raw_losses_3, quantities_to_log_3 = soft_q_loss_with_sparse_rewards_3(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length)

    add_prefix_to_dict_keys_inplace(
        quantities_to_log_3, prefix="0/")

    if coefficient is not None:
        raw_losses_3_r, quantities_to_log_3_r = soft_q_loss_with_sparse_rewards_3(
            logits=logits_,
            logits_=logits,
            actions=actions,
            rewards=rewards,
            sequence_length=sequence_length)

        raw_losses = (
            coefficient * raw_losses_3 +
            (1 - coefficient) * raw_losses_3_r)

        add_prefix_to_dict_keys_inplace(
            quantities_to_log_3_r, prefix="1/")

        quantities_to_log = unionize_dicts([
            quantities_to_log_3,
            quantities_to_log_3_r,
        ])
    else:
        raw_losses = raw_losses_3
        quantities_to_log = quantities_to_log_3

    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_2_2_reversed_3_3_reversed(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: Optional[float] = None,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    raw_losses_2, quantities_to_log_2 = soft_q_loss_with_sparse_rewards_2_2_reversed(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length,
        coefficient=coefficient)

    raw_losses_3, quantities_to_log_3 = soft_q_loss_with_sparse_rewards_3_3_reversed(
        logits=logits,
        logits_=logits_,
        actions=actions,
        rewards=rewards,
        sequence_length=sequence_length,
        coefficient=coefficient)

    raw_losses = (raw_losses_2 + raw_losses_3) / 2

    add_prefix_to_dict_keys_inplace(
        quantities_to_log_2, prefix="v2/")
    add_prefix_to_dict_keys_inplace(
        quantities_to_log_3, prefix="v3/")
    quantities_to_log = unionize_dicts([
        quantities_to_log_2,
        quantities_to_log_3,
    ])
    return raw_losses, quantities_to_log


def soft_q_loss_with_sparse_rewards_0(
        logits: FloatTensor,
        logits_: FloatTensor,
        actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
) -> Tuple[FloatTensor, Dict[str, Any]]:
    V = logits.logsumexp(dim=-1)
    V_ = logits_.logsumexp(dim=-1)
    raw_losses = F.mse_loss(V, V_, reduction="none")
    quantities_to_log = {
        "V": V,
        "V_": V_,
    }

    return raw_losses, quantities_to_log


def soft_actor_critic_loss(
        logits_pi: FloatTensor,
        logits_Q: FloatTensor,
        logits_Q_: FloatTensor,
        actions: LongTensor,
        sampled_actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
        coefficient: float = 0.5,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    # critic should be freezed for computing the actor loss.
    actor_loss, actor_quantities_to_log = soft_actor_loss(
        logits_pi=logits_pi,
        logits_Q=logits_Q.detach())

    # actor should be freezed for computing the critic loss.
    critic_loss, critic_quantities_to_log = soft_critic_loss_with_sparse_rewards(
        logits_pi=logits_pi.detach(),
        logits_Q=logits_Q,
        logits_Q_=logits_Q_,
        actions=actions,
        sampled_actions=sampled_actions,
        rewards=rewards,
        sequence_length=sequence_length)

    raw_losses = coefficient * actor_loss + (1 - coefficient) * critic_loss

    add_prefix_to_dict_keys_inplace(
        actor_quantities_to_log, prefix="actor/")
    add_prefix_to_dict_keys_inplace(
        critic_quantities_to_log, prefix="critic/")
    quantities_to_log = unionize_dicts([
        actor_quantities_to_log,
        critic_quantities_to_log,
    ])
    return raw_losses, quantities_to_log


def soft_actor_loss(
        logits_pi: FloatTensor,
        logits_Q: FloatTensor,
) -> Tuple[FloatTensor, Dict[str, Any]]:
    raw_losses = (
        F.kl_div(
            # Since we are doing the KL in the opposite direction
            # we need to swap the order of the inputs and the targets.
            F.log_softmax(logits_Q, dim=-1),
            F.log_softmax(logits_pi, dim=-1),
            log_target=True,
            reduction="none")
        # We need to sum the last dimension
        # to get proper KL divergence
        .sum(dim=-1))

    quantities_to_log = {
        "loss": raw_losses,
    }
    return raw_losses, quantities_to_log


def soft_critic_loss_with_sparse_rewards(
        logits_pi: FloatTensor,
        logits_Q: FloatTensor,
        logits_Q_: FloatTensor,
        actions: LongTensor,
        sampled_actions: LongTensor,
        rewards: FloatTensor,
        sequence_length: LongTensor,
) -> Tuple[FloatTensor, Dict[str, Any]]:

    # In Soft Actor Critic, `(s, a, s', r)` should come from the replay buffer
    # but the next actions `a'` have to be sampled fresh from the policy.
    # In this function, we use `actions` to refer to `a` and
    # `sampled_actions` to refer to the next actions `a'`.
    Q = gather_2d_on_last_dim(
        tensor=logits_Q,
        index=actions,
        shape=actions.shape)
    # Q(s) = r + E_a'[ Q(s', a') + \log\pi(a'|s')], where a'~\pi(|s')
    #
    # since rewards are sparse, we have
    # Q(s) = E_a'[ Q(s', a') + \log\pi(a'|s')] for non-terminal state
    # Q(s) = r for terminal state
    #
    # since V(s) = E_a [Q(s, a) - \log\pi(a|s)], where a~\pi(|s), we have
    # Q(s) = V(s') for non-terminal state
    # Q(s) = r for terminal state
    V_ = gather_2d_on_last_dim(
        tensor=logits_Q_ - F.log_softmax(logits_pi, dim=-1),
        index=sampled_actions,
        shape=sampled_actions.shape)

    # Build the target `= V_t+1 + r`
    # where we assume the rewards to be sparse
    # i.e., only comes at the final step
    Q_ = torch.zeros_like(Q)
    Q_[:, :-1] = V_[:, 1:]
    Q_[
        torch.arange(sequence_length.shape[0]),
        sequence_length - 1] = rewards

    raw_losses = F.mse_loss(Q, Q_, reduction="none")
    quantities_to_log = {
        "Q": Q,
        "Q_": Q_,
        "V_": V_,
        "loss": raw_losses,
    }

    return raw_losses, quantities_to_log


def masked_reverse_cumsum(
        X: FloatTensor,
        lengths: LongTensor,
        dim: int
) -> FloatTensor:
    masked_X = X * sequence_mask(
        lengths,
        max_len=X.shape[1])

    return (masked_X
            .flip(dims=[dim])
            .cumsum(dim=dim)
            .flip(dims=[dim]))


def get_masked_mean_min_max(
        X: FloatTensor,
        lengths: LongTensor
) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
    if X.ndim != 2 and lengths.ndim != 1:
        raise ValueError

    if X.shape[0] != lengths.shape[0]:
        raise ValueError

    mask = get_lengths_mask(
        X=X,
        lengths=lengths)

    masked_min = X.masked_fill(~mask, np.inf).min(dim=1)
    masked_max = X.masked_fill(~mask, -np.inf).max(dim=1)
    masked_mean = mask_and_reduce(
        sequence=X,
        sequence_length=lengths,
        average_across_timesteps=True,
        sum_over_timesteps=False)

    return (masked_mean,
            masked_min.values.mean(),
            masked_max.values.mean())


def get_lengths_mask(
        X: FloatTensor,
        lengths: LongTensor
) -> FloatTensor:

    if any([
        X.ndim != 2,
        lengths.ndim != 1,
        X.shape[0] != lengths.shape[0]],
    ):
        raise ValueError

    # [batch_size, 1]
    expanded_lengths = lengths.unsqueeze(dim=1)
    return (
        torch.arange(X.shape[1], device=X.device)
        .expand(X.shape) < expanded_lengths)


def large_margin_classification_loss(
        logits: FloatTensor,
        expert_actions: LongTensor,
        margin_constant: float,
) -> Tuple[FloatTensor, Dict[str, Any]]:
    """Deep Q-learning from Demonstrations

    Arguments:
        logits: [batch_size, sequence_length, vocab_size]
        expert_actions: [batch_size, sequence_length]
    """
    # [0, 0, 0, ..., 1, 1, 1, ..., N, N, N, ...]
    batch_indices = (
        torch
        .arange(expert_actions.shape[0])
        .repeat_interleave(expert_actions.shape[1], dim=0))

    # [0, 1, 2, ..., 0, 1, 2, ..., 0, 1, 2, ...]
    sequence_indices = (
        torch
        .arange(expert_actions.shape[1])
        .repeat(expert_actions.shape[0]))

    # indices for the expert actions
    indices = (
        batch_indices,
        sequence_indices,
        expert_actions.flatten())

    # get the margin, and mask margins of expert actions
    margin = margin_constant * torch.ones_like(logits)
    margin[indices] = 0

    # [batch_size, sequence_length]
    raw_losses = (
        (logits + margin).max(dim=-1).values -
        logits[indices].view(expert_actions.shape)
    )

    quantities_to_log = {
        "loss": raw_losses,
    }

    return raw_losses, quantities_to_log
