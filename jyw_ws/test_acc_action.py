import torch

def accumulate_delta_chunk(
    pred_delta: torch.Tensor,  # (B, K, D)
    state_t: torch.Tensor      # (B, D)
) -> torch.Tensor:
    """
    Convert per-step delta actions to absolute joint targets.

    Returns:
        abs_action: (B, K, D)
            abs_action[:, i] = state_t + sum_{j<=i} pred_delta[:, j]
    """
    # cumulative sum along time dimension
    delta_cum = torch.cumsum(pred_delta, dim=1)   # (B, K, D)

    print(delta_cum)

    # broadcast add state_t
    abs_action = state_t.unsqueeze(1) + delta_cum

    return abs_action

def accumulate_delta_chunk2(
    pred_delta: torch.Tensor,  # (B, K, D)
) -> torch.Tensor:
    """
    Convert per-step delta actions to absolute joint targets.

    Returns:
        abs_action: (B, K, D)
            abs_action[:, i] = state_t + sum_{j<=i} pred_delta[:, j]
    """
    # cumulative sum along time dimension
    delta_cum = torch.cumsum(pred_delta, dim=1)   # (B, K, D)


    return delta_cum

import numpy as np
state=torch.from_numpy(np.array([[1]]))

pred=torch.from_numpy(np.array([[[0.1],[0.1],[0.1]]]))

print(state)
print(pred.shape)
res=accumulate_delta_chunk2(pred)
print(res,res.shape)