from typing import Optional, Tuple

import torch

def scatter_max(
    src: torch.Tensor, index: torch.Tensor, dim: Optional[int] = ...
) -> Tuple[torch.Tensor, torch.Tensor]: ...
