from typing import Optional

import torch

def segment_csr(
    src: torch.Tensor, indptr: torch.Tensor, out: Optional[torch.Tensor] = ..., reduce: str = "sum"
) -> torch.Tensor: ...
