from typing import List, Union

import torch
from torchtyping import TensorType

from tqdm import tqdm

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat, tint, tlong

AALETTERS = tuple("ACDEFGHIKLMNPQRSTVWY")

class DockingScorer(Proxy):
    """
    input: amino acid sequence (string)
    output: docking score
    1. use ESM3 to fold the sequence
    2. use DockString to calculate the docking score
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def setup(self, env=None):
        pass
    
    def __call__(
        self, states: Union[List[str], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        scores = []
        for state in enumerate(states): 
            seq_str = state[1]
            score = seq_str.count("A") * 0.1
            
            scores.append(score)
        scores = torch.tensor(scores, device=self.device, dtype=self.float)
        # print(f"states: {states}")
        # print(f"Docking scores: {scores}")
        # scores = torch.randn(len(states), device=self.device, dtype=self.float)
        # print(f"Docking scores: {scores}")
        return scores
    
    def _sum_scores(self, sample: list) -> int:
        pass
    
    def _is_in_vocabulary(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch"]:
        pass
    

