from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong

AALETTERS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SCORES = {aa: score for aa, score in zip(AALETTERS, range(len(AALETTERS)))}

class SequenceMutation(GFlowNetEnv):
    """
    SequenceMutation Environment: Starting from a wild-type sequence and mutate it to a new one by replacing amino acids. Insertions and deletions are not allowed. The trajectory is of fixed length length_traj.

    States are represented by a list of indices corresponding to each letter, with the length of the sequence length.

    Actions are represented by a two-element tuple: (pos, token). The position is the index of the letter to be replaced, and the token is the index of the new letter to be added. The EOS action is represented by (-1, -1), indicating the end of the sequence.

    Attributes
    ----------
    seq_wt : str
        The wild-type sequence.
    
    length_traj : int
        The length of the trajectory. By default, 1.
        
    letters : tuple
        An tuple containing the letters to form words. By default, AALETTERS is used.
    """    
    def __init__(self, seq_wt: str, length_traj: int, letters: Tuple[str] = AALETTERS, **kwargs):
        self.seq_wt = list(seq_wt)     
        self.length_traj = length_traj
        self.letters = letters
        self.n_letters = len(self.letters)
        self.seq_length = len(self.seq_wt)
        # Dictionaries
        self.idx2token = {idx: token for idx, token in enumerate(self.letters)}
        self.token2idx = {token: idx for idx, token in self.idx2token.items()}               
        # Source state: list of length max_length filled with pad token
        self.source = [self.token2idx[token] for token in self.seq_wt]
        self.eos = (-1, -1)        
        super().__init__(**kwargs)
    
    def get_action_space(self) -> List[Tuple[int, int]]:
        return [(pos, token) for pos in range(self.seq_length) for token in range(self.n_letters)] + [self.eos]
    
    def get_parents(self, state: Optional[List[int]] = None, done: Optional[bool] = None, action: Optional[Tuple] = None) -> Tuple[List[int], bool]:
        """
        Determines all parents and actions that lead to state.

        The GFlowNet graph is a tree and there is only one parent per state.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        parents = []
        actions = []
        for (pos, token) in self.action_space:
            if state[pos] != token:
                new_state = copy(state)
                new_state[pos] = token
                parents.append(new_state)
                actions.append((pos, token))
        return parents, actions
    
    def step(self, action, skip_mask_check = False):
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """        
        # 終端状態ならvalid = Falseを返す
        if action == self.eos:
            return self.state, action, False
        # 終端状態でない場合、actionを実行し、doneをTrueにする
        else:
            pos, token = action
            self.state[pos] = token
            self.n_actions += 1
            self.done = True
        return self.state, action, True
    
    def step_backwards(self, action, skip_mask_check = False):
        return super().step_backwards(action, skip_mask_check)
    
    def states2proxy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as list of strings
        [seq_mutated1, seq_mutated2, ...]
        """
        
        seq_strings = []
        for state in states:
            seq_string = self.state2readable(state)
            seq_strings.append(seq_string)
        return seq_strings        
        
    
    def states2policy(
            self, states: Union[List, TensorType["batch", "state_dim"]]
        ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)
        # if (states < 0).any():
        #     states[states < 0] = 0
        return (
            F.one_hot(states, self.n_letters * self.seq_length + 1)
            .reshape(states.shape[0], -1)
            .to(dtype=self.float)
        )        
    
    def state2readable(self, state: List) -> str:
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        state = self._get_state(state)
        return "".join([self.idx2token[idx] for idx in state])
    
    def readable2state(self, readable: str) -> List:
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        if readable == "":
            return self.source
        return [self.token2idx[token] for token in list(readable)]
    
    def get_random_terminating_states(self, n_states, unique = True, max_attempts = 100000):
        return super().get_random_terminating_states(n_states, unique, max_attempts)