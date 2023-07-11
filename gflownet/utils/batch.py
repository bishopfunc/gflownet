from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import (
    concat_items,
    copy,
    extend,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tlong,
)


class Batch:
    """
    Class to handle GFlowNet batches.

    Important note: one env should correspond to only one trajectory, all env_id should
    be unique.

    Note: self.state_indices start by index 1 to indicate that index 0 would correspond
    to the source state but the latter is not stored in the batch for each trajectory.
    This implies that one has to be careful when indexing the list of batch_indices in
    self.trajectories by using self.state_indices. For example, the batch index of
    state state_idx of trajectory traj_idx is self.trajectories[traj_idx][state_idx-1]
    (not self.trajectories[traj_idx][state_idx]).
    """

    def __init__(
        self,
        env: Optional[GFlowNetEnv] = None,
        device: Union[str, torch.device] = "cpu",
        float_type: Union[int, torch.dtype] = 32,
    ):
        """
        env : GFlowNetEnv
            An instance of the environment that will be used to form the batch.

        device : str or torch.device
            torch.device or string indicating the device to use ("cpu" or "cuda")

        float_type : torch.dtype or int
            One of float torch.dtype or an int indicating the float precision (16, 32
            or 64).

        """
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_type)
        # Generic environment, properties and dictionary of state and forward mask of
        # source (as tensor)
        if env is not None:
            self.set_env(env)
        else:
            self.env = None
            self.source = None
            self.conditional = None
            self.continuous = None
        # Initialize batch size 0
        self.size = 0
        # Initialize empty batch variables
        self.envs = OrderedDict()
        self.trajectories = OrderedDict()
        self.traj_indices = []
        # TODO: state_indices is currently unused, it is redundant and inconsistent
        # between forward and backward trajectories. We may want to remove it.
        self.state_indices = []
        self.states = []
        self.actions = []
        self.done = []
        self.masks_invalid_actions_forward = []
        self.masks_invalid_actions_backward = []
        self.parents = []
        self.parents_all = []
        self.parents_actions_all = []
        self.n_actions = []
        self.states_policy = None
        self.parents_policy = None
        # Flags for available items
        self.parents_available = False
        self.parents_all_available = False
        self.masks_forward_available = False
        self.masks_backward_available = False
        self.rewards_available = False

    def __len__(self):
        return self.size

    def batch_idx_to_traj_state_idx(batch_idx: int):
        traj_idx = self.traj_indices[batch_idx]
        state_idx = self.state_indices[batch_idx]
        return traj_idx, state_id

    def traj_idx_to_batch_indices(traj_idx: int):
        batch_indices = self.trajectories[traj_idx]
        return batch_indices

    def traj_state_idx_to_batch_idx(traj_idx: int, state_idx: int):
        batch_idx = self.trajectories[traj_idx][state_idx]
        return batch_idx

    def idx2state_idx(self, idx: int):
        return self.trajectories[self.traj_indices[idx]].index(idx)

    def set_env(self, env: GFlowNetEnv):
        """
        Sets the generic environment passed as an argument an initializes the
        environment-dependent properties.
        """
        self.env = env.copy().reset()
        self.source = {
            "state": self.env.source,
            "mask_forward": tbool(
                self.env.get_mask_invalid_actions_forward(), device=self.device
            ),
        }
        self.conditional = self.env.conditional
        self.continuous = self.env.continuous

    def add_to_batch(
        self,
        envs: List[GFlowNetEnv],
        actions: List[Tuple],
        valids: List[bool],
        masks_invalid_actions: Optional[List[List[bool]]] = None,
        backward: Optional[bool] = False,
        train: Optional[bool] = True,
    ):
        """
        Adds information from a list of environments and actions to the batch after
        performing steps in the envs. If train is False, only the variables of
        terminating states are stored.

        Args
        ----
        envs : list
            A list of environments (GFlowNetEnv).

        actions : list
            A list of actions attempted or performed on the envs.

        valids : list
            A list of boolean values indicated whether the actions were valid.

        masks_invalid_actions_forward : list
            A list of masks indicating, among all the actions that could have been
            selected for all environments, which ones were invalid. Optional, will be
            computed if not provided.

        backward : bool
            A boolean value indicating whether the action was sampled backward (False
            by dfefault). If True, the behavior is slightly different so as to match
            what is stored in forward sampling:
                - If it is the first state in the trajectory (action from a done
                  state/env), then done is stored as True, instead of taking env.done
                  which will be False after having performed the step.
                - If it is not the first state in the trajectory, the stored state will
                  be the previous one in the trajectory, to match the state-action
                  stored in forward sampling and the convention that the source state
                  is not stored, but the terminating state is repeated with action eos.

        train : bool
            A boolean value indicating whether the data to add to the batch will be used
            for training. Optional, default is True.
        """
        # TODO: do we need this?
        if self.continuous is None:
            self.continuous = envs[0].continuous

        # Sample masks of invalid actions if required and none are provided
        if masks_invalid_actions is None:
            if train:
                if backward:
                    masks_invalid_actions = [
                        env.get_mask_invalid_actions_backward() for env in envs
                    ]
                else:
                    masks_invalid_actions = [
                        env.get_mask_invalid_actions_forward() for env in envs
                    ]
            else:
                masks_invalid_actions = [None] * len(envs)

        # Add data samples to the batch
        for sample_data in zip(envs, actions, valids, masks_invalid_actions):
            env, action, valid, mask = sample_data
            if train is False and env.done is False:
                continue
            if not valid:
                continue
            # Add env to dictionary
            if env.id not in self.envs:
                self.envs.update({env.id: env})
            # Add batch index to trajectory
            if env.id not in self.trajectories:
                self.trajectories.update({env.id: [len(self)]})
            else:
                if backward:
                    self.trajectories[env.id].insert(0, len(self))
                else:
                    self.trajectories[env.id].append(len(self))
            # Add trajectory index and state index
            self.traj_indices.append(env.id)
            self.state_indices.append(env.n_actions)
            # Add states, actions, done and masks
            if backward:
                self.parents.append(copy(env.state))
                if backward and len(self.trajectories[env.id]) == 1:
                    self.states.append(copy(env.state))
                    self.done.append(True)
                    self.masks_invalid_actions_backward.append(mask)
                else:
                    self.states.append(copy(self.parents[self.trajectories[env.id][1]]))
                    self.done.append(env.done)
            else:
                self.states.append(copy(env.state))
                self.done.append(env.done)
                self.masks_invalid_actions_forward.append(mask)
            self.actions.append(action)
            # Increment size of batch
            self.size += 1

    def get_n_trajectories(self) -> int:
        """
        Returns the number of trajectories in the batch.

        Returns
        -------
        The number of trajectories in the batch (int).
        """
        return len(self.trajectories)

    def get_trajectory_indices(self) -> TensorType["n_states", int]:
        """
        Returns the trajectory index of all elements in the batch as a long int torch
        tensor.

        Returns
        -------
        traj_indices : torch.tensor
            self.traj_indices as a long int torch tensor.
        """
        return tlong(self.traj_indices, device=self.device)

    def get_state_indices(self) -> TensorType["n_states", int]:
        """
        Returns the state index of all elements in the batch as a long int torch
        tensor.

        Returns
        -------
        state_indices : torch.tensor
            self.state_indices as a long int torch tensor.
        """
        return tlong(self.state_indices, device=self.device)

    def get_states(
        self,
        policy: Optional[bool] = False,
        proxy: Optional[bool] = False,
        force_recompute: Optional[bool] = False,
    ) -> Union[TensorType["n_states", "..."], npt.NDArray[np.float32], List]:
        """
        Returns all the states in the batch.

        The states are returned in "policy format" if policy is True, in "proxy format"
        if proxy is True and otherwise they are returned in "GFlowNet" format by
        default. An error is raised if both policy and proxy are True.

        Args
        ----
        policy : bool
            If True, the policy format of the states is returned and self.states_policy
            is updated if not available yet or if force_recompute is True.

        proxy : bool
            If True, the proxy format of the states is returned. States in proxy format
            are not stored.

        force_recompute : bool
            If True, the policy states are recomputed even if they are available.
            Ignored if policy is False.

        Returns
        -------
        self.states or self.states_policy or self.states2proxy(self.states) : list or
        torch.tensor or ndarray
            The set of all states in the batch.
        """
        if policy is True and proxy is True:
            raise ValueError(
                "Ambiguous request! Only one of policy or proxy can be True."
            )
        if policy is True:
            if self.states_policy is None or force_recompute is True:
                self.states_policy = self.states2policy()
            return self.states_policy
        if proxy is True:
            return self.states2proxy()
        return self.states

    def states2policy(
        self,
        states: Optional[Union[List, TensorType["n_states", "..."]]] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> TensorType["n_states", "state_policy_dims"]:
        """
        Converts states from a list of states in GFlowNet format to a tensor of states
        in policy format.

        Args
        ----
        states: list or torch.tensor
            States in GFlowNet format.

        traj_indices: list or torch.tensor
            Ids indicating which env corresponds to each state in states. It is only
            used if the environments are conditional to call state2policy from the
            right environment. Ignored if self.conditional is False.

        Returns
        -------
        states: torch.tensor
            States in policy format.
        """
        # If traj_indices is not None and self.conditional is True, then both states
        # and traj_indices must be the same type and have the same length.
        if traj_indices is not None and self.conditional is True:
            assert type(states) == type(traj_indices)
            assert len(states) == len(traj_indices)
        if states is None:
            states = self.states
            traj_indices = self.traj_indices
        # TODO: will env.policy_input_dim be the same for all envs if conditional?
        if self.conditional:
            states_policy = torch.zeros(
                (len(states), self.env.policy_input_dim),
                device=self.device,
                dtype=self.float,
            )
            traj_indices_torch = tlong(traj_indices, device=self.device)
            for traj_idx in self.trajectories:
                if traj_idx not in traj_indices:
                    continue
                states_policy[traj_indices_torch == traj_idx] = self.envs[
                    traj_idx
                ].statebatchpolicy(
                    self.get_states_of_trajectory(traj_idx, states, traj_indices)
                )
            return states_policy
        # TODO: do we need tfloat or is done in env.statebatch2policy?
        return tfloat(
            self.env.statebatch2policy(states),
            device=self.device,
            float_type=self.float,
        )

    def states2proxy(
        self,
        states: Optional[Union[List, TensorType["n_states", "..."]]] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> Union[
        TensorType["n_states", "state_proxy_dims"], npt.NDArray[np.float32], List
    ]:
        """
        Converts states from a list of states in GFlowNet format to a tensor of states
        in proxy format. Note that the implementatiuon of this method differs from
        Batch.states2policy() because the latter always returns torch.tensors. The
        output of the present method can also be numpy arrays or Python lists,
        depending on the proxy.

        Args
        ----
        states: list or torch.tensor
            States in GFlowNet format.

        traj_indices: list or torch.tensor
            Ids indicating which env corresponds to each state in states. It is only
            used if the environments are conditional to call state2proxy from the right
            environment. Ignored if self.conditional is False.

        Returns
        -------
        states: torch.tensor or ndarray or list
            States in proxy format.
        """
        # If traj_indices is not None and self.conditional is True, then both states
        # and traj_indices must be the same type and have the same length.
        if traj_indices is not None and self.conditional is True:
            assert type(states) == type(traj_indices)
            assert len(states) == len(traj_indices)
        if states is None:
            states = self.states
            traj_indices = self.traj_indices
        if self.conditional:
            states_proxy = []
            index = torch.arange(len(states), device=self.device)
            perm_index = []
            # TODO: rethink this
            for traj_idx in self.trajectories:
                if traj_idx not in traj_indices:
                    continue
                states_proxy.append(
                    self.envs[traj_idx].statebatch2proxy(
                        self.get_states_of_trajectory(traj_idx, states, traj_indices)
                    )
                )
                perm_index.append(index[env_ids == env_id])
            perm_index = torch.cat(perm_index)
            # Reverse permutation to make it index the states_proxy array
            index[perm_index] = index.clone()
            states_proxy = concat_items(states_proxy, index)
            return states_proxy
        return self.env.statebatch2proxy(states)

    def get_actions(self) -> TensorType["n_states, action_dim"]:
        """
        Returns the actions in the batch as a float tensor.
        """
        return tfloat(self.actions, float_type=self.float, device=self.device)

    def get_done(self) -> TensorType["n_states"]:
        """
        Returns the list of done flags as a boolean tensor.
        """
        return tbool(self.done, device=self.device)

    def get_parents(
        self, policy: Optional[bool] = False, force_recompute: Optional[bool] = False
    ) -> TensorType["n_states", "..."]:
        """
        Returns the parent (single parent for each state) of all states in the batch.
        The parents are computed, obtaining all necessary components, if they are not
        readily available. Missing components and newly computed components are added
        to the batch (self.component is set).

        The parents are returned in "policy format" if policy is True, otherwise they
        are returned in "GFlowNet" format (default).

        Args
        ----
        policy : bool
            If True, the policy format of parents is returned. Otherwise, the GFlowNet
            format is returned.

        force_recompute : bool
            If True, the parents are recomputed even if they are available.

        Returns
        -------
        self.parents or self.parents_policy : torch.tensor
            The parent of all states in the batch.
        """
        if self.parents_available is False or force_recompute is True:
            self._compute_parents()
        if policy:
            return self.parents_policy
        else:
            return self.parents

    def _compute_parents(self):
        """
        Obtains the parent (single parent for each state) of all states in the batch.
        The parents are computed, obtaining all necessary components, if they are not
        readily available. Missing components and newly computed components are added
        to the batch (self.component is set). The following components are obtained:

        - self.parents: the parent of each state in the batch. It will be the same type
          as self.states (list of lists or tensor)
            Length: n_states
            Shape: [n_states, state_dims]
        - self.parents_policy: the parent of each state in the batch in policy format.
            Shape: [n_states, state_policy_dims]

        self.parents_policy is stored as a torch tensor and self.parents_available is
        set to True.
        """
        self.states_policy = self.get_states(policy=True)
        self.parents_policy = torch.zeros_like(self.states_policy)
        self.parents = []
        indices = []
        # Iterate over the trajectories to obtain the parents from the states
        for traj_idx, batch_indices in self.trajectories.items():
            # parent is source
            self.parents.append(self.envs[traj_idx].source)
            self.parents_policy[batch_indices[0]] = tfloat(
                self.envs[traj_idx].state2policy(self.envs[traj_idx].source),
                device=self.device,
                float_type=self.float,
            )
            # parent is not source
            # TODO: check if tensor and sort without iter
            self.parents.extend([self.states[idx] for idx in batch_indices[:-1]])
            self.parents_policy[batch_indices[1:]] = self.states_policy[
                batch_indices[:-1]
            ]
            indices.extend(batch_indices)
        # Sort parents list in the same order as states
        # TODO: check if tensor and sort without iter
        self.parents = [self.parents[indices.index(idx)] for idx in range(len(self))]

    def get_parents_all(
        self, policy: bool = False, force_recompute: bool = False
    ) -> Tuple[
        Union[List, TensorType["n_parents", "..."]],
        TensorType["n_parents", "..."],
        TensorType["n_parents"],
    ]:
        """
        Returns the whole set of parents, their corresponding actions and indices of
        all states in the batch. If the parents are not available
        (self.parents_all_available is False) or if force_recompute is True, then
        self._compute_parents_all() is called to compute the required components.

        The parents are returned in "policy format" if policy is True, otherwise they
        are returned in "GFlowNet" format (default).

        Args
        ----
        policy : bool
            If True, the policy format of parents is returned. Otherwise, the GFlowNet
            format is returned.

        force_recompute : bool
            If True, the parents are recomputed even if they are available.

        Returns
        -------
        self.parents_all or self.parents_all_policy : list or torch.tensor
            The whole set of parents of all states in the batch.

        self.parents_actions_all : torch.tensor
            The actions corresponding to each parent in self.parents_all or
            self.parents_all_policy, linking them to the corresponding state in the
            trajectory.

        self.parents_all_indices : torch.tensor
            The state index corresponding to each parent in self.parents_all or
            self.parents_all_policy, linking them to the corresponding state in the
            batch.
        """
        if self.continuous:
            raise Exception("get_parents() is ill-defined for continuous environments!")
        if self.parents_all_available is False or force_recompute is True:
            self._compute_parents_all()
        if policy:
            return (
                self.parents_all_policy,
                self.parents_actions_all,
                self.parents_all_indices,
            )
        else:
            return self.parents_all, self.parents_actions_all, self.parents_all_indices

    def _compute_parents_all(self):
        """
        Obtains the whole set of parents all states in the batch. The parents are
        computed via env.get_parents(). The following components are obtained:

        - self.parents_all: all the parents of all states in the batch. It will be the
          same type as self.states (list of lists or tensor)
            Length: n_parents
            Shape: [n_parents, state_dims]
        - self.parents_actions_all: the actions corresponding to the transition from
          each parent in self.parents_all to its corresponding state in the batch.
            Shape: [n_parents, action_dim]
        - self.parents_all_indices: the indices corresponding to the state in the batch
          of which each parent in self.parents_all is a parent.
            Shape: [n_parents]
        - self.parents_all_policy: self.parents_all in policy format.
            Shape: [n_parents, state_policy_dims]

        All the above components are stored as torch tensors and
        self.parents_all_available is set to True.
        """
        # Iterate over the trajectories to obtain all parents
        self.parents_all = []
        self.parents_actions_all = []
        self.parents_all_indices = []
        self.parents_all_policy = []
        for idx, traj_idx in enumerate(self.traj_indices):
            state = self.states[idx]
            done = self.done[idx]
            action = self.actions[idx]
            parents, parents_a = self.envs[traj_idx].get_parents(
                state=state,
                done=done,
                action=action,
            )
            assert (
                action in parents_a
            ), f"""
            Sampled action is not in the list of valid actions from parents.
            \nState:\n{state}\nAction:\n{action}
            """
            self.parents_all.extend(parents)
            self.parents_actions_all.extend(parents_a)
            self.parents_all_indices.extend([idx] * len(parents))
            self.parents_all_policy.append(
                tfloat(
                    self.envs[traj_idx].statebatch2policy(parents),
                    device=self.device,
                    float_type=self.float,
                )
            )
        # Convert to tensors
        self.parents_actions_all = tfloat(
            self.parents_actions_all,
            device=self.device,
            float_type=self.float,
        )
        self.parents_all_indices = tlong(
            self.parents_all_indices,
            device=self.device,
        )
        self.parents_all_policy = torch.cat(self.parents_all_policy)
        self.parents_all_available = True

    # TODO: handle mix of backward and forward trajectories
    # TODO: opportunity to improve efficiency by caching.
    def get_masks_forward(
        self,
        of_parents: bool = False,
        force_recompute: bool = False,
    ) -> TensorType["n_states", "action_space_dim"]:
        """
        Returns the forward mask of invalid actions of all states in the batch or of
        their parent in the trajectory if of_parents is True. The masks are computed
        via self._compute_masks_forward if they are not available or if force_recompute
        is True.

        Args
        ----
        of_parents : bool
            If True, the returned masks will correspond to the parents of the states,
            instead of to the states (default).

        force_recompute : bool
            If True, the masks are recomputed even if they are available.

        Returns
        -------
        self.masks_invalid_actions_forward : torch.tensor
            The forward mask of all states in the batch.
        """
        # TODO: check availability element-wise
        if self.masks_forward_available is False or force_recompute is True:
            self._compute_masks_forward()
        if of_parents:
            trajectories_parents = {
                traj_idx: [-1] + batch_indices[:-1]
                for traj_idx, batch_indices in self.trajectories.items()
            }
            parents_indices = tlong(
                [
                    trajectories_parents[traj_idx][
                        self.trajectories[traj_idx].index(idx)
                    ]
                    for idx, traj_idx in enumerate(self.traj_indices)
                ],
                device=self.device,
            )
            masks_invalid_actions_forward_parents = torch.zeros_like(
                self.masks_invalid_actions_forward
            )
            masks_invalid_actions_forward_parents[parents_indices == -1] = self.source[
                "mask_forward"
            ]
            masks_invalid_actions_forward_parents[
                parents_indices != -1
            ] = self.masks_invalid_actions_forward[
                parents_indices[parents_indices != -1]
            ]
            return masks_invalid_actions_forward_parents
        return self.masks_invalid_actions_forward

    def _compute_masks_forward(self):
        """
        Computes the forward mask of invalid actions of all states in the batch, by
        calling env.get_mask_invalid_actions_forward().
        self.masks_invalid_actions_forward is stored as a torch tensor and
        self.masks_forward_available is set to True.
        """
        # Iterate over the trajectories to compute all forward masks
        self.masks_invalid_actions_forward = []
        for idx, traj_idx in enumerate(self.traj_indices):
            state = self.states[idx]
            done = self.done[idx]
            action = self.actions[idx]
            self.masks_invalid_actions_forward.append(
                self.envs[traj_idx].get_mask_invalid_actions_forward(state, done)
            )
        # Make tensor
        self.masks_invalid_actions_forward = tbool(
            self.masks_invalid_actions_forward, device=self.device
        )
        self.masks_forward_available = True

    # TODO: handle mix of backward and forward trajectories
    # TODO: opportunity to improve efficiency by caching. Note that
    # env.get_masks_invalid_actions_backward() may be expensive because it calls
    # env.get_parents().
    def get_masks_backward(
        self,
        force_recompute: bool = False,
    ) -> TensorType["n_states", "action_space_dim"]:
        """
        Computes (and returns) the backward mask of invalid actions of all states in the
        batch, by calling env.get_mask_invalid_actions_backward().

        Args
        ----
        force_recompute : bool
            If True, the masks are recomputed even if they are available.

        Returns
        -------
        self.masks_invalid_actions_backward : torch.tensor
            The backward mask of all states in the batch.
        """
        if self.masks_backward_available is True and force_recompute is False:
            return tbool(self.masks_invalid_actions_backward, device=self.device)
        # Iterate over the trajectories to compute all backward masks
        self.masks_invalid_actions_backward = []
        for idx, traj_idx in enumerate(self.traj_indices):
            state = self.states[idx]
            done = self.done[idx]
            # TODO: if we pass parents_all_actions to get_mask... then we avoid calling
            # get_parents again.
            self.masks_invalid_actions_backward.append(
                self.envs[traj_idx].get_mask_invalid_actions_backward(state, done)
            )
        # Make tensor
        self.masks_invalid_actions_backward = tbool(
            self.masks_invalid_actions_backward, device=self.device
        )
        self.masks_backward_available = True
        return self.masks_invalid_actions_backward

    def get_rewards(
        self, force_recompute: Optional[bool] = False
    ) -> TensorType["n_states"]:
        """
        Returns the rewards of all states in the batch (including not done).

        Args
        ----
        force_recompute : bool
            If True, the rewards are recomputed even if they are available.
        """
        if self.rewards_available is False or force_recompute is True:
            self._compute_rewards()
        return self.rewards

    def _compute_rewards(self):
        """
        Computes rewards for all self.states by first converting the states into proxy
        format.

        Returns
        -------
        rewards: torch.tensor
            Tensor of rewards.
        """
        states_proxy_done = self.get_terminating_states(proxy=True)
        self.rewards = torch.zeros(len(self), dtype=self.float, device=self.device)
        done = self.get_done()
        if len(done) > 0:
            self.rewards[done] = self.env.proxy2reward(
                self.env.proxy(states_proxy_done)
            )
        self.rewards_available = True

    def get_terminating_states(
        self,
        sort_by: str = "insertion",
        policy: Optional[bool] = False,
        proxy: Optional[bool] = False,
    ) -> Union[TensorType["n_trajectories", "..."], npt.NDArray[np.float32], List]:
        """
        Returns the terminating states in the batch, that is all states with done =
        True. The states will be returned in either GFlowNet format (default), policy
        (policy = True) or proxy (proxy = True) format. If both policy and proxy are
        True, it raises an error due to the ambiguity. The returned states may be
        sorted by order of insertion (sort_by = "insert[ion]", default) or by
        trajectory index (sort_by = "traj[ectory]".

        Args
        ----
        sort_by : str
            Indicates how to sort the output:
                - insert[ion]: sort by order of insertion (states of trajectories that
                  reached the terminating state first come first)
                - traj[ectory]: sort by trajectory index (the order in the ordered
                  dict self.trajectories)

        policy : bool
            If True, the policy format of the states is returned.

        proxy : bool
            If True, the proxy format of the states is returned.
        """
        if sort_by == "insert" or sort_by == "insertion":
            indices = np.arange(len(self))
        elif sort_by == "traj" or sort_by == "trajectory":
            indices = np.argsort(self.traj_indices)
        else:
            raise ValueError("sort_by must be either insert[ion] or traj[ectory]")
        if policy is True and proxy is True:
            raise ValueError(
                "Ambiguous request! Only one of policy or proxy can be True."
            )
        traj_indices = None
        if torch.is_tensor(self.states):
            indices = tlong(indices, device=self.device)
            done = self.get_done()[indices]
            states_term = self.states[indices][done, :]
            if self.conditional and (policy is True or proxy is True):
                traj_indices = tlong(self.traj_indices, device=self.device)[indices][
                    done
                ]
                assert len(traj_indices) == len(torch.unique(traj_indices))
        elif isinstance(self.states, list):
            states_term = [self.states[idx] for idx in indices if self.done[idx]]
            if self.conditional and (policy is True or proxy is True):
                done = np.array(self.done, dtype=bool)[indices]
                traj_indices = np.array(self.traj_indices)[indices][done]
                assert len(traj_indices) == len(np.unique(traj_indices))
        else:
            raise NotImplementedError("self.states can only be list or torch.tensor")
        if policy is True:
            return self.states2policy(states_term, traj_indices)
        elif proxy is True:
            return self.states2proxy(states_term, traj_indices)
        else:
            return states_term

    def get_terminating_rewards(
        self,
        sort_by: str = "insertion",
        force_recompute: Optional[bool] = False,
    ) -> TensorType["n_trajectories"]:
        """
        Returns the reward of the terminating states in the batch, that is all states
        with done = True. The returned rewards may be sorted by order of insertion
        (sort_by = "insert[ion]", default) or by trajectory index (sort_by =
        "traj[ectory]".

        Args
        ----
        sort_by : str
            Indicates how to sort the output:
                - insert[ion]: sort by order of insertion (rewards of trajectories that
                  reached the terminating state first come first)
                - traj[ectory]: sort by trajectory index (the order in the ordered
                  dict self.trajectories)

        force_recompute : bool
            If True, the rewards are recomputed even if they are available.
        """
        if sort_by == "insert" or sort_by == "insertion":
            indices = np.arange(len(self))
        elif sort_by == "traj" or sort_by == "trajectory":
            indices = np.argsort(self.traj_indices)
        else:
            raise ValueError("sort_by must be either insert[ion] or traj[ectory]")
        if self.rewards_available is False or force_recompute is True:
            self._compute_rewards()
        done = self.get_done()[indices]
        return self.rewards[indices][done]

    def get_actions_trajectories(self) -> List[List[Tuple]]:
        actions_trajectories = []
        for batch_indices in self.trajectories.values():
            actions_trajectories.append([self.actions[idx] for idx in batch_indices])
        return actions_trajectories

    def get_states_of_trajectory(
        self,
        traj_idx: int,
        states: Optional[
            Union[TensorType["n_states", "..."], npt.NDArray[np.float32], List]
        ] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> Union[
        TensorType["n_states", "state_proxy_dims"], npt.NDArray[np.float32], List
    ]:
        """
        TODO: docstring
        """
        # TODO: re-implement using the batch indices in self.trajectories[traj_idx]
        # If either states or traj_indices are not None, both must be the same type and
        # have the same length.
        # TODO: or add sort_by
        if states is not None or traj_indices is not None:
            assert type(states) == type(traj_indices)
            assert len(states) == len(traj_indices)
        else:
            states = self.states
            traj_indices = self.traj_indices
        if torch.is_tensor(states):
            return states[tlong(traj_indices, device=self.device) == traj_idx]
        elif isinstance(states, list):
            return [
                state for state, idx in zip(states, traj_indices) if idx == traj_idx
            ]
        elif isinstance(states, np.ndarray):
            return states[np.array(traj_indices) == traj_idx]
        else:
            raise ValueError("states can only be list, torch.tensor or ndarray")

    def merge(self, batches: List):
        """
        Merges the current Batch (self) with the Batch or list of Batches passed as
        argument.

        Returns
        -------
        self
        """
        if not isinstance(batches, list):
            batches = [batches]
        for batch in batches:
            # Shift trajectory indices of batch to merge
            if len(self) == 0:
                traj_idx_shift = 0
            else:
                traj_idx_shift = np.max(list(self.trajectories.keys())) + 1
            batch._shift_traj_indices(by=traj_idx_shift)
            # Merge main data
            self.size += batch.size
            self.envs.update(batch.envs)
            self.trajectories.update(batch.trajectories)
            self.traj_indices.extend(batch.traj_indices)
            self.state_indices.extend(batch.state_indices)
            self.states.extend(batch.states)
            self.actions.extend(batch.actions)
            self.done.extend(batch.done)
            # Merge "optional" data
            if self.states_policy is not None and batch.states_policy is not None:
                self.states_policy = extend(self.states_policy, batch.states_policy)
            else:
                self.states_policy = None
            if self.parents_policy is not None and batch.parents_policy is not None:
                self.parents_policy = extend(self.parents_policy, batch.parents_policy)
            else:
                self.parents_policy = None
            if self.masks_forward_available and batch.masks_forward_available:
                self.masks_invalid_actions_forward = extend(
                    self.masks_invalid_actions_forward,
                    batch.masks_invalid_actions_forward,
                )
            else:
                self.masks_forward = None
            if self.masks_backward_available and batch.masks_backward_available:
                self.masks_invalid_actions_backward = extend(
                    self.masks_invalid_actions_backward,
                    batch.masks_invalid_actions_backward,
                )
            else:
                self.masks_backward = None
            if self.parents_available and batch.parents_available:
                self.parents = extend(self.parents, batch.parents)
            else:
                self.parents = None
            if self.parents_all_available and batch.parents_all_available:
                self.parents_all = extend(self.parents_all, batch.parents_all)
            else:
                self.parents_all = None
            if self.rewards_available and batch.rewards_available:
                self.rewards = extend(self.rewards, batch.rewards)
            else:
                self.rewards = None
        assert self.is_valid()
        return self

    def is_valid(self) -> bool:
        """
        Performs basic checks on the current state of the batch.

        Returns
        -------
        True if all the checks are valid, False otherwise.
        """
        if len(self.states) != len(self):
            return False
        if len(self.actions) != len(self):
            return False
        if len(self.done) != len(self):
            return False
        if len(self.traj_indices) != len(self):
            return False
        if len(self.state_indices) != len(self):
            return False
        if set(np.unique(self.traj_indices)) != set(self.envs.keys()):
            return False
        if set(self.trajectories.keys()) != set(self.envs.keys()):
            return False
        batch_indices = [
            idx for indices in self.trajectories.values() for idx in indices
        ]
        if len(batch_indices) != len(self):
            return False
        if len(np.unique(batch_indices)) != len(batch_indices):
            return False
        return True

    def _shift_traj_indices(self, by: int):
        """
        Adds the integer by given as an argument to all the trajectory indices and
        environment ids.

        Returns
        -------
        self
        """
        if not self.is_valid():
            raise Exception("Batch is not valid before attempting indices shift")
        self.traj_indices = [idx + by for idx in self.traj_indices]
        self.trajectories = {k + by: v for k, v in self.trajectories.items()}
        self.envs = {k + by: env.set_id(k + by) for k, env in self.envs.items()}
        if not self.is_valid():
            raise Exception("Batch is not valid after performing indices shift")
        return self
