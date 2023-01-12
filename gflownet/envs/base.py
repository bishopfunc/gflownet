"""
Base class of GFlowNet environments
"""
from abc import abstractmethod
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pathlib import Path
from torch.distributions import Categorical
from torchtyping import TensorType


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        reward_norm_std_mult=0,
        reward_func="power",
        energies_stats=None,
        denorm_proxy=False,
        proxy=None,
        oracle=None,
        proxy_state_format=None,
        **kwargs,
    ):
        self.state = []
        self.done = False
        self.n_actions = 0
        self.id = env_id
        self.min_reward = 1e-8
        self.reward_beta = reward_beta
        self.reward_norm = reward_norm
        self.reward_norm_std_mult = reward_norm_std_mult
        self.reward_func = reward_func
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        self.proxy = proxy
        if oracle is None:
            self.oracle = self.proxy
        else:
            self.oracle = oracle
        self.proxy_state_format = proxy_state_format
        self._true_density = None
        self.action_space = []
        self.eos = len(self.action_space)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Assertions
        assert self.reward_norm > 0
        assert self.reward_beta > 0
        assert self.min_reward > 0

    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    @abstractmethod
    def get_actions_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        pass

    def get_fixed_policy_output(self):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy. As a baseline, the fixed policy is uniform over the
        dimensionality of the action space.
        """
        return np.ones(len(self.action_space))

    def get_max_path_len(
        self,
    ):
        return 1

    def state2proxy(self, state: List = None):
        """
        Prepares a state in "GFlowNet format" for the proxy.

        Args
        ----
        state : list
            A state
        """
        if state is None:
            state = self.state.copy()
        return self.statebatch2proxy([state])

    def statebatch2proxy(self, states: List[List]) -> ndt.NDArray[np.float32]:
        """
        Prepares a batch of states in "GFlowNet format" for the proxy.

        Args
        ----
        state : list
            A state
        """
        return np.array(states)

    def state2oracle(self, state: List = None):
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state : list
            A state
        """
        if state is None:
            state = self.state.copy()
        return state

    def statebatch2oracle(self, states: List[List]):
        """
        Prepares a batch of states in "GFlowNet format" for the oracles
        """
        return states

    def reward(self, state=None, done=None):
        """
        Computes the reward of a state
        """
        if done is None:
            done = self.done
        if done:
            return np.array(0.0)
        if state is None:
            state = self.state.copy()
        return self.proxy2reward(self.proxy(self.state2proxy(state)))

    def reward_batch(self, states: List[List], done):
        """
        Computes the rewards of a batch of states, given a list of states and 'dones'
        """
        states_proxy = self.statebatch2proxy(states)[list(done), :]
        reward = np.zeros(len(done))
        if states_proxy.shape[0] > 0:
            reward[list(done)] = self.proxy2reward(self.proxy(states_proxy))
        return reward

    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet: the inputs proxy_vals is
        expected to be a negative value (energy), unless self.denorm_proxy is True. If
        the latter, the proxy values are first de-normalized according to the mean and
        standard deviation in self.energies_stats. The output of the function is a
        strictly positive reward - provided self.reward_norm and self.reward_beta are
        positive - and larger than self.min_reward.
        """
        if self.denorm_proxy:
            proxy_vals = proxy_vals * self.energies_stats[3] + self.energies_stats[2]
        if self.reward_func == "power":
            return np.clip(
                (-1.0 * proxy_vals / self.reward_norm) ** self.reward_beta,
                self.min_reward,
                None,
            )
        elif self.reward_func == "boltzmann":
            return np.clip(
                np.exp(-1.0 * self.reward_beta * proxy_vals),
                self.min_reward,
                None,
            )
        else:
            raise NotImplemented

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        if self.reward_func == "power":
            return -np.exp(
                (np.log(reward) + self.reward_beta * np.log(self.reward_norm))
                / self.reward_beta
            )
        elif self.reward_func == "boltzmann":
            return -1.0 * np.log(reward) / self.reward_beta
        else:
            raise NotImplemented

    def state2policy(self, state=None):
        """
        Converts a state into a format suitable for a machine learning model, such as a
        one-hot encoding.
        """
        if state is None:
            state = self.state
        return state

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Converts a batch of states into a format suitable for a machine learning model,
        such as a one-hot encoding. Returns a numpy array.
        """
        return np.array(states)

    def obs2state(self, obs: List) -> List:
        """
        Converts the model (e.g. one-hot encoding) version of a state given as
        argument into a state.
        """
        return obs

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state
        return str(state)

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        return readable

    def path2readable(self, path=None):
        """
        Converts a path into a human-readable string.
        """
        return str(path).replace("(", "[").replace(")", "]").replace(",", "")

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = []
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : tuple
            Last action performed

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        else:
            parents = []
            actions = []
        return parents, actions

    def sample_actions(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["n_states", "policy_output_dim"] = None,
        temperature_logits: float = 1.0,
        loginf: float = 1000,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs. This implementation
        is generally valid for all discrete environments.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        if sampling_method == "uniform":
            logits = torch.ones(policy_outputs.shape).to(device)
        elif sampling_method == "policy":
            logits = policy_outputs
            logits /= temperature_logits
        if mask_invalid_actions is not None:
            logits[mask_invalid_actions] = -loginf
        action_indices = Categorical(logits=logits).sample()
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        # Build actions
        actions = [self.action_space[idx] for idx in action_indices]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", 2],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions. This
        implementation is generally valid for all discrete environments.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        logits = policy_outputs
        if mask_invalid_actions is not None:
            logits[mask_invalid_actions] = -loginf
        action_indices = [self.action_space.index(action) for action in actions]
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        return logprobs

    def get_paths(self, path_list, path_actions_list, current_path, current_actions):
        """
        Determines all paths leading to each state in path_list, recursively.

        Args
        ----
        path_list : list
            List of paths (lists)

        path_actions_list : list
            List of actions within each path

        current_path : list
            Current path

        current_actions : list
            Actions of current path

        Returns
        -------
        path_list : list
            List of paths (lists)

        path_actions_list : list
            List of actions within each path
        """
        parents, parents_actions = self.get_parents(current_path[-1], False)
        if parents == []:
            path_list.append(current_path)
            path_actions_list.append(current_actions)
            return path_list, path_actions_list
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            path_list, path_actions_list = self.get_paths(
                path_list, path_actions_list, current_path + [p], current_actions + [a]
            )
        return path_list, path_actions_list

    def step(self, action_idx):
        """
        Executes step given an action.

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action_idx : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if action < self.eos:
            self.done = False
            valid = True
        else:
            self.done = True
            valid = True
            self.n_actions += 1
        return self.state, action, valid

    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state
        return False

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state, False otherwise.
        """
        if state is None:
            state = self.state
        if done is None:
            done = self.done
        mask = [False for _ in range(len(self.action_space))]
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        Returns a vector with the length of the discrete part of the action space + 1:
        True if action is invalid going backward given the current state, False
        otherwise.
        """
        # TODO
        pass

    def set_state(self, state, done):
        """
        Sets the state and done of an environment.
        """
        self.state = state
        self.done = done
        return self

    def true_density(self):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - un-normalized reward
          - states
        """
        return (None, None, None)

    @staticmethod
    def np2df(*args):
        """
        Args
        ----
        """
        return None


class Buffer:
    """
    Implements the functionality to manage various buffers of data: the records of
    training samples, the train and test data sets, a replay buffer for training, etc.
    """

    def __init__(
        self,
        env,
        make_train_test=False,
        replay_capacity=0,
        output_csv=None,
        data_path=None,
        train=None,
        test=None,
        **kwargs,
    ):
        self.env = env
        self.replay_capacity = replay_capacity
        self.main = pd.DataFrame(columns=["state", "path", "reward", "energy", "iter"])
        self.replay = pd.DataFrame(
            np.empty((self.replay_capacity, 5), dtype=object),
            columns=["state", "path", "reward", "energy", "iter"],
        )
        self.replay.reward = pd.to_numeric(self.replay.reward)
        self.replay.energy = pd.to_numeric(self.replay.energy)
        self.replay.reward = [-1 for _ in range(self.replay_capacity)]
        # Define train and test data sets
        self.train = self.make_data_set(train)
        if (
            self.train is not None
            and "output_csv" in train
            and train.output_csv is not None
        ):
            self.train.to_csv(train.output_csv)
        self.test = self.make_data_set(test)
        if (
            self.test is not None
            and "output_csv" in test
            and test.output_csv is not None
        ):
            self.test.to_csv(test.output_csv)
        # Compute buffer statistics
        if self.train is not None:
            (
                self.mean_tr,
                self.std_tr,
                self.min_tr,
                self.max_tr,
                self.max_norm_tr,
            ) = self.compute_stats(self.train)
        if self.test is not None:
            self.mean_tt, self.std_tt, self.min_tt, self.max_tt, _ = self.compute_stats(
                self.test
            )

    def add(
        self,
        states,
        paths,
        rewards,
        energies,
        it,
        buffer="main",
        criterion="greater",
    ):
        if buffer == "main":
            self.main = pd.concat(
                [
                    self.main,
                    pd.DataFrame(
                        {
                            "state": [self.env.state2readable(s) for s in states],
                            "path": [self.env.path2readable(p) for p in paths],
                            "reward": rewards,
                            "energy": energies,
                            "iter": it,
                        }
                    ),
                ],
                axis=0,
                join="outer",
            )
        elif buffer == "replay" and self.replay_capacity > 0:
            if criterion == "greater":
                self.replay = self._add_greater(states, paths, rewards, energies, it)

    def _add_greater(
        self,
        states,
        paths,
        rewards,
        energies,
        it,
    ):
        rewards_old = self.replay["reward"].values
        rewards_new = rewards.copy()
        while np.max(rewards_new) > np.min(rewards_old):
            idx_new_max = np.argmax(rewards_new)
            self.replay.iloc[self.replay.reward.argmin()] = {
                "state": self.env.state2readable(states[idx_new_max]),
                "path": self.env.path2readable(paths[idx_new_max]),
                "reward": rewards[idx_new_max],
                "energy": energies[idx_new_max],
                "iter": it,
            }
            rewards_new[idx_new_max] = -1
            rewards_old = self.replay["reward"].values
        return self.replay

    def make_data_set(self, config):
        """
        Constructs a data set asa DataFrame according to the configuration.
        """
        if config is None:
            return None
        elif "type" not in config:
            return None
        elif config.type == "all" and hasattr(self.env, "get_all_terminating_states"):
            samples = self.env.get_all_terminating_states()
        elif (
            config.type == "uniform"
            and "n" in config
            and hasattr(self.env, "get_uniform_terminating_states")
        ):
            samples = self.env.get_uniform_terminating_states(config.n)
        elif (
            config.type == "random"
            and "n" in config
            and "seed" in config
            and hasattr(self.env, "get_random_terminating_states")
        ):
            samples = self.env.get_random_terminating_states(config.n, config.seed)
        else:
            return None
        energies = self.env.oracle(self.env.statebatch2oracle(samples))
        df = pd.DataFrame(
            {
                "samples": [self.env.state2readable(s) for s in samples],
                "energies": energies,
            }
        )
        return df

    def compute_stats(self, data):
        mean_data = data["energies"].mean()
        std_data = data["energies"].std()
        min_data = data["energies"].min()
        max_data = data["energies"].max()
        data_zscores = (data["energies"] - mean_data) / std_data
        max_norm_data = data_zscores.max()
        return mean_data, std_data, min_data, max_data, max_norm_data

    def sample(
        self,
    ):
        pass

    def __len__(self):
        return self.capacity

    @property
    def transitions(self):
        pass

    def save(
        self,
    ):
        pass

    @classmethod
    def load():
        pass

    @property
    def dummy(self):
        pass
