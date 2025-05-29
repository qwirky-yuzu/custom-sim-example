"""Environment creation module
"""
# Basic packages
import json
import pandas as pd
import numpy as np

# Gynasium packages
from gymnasium.utils import seeding
from gymnasium.spaces import Dict, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

try:
    # Custom packages via pip
    print("Importing modules for pip installable environment...", flush=True)
    from rlhr_sim.simulator.utils.spaces import ObsSpace, ActSpace
except ModuleNotFoundError:
    # Custom packages via path-based import
    print("Importing modules for local environment...",  flush=True)
    from utils.spaces import ObsSpace, ActSpace


class CustomEnv(AECEnv):
    """Creates a custom PettingZoo environment. In this scenario, we
    are simulating a custom use case where we want to use RL to move
    staff around an organization in an optimal manner. 

    This is merely a sample and we have simplified the use case by
    reducing it down to its barebones architecture. The internal workings
    have been redacted but by complying with this boilerplate code, 
    you should be able to create your own custom simulator. 

    While we have chosen to use an AEC environment here, under the hood, 
    RLSuite wraps all environments as a Parellel environment. 

    Read more about how to choose between AEC and Parellel environments
    here:
    - https://pettingzoo.farama.org/api/aec/
    - https://pettingzoo.farama.org/api/parallel/
    """

    # Default parameter
    metadata = {
        "render_modes": ["ansi", "human"], "is_parallelizable": True
    }

    def __init__(self,
                 max_action_space_size:int,
                 eps_end_timestep:int,
                 render_mode="ansi",
                 agent:str="HR_1",
                 **kwargs
                 ):
        """
        Initializes the CustomEnv class.
        """
        super().__init__()

        # Custom data
        # You will need an actual dataframe or any relevant object
        self.staff_data = pd.DataFrame(
            data={"staff_id":["abc123" for _ in range(10)]}
        )
        self.step_count = 0
        self.eps_end_timestep = eps_end_timestep
        self.max_action_space_size = max_action_space_size
        self.kwargs = kwargs

        # Global variables for PZ environment compliance
        self.agents = [agent]
        self.possible_agents = self.agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.render_mode = render_mode

        # Seed variable - Compulsory to be set
        self._seed()

        # Create the default observation space
        self.observation_spaces = {
                    i: Dict({
                        "observation":Dict({
                            "position_details":ObsSpace.create_pos_details(),
                            "staff_details":ObsSpace.create_staff_details(max_staff_limit=self.max_action_space_size)
                            }),
                        "action_mask":Box(low=0, high=1, shape=(self.max_action_space_size, ), dtype=np.float16)
                        })
                    for i in self.agents
                }

        # Create the default action space
        self.action_spaces = {
            i: ActSpace.create(max_action_space_size=self.max_action_space_size)
            for i in self.agents
        }

    def _seed(self, seed=None):
        # This is a default function, don't change it
        self.np_random, seed = seeding.np_random(seed)

    def observation_space(self, agent) -> dict:
        """Creates the base observation space in which the agent will
        function in.

        Args:
            agent (str): Name of the agent which should be a string object

        Returns:
            obs_space (dict): The base observation space represented as a
                              dictionary object.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent) -> dict:
        """Set of all possible actions in an environment

        Args:
            agent (str): Name of the agent which should be a string object

        Returns:
            act_space (dict): The actions that the agent can take, in this
                              case, the maximum number of actions the agent
                              can take at any point in time is to move a
                              limited to the maximum number of staff that
                              we will consider at any given point in time.
        """

        return self.action_spaces[agent]

    #@profile
    def observe(self, agent) -> dict:
        """Returns a single observation by the agent

        Args:
            agent (str): Name of the agent which must be a string object
                         Defaults to None.

        Returns:
            obs (dict): The observation of the agent
        """

        obs = self.observation_space(agent=agent).sample()

        return obs

    #@profile
    def step(self, action):
        """Takes and executes the `action` of the agent in the environment.
        Returns the resultant observation after taking the action.

        Args:
            action (int): Integer between 0 and 99, representing the temporary
                          indexes of the 100 eligible staff
        """

        # If truncation condition is met, the run will end & env will reset
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
            or action is None
        ):
            return self._was_dead_step(action)

        # Step incremented regardless of whether a staff is moved or not
        self.step_count += 1
        self._clear_rewards()
        print(f"Step count: {self.step_count}", flush=True)

        # You will need a set of code to handle the movement
        # This is not shown here as it is too complex but essentially
        # this is how your agent interacts with your custom sim env
        
        # Update rewards
        self.rewards[self.agents[0]] += 1

        if self.step_count % self.eps_end_timestep == 0:
            print("End of episode.", flush=True)
            infos_dictionary = {}
            self.infos = {i: infos_dictionary for i in self.agents}

            # Episode completion conditions
            self.terminations = {i: True for i in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self._accumulate_rewards()

    #@profile
    def reset(self, seed=None, options=None):
        """
        Resets the environment
        """
        if seed is not None:
            self._seed(seed=seed)

        # PZ variables to reset
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        # Reselect agent - PZ compliance requirement
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Custom variable to reset
        self.step_count = 0


def make_env(raw_env):
    """
    Wraps the raw simulator environment up for RLSuite to interface.

    Args:
        raw_env (object): The raw simulator environment.

    Returns:
        object: The wrapped simulator environment.
    """

    def env(**kwargs):

        env = raw_env(**kwargs)

        # this wrapper helps error handling for discrete action spaces
        env = wrappers.AssertOutOfBoundsWrapper(env)
        # Provides a wide vareity of helpful user errors
        # Strongly recommended
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env

env = make_env(CustomEnv)
parallel_env = parallel_wrapper_fn(env)
