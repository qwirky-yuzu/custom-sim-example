"""Creates the observation and action spaces of the RLHR problem set.
"""

from dataclasses import dataclass
import numpy as np

# Gynasium packages
from gymnasium.spaces import Discrete, Box, Tuple

@dataclass
class ObsSpace:
    """Observation space for the RLHR problem set."""

    @staticmethod
    def create_pos_details():
        """
        Creates the observation space for the position attributes. You 
        may mix and match different data types together depending on your
        attributes but do take not of the dimensionality that will blow up
        as a result.

        Returns:
            Tuple: A tuple of Box spaces representing the position attributes.
        """

        # Position attributes
        feat_1 = Box(low=0, high=1, dtype=np.float16)
        feat_2 = Box(low=0, high=1, dtype=np.float16)
        feat_3 = Box(low=0, high=1, dtype=np.float16)
        feat_4 = Box(low=0, high=1, dtype=np.float16)
        feat_5 = Box(low=0, high=1, dtype=np.float16)
        feat_6 = Box(low=0, high=1, shape=(92,), dtype=np.float16)

        return Tuple(
            (feat_1, feat_2, feat_3, feat_4, feat_5, feat_6)
        )

    @staticmethod
    def create_staff_details(max_staff_limit: int):
        """
        Creates the observation space for a staff member's details.

        Args:
            max_staff_limit (int): The maximum number of staff members.

        Returns:
            Tuple: A tuple of Box spaces representing the staff member's details.
        """
        # Staff attributes
        attr_1 = Box(low=0, high=1, dtype=np.float16)
        attr_2 = Box(low=0, high=1, dtype=np.float16)
        attr_3 = Box(low=0, high=1, dtype=np.float16)
        attr_4 = Box(low=0, high=1, dtype=np.float16)
        attr_5 = Box(low=0, high=1, dtype=np.float16)
        attr_6 = Box(low=0, high=1, shape=(4,), dtype=np.float16)

        return Tuple(
            (
                Tuple((attr_1, attr_2, attr_3, attr_4, attr_5, attr_6))
                for _ in range(max_staff_limit)
            )
        )


@dataclass
class ActSpace:
    """Action space for the RLHR problem set."""

    @staticmethod
    def create(max_action_space_size: int) -> Discrete:
        """
        Creates the action space for the RLHR problem set. The action space is
        represented by a discrete integer from 0 to 99 (depending on param
        max_action_space_size, defaults to 100).

        Args:
            max_action_space_size (int): The maximum size of the action space.

        Returns:
            Discrete: The action space.
        """
        # Create the action space with the specified maximum size.
        # The action space is a discrete space with values ranging from 0 to
        # max_action_space_size - 1.
        return Discrete(max_action_space_size)


if __name__ == "__main__":
    print(ObsSpace.create_pos_details())
