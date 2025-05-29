"""Pettingzoo style scripting to import environment
"""
import sys
import os
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)
a=str(path.parent.absolute())
sys.path.append(a)

from .rlhr_env import env

__all__ = ["env"]