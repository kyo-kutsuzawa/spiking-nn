import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../synergy/build"))
from Release.synergy import TimeVaryingSynergy, decode, encode, extract  # type: ignore
