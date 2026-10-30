"""A terminal coding agent built on Claude's bash and text editor tools."""

from .agent import Agent, Events, Result
from .policy import Approver
from .workspace import Workspace

__all__ = ["Agent", "Approver", "Events", "Result", "Workspace"]
__version__ = "0.1.0"
