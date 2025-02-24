from typing import TypeVar

from elasticai.creator import ir

PNode = TypeVar("PNode", bound=ir.Node)
GNode = TypeVar("GNode", bound=ir.Node)
# This is a type alias for a tuple of two implementations
