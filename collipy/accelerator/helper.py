"""Helper module for accelerator pkg"""
from ..injection import DecayMode, Injection
from typing import Callable


def rel_cond_func(threshold: float) -> Callable:

    def foo(inj: Injection) -> bool:
        if inj.max_rel <= threshold:
            return True
        return False

    return foo

