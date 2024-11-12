import jax
import jax.numpy as jnp
import flax.linen as nn

# -- taken from netket -- #
import sys
from functools import partial, wraps

class HashablePartial(partial):
    """
    A class behaving like functools.partial, but that retains it's hash
    if it's created with a lexically equivalent (the same) function and
    with the same partially applied arguments and keywords.

    It also stores the computed hash for faster hashing.
    """

    # TODO remove when dropping support for Python < 3.10
    def __new__(cls, func, *args, **keywords):
        # In Python 3.10+ if func is itself a functools.partial instance,
        # functools.partial.__new__ would merge the arguments of this HashablePartial
        # instance with the arguments of the func
        # Pre 3.10 this does not happen, so here we emulate this behaviour recursively
        # This is necessary since functools.partial objects do not have a __code__
        # property which we use for the hash
        # For python 3.10+ we still need to take care of merging with another HashablePartial
        while isinstance(
            func, partial if sys.version_info < (3, 10) else HashablePartial
        ):
            original_func = func
            func = original_func.func
            args = original_func.args + args
            keywords = {**original_func.keywords, **keywords}
        return super().__new__(cls, func, *args, **keywords)

    def __init__(self, *args, **kwargs):
        self._hash = None

    def __eq__(self, other):
        return (
            type(other) is HashablePartial
            and self.func.__code__ == other.func.__code__
            and self.args == other.args
            and self.keywords == other.keywords
        )

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(
                (self.func.__code__, self.args, frozenset(self.keywords.items()))
            )

        return self._hash

    def __repr__(self):
        return f"<hashable partial {self.func.__name__} with args={self.args} and kwargs={self.keywords}, hash={hash(self)}>"


def reim(f):
    r"""Modifies a non-linearity to act separately on the real and imaginary parts"""

    @wraps(f)
    def reim_f(f, x):
        sqrt2 = jnp.sqrt(jnp.array(2, dtype=x.real.dtype))
        if jnp.iscomplexobj(x):
            return jax.lax.complex(f(sqrt2 * x.real), f(sqrt2 * x.imag)) / sqrt2
        else:
            return f(x)

    fun = HashablePartial(reim_f, f)

    fun.__name__ = f"reim_{f.__name__}"  # type: ignore
    fun.__doc__ = (
        f"{f.__name__} applied separately to the real and"
        f"imaginary parts of it's input.\n\n"
        f"The docstring to the original function follows.\n\n"
        f"{f.__doc__}"
    )

    return fun

reim_selu = reim(jax.nn.selu)
# -- taken from netket -- #

def square(x):
    return x**2


def poly6(x):
    x = x**2
    return ((0.022222222 * x - 0.083333333) * x + 0.5) * x


def poly5(x):
    xsq = x**2
    return ((0.133333333 * xsq - 0.333333333) * xsq + 1.) * x

def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


activationFunctions = {
    "square": square,
    "poly5": poly5,
    "poly6": poly6,
    "elu": nn.elu,
    "relu": nn.relu,
    "tanh": jnp.tanh,
    "reim_selu": reim_selu,
}


