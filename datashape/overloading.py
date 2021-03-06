from __future__ import print_function, division, absolute_import

import types
import inspect
import functools

from collections import namedtuple, defaultdict
from itertools import chain

from datashape.error import UnificationError, CoercionError, OverloadError
from datashape import (coretypes as T, unify, dshape, dummy_signature)

# -- utils -- #

def listify(f):
    """Decorator to turn generator results into lists"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return list(f(*args, **kwargs))
    return wrapper

# -- overloading / dispatching -- #

class Dispatcher(object):
    """
    Dispatcher for overloaded functions

    Attributes
    ==========

    f: FunctionType
        Initial python function that got overloaded

    overloads: (FunctionType, str, dict)
        A three-tuple of (py_func, signature, kwds)
    """

    def __init__(self):
        self.f = None
        self.overloads = []
        self.argspec = None

    def add_overload(self, f, signature, kwds, argspec=None):
        # TODO: assert signature is "compatible" with current signatures
        if self.f is None:
            self.f = f

        # Process signature
        if isinstance(f, types.FunctionType):
            argspec = argspec or inspect.getargspec(f)
            if self.argspec is None:
                self.argspec = argspec
            alpha_equivalent(self.argspec, argspec)

        # TODO: match signature to be a Function type with correct arity
        self.overloads.append((f, signature, kwds))

    def lookup_dispatcher(self, args, kwargs, constraints=None):
        assert self.f is not None
        args = flatargs(self.f, tuple(args), kwargs)
        types = list(map(T.typeof, args))
        match = best_match(self, types, constraints)
        return match, args

    def dispatch(self, *args, **kwargs):
        match, args = self.lookup_dispatcher(args, kwargs)
        return match.func(*args)

    def simple_dispatch(self, *args, **kwargs):
        assert self.f is not None
        args = flatargs(self.f, args, kwargs)
        types = list(map(T.typeof, args))
        candidates = find_matches(self.overloads, types)
        if len(candidates) != 1:
            raise OverloadError(
                "Cannot perform simple dispatch with %d input types")

        [dst_sig, sig, func] = candidates
        # TODO: convert argument types using dst_sig
        return func(*args)

    __call__ = dispatch

    def __repr__(self):
        signatures = [sig for f, sig, _ in self.overloads]
        return '<%s: \n%s>' % (self.f and self.f.__name__,
                               "\n".join("    %s" % (s,) for s in signatures))


def overload(signature, dispatcher=None, **kwds):
    """
    Overload `func` with new signature, or find this function in the local
    scope with the same name.

        @overload('Array[dtype, ndim] -> dtype')
        def myfunc(...):
            ...
    """
    def decorator(f, signature=signature):
        if signature is None:
            signature = dummy_signature(f)
        else:
            signature = dshape(signature)

        disp = dispatcher or f.__globals__.get(f.__name__) or Dispatcher()
        disp.add_overload(f, signature, kwds)
        return disp

    return decorator


def overloadable(f):
    """
    Make a function overloadable, useful if there's no useful defaults to
    overload on
    """
    return Dispatcher()

#------------------------------------------------------------------------
# Matching
#------------------------------------------------------------------------

Overload = namedtuple('Overload', 'resolved_sig, sig, func, constraints, kwds')

def best_match(func, argtypes, constraints=None):
    """
    Find a best match in for overloaded function `func` given `argtypes`.

    Parameters
    ----------
    func: Dispatcher
        Overloaded Blaze function

    argtypes: [Mono]
        List of input argument types

    constraints: [(TypeVar, Mono)]
        Optional set of constraints, see unification.py

    Returns
    -------
    Overloaded function as an `Overload` instance.
    """
    matches = match_by_weight(func, argtypes, constraints=constraints)

    if not matches:
        raise OverloadError(
            "No overload for function %s matches for argtypes (%s)" % (
                                    func, ", ".join(map(str, argtypes))))

    # -------------------------------------------------
    # Return candidate with minimum weight

    candidates = matches[min(matches)]
    if len(candidates) > 1:
        raise OverloadError(
            "Ambiguous overload for function %s with\ninput types:\n%s\nambiguous candidates:\n%s" % (
                func.f.__name__,
                "    " + "; ".join(map(str, argtypes)),
                "\n".join("    %s" % (overload.resolved_sig,) for overload in candidates)))
    else:
        return candidates[0]

def match_by_weight(func, argtypes, constraints=None):
    """
    Return all matched overloads for function `func` given `argtypes`.

    Parameters
    ----------
    func: Dispatcher
        Overloaded Blaze function

    argtypes: [Mono]
        List of input argument types

    constraints: [(TypeVar, Mono)]
        Optional set of constraints, see unification.py

    Returns
    -------
    { weight : [Overload] }
    """
    from datashape import coercion_cost
    overloads = func.overloads

    # -------------------------------------------------
    # Find candidates

    candidates = find_matches(overloads, argtypes, constraints or [])

    # -------------------------------------------------
    # Weigh candidates

    matches = defaultdict(list)
    for match in candidates:
        in_signature = T.Function(*list(argtypes) + [T.TypeVar('R')])
        signature = match.sig
        try:
            weight = coercion_cost(in_signature, signature)
        except CoercionError:
            pass
        else:
            matches[weight].append(match)

    return matches


@listify
def find_matches(overloads, argtypes, constraints=()):
    """Find all overloads that unify with the given inputs"""
    input = T.Function(*list(argtypes) + [T.TypeVar('R')])
    for func, sig, kwds in overloads:
        assert isinstance(sig, T.Function), sig

        # -------------------------------------------------
        # Error checking
        l1, l2 = len(sig.argtypes), len(argtypes)
        if l1 != l2:
            raise TypeError(
                "Expected %d args, got %d for function %s" % (l1, l2, func))

        # -------------------------------------------------
        # Unification

        equations = list(chain([(input, sig)], constraints))
        broadcasting = [True] * l1

        try:
            result, remaining = unify(equations, broadcasting)
        except UnificationError:
            continue
        else:
            dst_sig = result[0]
            yield Overload(dst_sig, sig, func, remaining, kwds)

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def flatargs(f, args, kwargs, argspec=None):
    """
    Return a single args tuple matching the actual function signature, with
    extraneous args appended to a new tuple 'args' and extraneous keyword
    arguments inserted in a new dict 'kwargs'.

        >>> def f(a, b=2, c=None): pass
        >>> flatargs(f, (1,), {'c':3})
        (1, 2, 3)
        >>> flatargs(f, (), {'a': 1})
        (1, 2, None)
        >>> flatargs(f, (1, 2, 3), {})
        (1, 2, 3)
        >>> flatargs(f, (2,), {'a': 1})
        Traceback (most recent call last):
            ...
        TypeError: f() got multiple values for keyword argument 'a'
    """
    argspec = inspect.getargspec(f) if argspec is None else argspec
    defaults = argspec.defaults or ()
    kwargs = dict(kwargs)

    def unreachable():
        f(*args, **kwargs)
        assert False, "unreachable"

    if argspec.varargs or argspec.keywords:
        raise TypeError("Variable arguments or keywords not supported")

    # -------------------------------------------------
    # Validate argcount

    if (len(args) < len(argspec.args) - len(defaults) - len(kwargs) or
            len(args) > len(argspec.args)):
        # invalid number of arguments
        unreachable()

    # -------------------------------------------------

    # Insert defaults

    tail = min(len(defaults), len(argspec.args) - len(args))
    if tail:
        for argname, default in zip(argspec.args[-tail:], defaults[-tail:]):
            kwargs.setdefault(argname, default)

    # Parse defaults
    extra_args = []
    for argpos in range(len(args), len(argspec.args)):
        argname = argspec.args[argpos]
        if argname not in kwargs:
            unreachable()

        extra_args.append(kwargs[argname])
        kwargs.pop(argname)

    # -------------------------------------------------

    if kwargs:
        unreachable()

    return args + tuple(extra_args)


def alpha_equivalent(spec1, spec2):
    """
    Return whether the inspect argspec `spec1` and `spec2` are equivalent
    modulo naming.
    """
    return (len(spec1.args) == len(spec2.args) and
            bool(spec1.varargs) == bool(spec2.varargs) and
            bool(spec1.keywords) == bool(spec2.keywords))


def lookup_previous(f, scopes=None):
    """
    Lookup a previous function definition in the current namespace, i.e.
    for overloading purposes.
    """
    if scopes is None:
        scopes = []

    scopes.append(f.__globals__)

    for scope in scopes:
        if scope.get(f.__name__):
            return scope[f.__name__]

    return None


if __name__ == '__main__':
    import doctest
    doctest.testmod()