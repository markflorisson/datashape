Unifiction
==========
The unification system addresses the problem of matching source types to
destination types. Source types originate from arrays, and destination types
from blaze function signatures. Example:

```python
@function('A... int32 -> A... int32 -> A... int32')
def add(a, b):
    ...

a = ones(dshape('10, 10, int32))
twos = add(a, a)
```

This will set up the following equations:

```python
    10, 10, int32 = A... int32
    10, 10, int32 = A... int32
```

During this matching process, the system will try to find solutions for the
type variables, and substitute the solution for the result type, conceptually:

```python
    # Set up equations for unification.

    # The left hand sides come from array
    # types (the first element in the two-tuples), i.e. array inputs to
    # blaze functions.

    # The right hand sides (the second element in the two-tuples) come from
    # the blaze function signatures.
    equations = [
        (dshape('10, 10, int32'), dshape('A... int32')),
        (dshape('10, 10, int32'), dshape('A... int32')),
    ]

    restype = dshape('A... int32')
    solution = unify(equations)     # { TypeVar('A') : [10, 10] }
    concrete_restype = substitute(solution, restype)

    print(concrete_restype) # dshape('10, 10, int32')
```

Semantics
---------
Equations can be fed into the system as demonstrated above: by matching
argument types with parameter types. A normalization pass will perform
relabeling and decay equations into further equations for which a solution
can be derived. This solution is then used for substitution to obtain the
result type.

The semantics of the unification system's equations are equality. Coercion
semantics may be explicitly signified in the terms of the equation, for which
we'll use the ```~``` operator (chosen somewhat arbitrarily).

Ellipses, TypeVars and Broadcasting
-----------------------------------
The semantics for ellipses with type variables is also equality. That means
that multiple matches for the same ellipsis with the same type variable will
need to be equal.

We lump broadcasting and element-type casting under coercion semantics, using
the ```~``` qualifier. For instance, if we were to accept broadcasting in our
```add``` function, we could write:

```python
    # Equations with coercion semantics

    equations = [
        (dshape('10, 10, int32'), dshape('~A... int32')),
        (dshape('10, 10, int32'), dshape('~A... int32')),
    ]
```

We could now put different dimensions that broadcast together on the left
hand sides, e.g. the following equations will be valid:

```python
    # Equations with coercion semantics

    equation1 = [
        (dshape('1,  10, int32'), dshape('~A... int32')),
        (dshape('10, 10, int32'), dshape('~A... int32')),
    ]
    # solution1 = { TypeVar('A') : [10, 10] }


    equation2 = [
        (dshape('10, int32'), dshape('~A... int32')),
        (dshape('10, 10, int32'), dshape('~A... int32')),
    ]
    # solution2 = { TypeVar('A') : [10, 10] }


    equation3 = [
        (dshape('10, 10, int32'), dshape('~A... int32')),
        (dshape('int32'), dshape('~A... int32')),
    ]
    # solution3 = { TypeVar('A') : [10, 10] }
```

But the below would not be valid:

```python
    # Equations with coercion semantics

    invalid_equation = [
        (dshape('1,  5, int32'), dshape('~A... int32')),
        (dshape('10, 10, int32'), dshape('~A... int32')),
    ]

    # '5' and '10' are not compatible!
```

We can define semantics for type variables analogously: solutions for type
variables must match exactly, unless the coercion semantics are active.
For dimensions this means broadcasting, and for dtypes this means casting:

```python
    # Equations with coercion semantics

    # In the following equation we allow broadcasting for the first dimension,
    # but not for the second
    dim_equation = [
        (dshape('1,  10, int32'), dshape('~a, b, int32')),
        (dshape('10, 10, int32'), dshape('~a, b, int32')),
    ]
    # dim_solution = { TypeVar('a') : 10, TypeVar('b'): 10 }

    # In the following equation we allow casting of the dtype, but no
    # broadcasting
    dtype_equation = [
        (dshape('10, 10, float64'), dshape('a, b, ~c')),
        (dshape('10, 10, int32'),   dshape('a, b, ~c')),
    ]
    # dtype_solution = { TypeVar('a'): 10, TypeVar('b'): 10,
    #                    TypeVar('c'): float64 }
```

In the above example we used type variables, but coercion semantics hold
just as well for concrete - non-variable - right hand sides. For instance,
the equation ```10, 10, int32 = 10, 10, ~float64``` will unify, since we allow
```int32``` to be cast to ```float64```.

These coercion operators are only allowed on the right-hand side. They
indicate that (parts of) the LHS may be coerced to the RHS. We can now allow
a type signature where one or several arguments determine the type, and coerce
other arguments to match that type. E.g.

```python
(dtype * ~dtype) -> dtype)
```

When we put in ```(float32, int32)```, we first establish that ```dtype``` must
be a ```float32```, since that is a non-coercible input type. We then say that
the second argument must match this, and may be coerced if it is not equal.

Hence the solver proceeds in two steps:

    * Solve all equality constraints
    * Solve all coercion constraints, under the assumptions of the previous
      step

