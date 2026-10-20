#!/usr/bin/env python3
"""Derive the polynomial coefficients in core/vmath.h.

vadito computes exp, and from it the sigmoid and tanh of the LSTM, with its
own polynomials rather than the platform's libm, for two reasons: libms
disagree with each other in the last bit (glibc, macOS and MSVC's expf all
round differently somewhere), and every backend has to compute the same
bits as backend/ref in its own vector registers. A function whose every
operation is written down is one the backends can copy exactly.

Two approximations, each a minimax fit of relative error found by the Remez
exchange, in float64:

  exp(r) on [-ln2/2, ln2/2]   c0 + c1 r + ... + c6 r^6
  tanh(x) on [0, X0]          x + x^3 (t0 + t1 x^2 + ... )

The first is what exp is after the usual reduction x = n ln2 + r; the
second covers the small arguments where tanh's exp form, 1 - 2/(e^2x + 1),
cancels. The fitted error printed here is the polynomial's alone, in exact
arithmetic: tests/mathcheck measures what the float32 code actually does.

    python3 tools/fit_vmath.py
"""
import numpy as np


def remez(f, basis, offset, lo, hi, iters=40):
    """Coefficients a minimising max |(offset(x) + sum a_j basis_j(x)) / f(x) - 1|
    over [lo, hi], and that maximum."""
    n = len(basis)
    k = np.arange(n + 1)
    x = (lo + hi) / 2 + (hi - lo) / 2 * np.cos(np.pi * (n - k) / n)   # Chebyshev extrema
    grid = np.linspace(lo, hi, 200001)
    for _ in range(iters):
        # sum a_j b_j(x_i) + (-1)^i E f(x_i) = f(x_i) - offset(x_i)
        A = np.column_stack([b(x) for b in basis] + [(-1.0) ** k * f(x)])
        sol = np.linalg.solve(A, f(x) - offset(x))
        a = sol[:n]
        err = (offset(grid) + sum(c * b(grid) for c, b in zip(a, basis))) / f(grid) - 1
        # one extremum per run of constant sign, n + 1 of them
        sgn = np.sign(err)
        edges = np.flatnonzero(np.diff(sgn) != 0) + 1
        runs = np.split(np.arange(len(grid)), edges)
        ext = np.array([r[np.argmax(np.abs(err[r]))] for r in runs])
        if len(ext) < n + 1:
            break
        if len(ext) > n + 1:                  # keep the n + 1 largest, alternating
            while len(ext) > n + 1:
                drop = 0 if abs(err[ext[0]]) < abs(err[ext[-1]]) else len(ext) - 1
                ext = np.delete(ext, drop)
        x = grid[ext]
    return a, np.abs(err).max()


def main():
    ln2 = np.log(2.0)
    # All seven free: pinning c0 = c1 = 1 makes every basis function vanish
    # at r = 0, which is not a Haar system and the exchange does not
    # converge. Free, they come out within 2^-25 of 1 and round to it.
    a, e = remez(np.exp, [lambda r, j=j: r ** j for j in range(7)],
                 lambda r: 0 * r, -ln2 / 2, ln2 / 2)
    print('exp: c0 + c1 r + ... + c6 r^6, max rel err %.3g (%.3f ulp at 2^-24)' % (e, e / 2 ** -24))
    for j, c in enumerate(a):
        print('  c%d = %a  (%.9e)' % (j, float(np.float32(c)), np.float32(c)))

    for x0, deg in ((0.625, 5), (0.625, 6), (0.55, 5)):
        t, e = remez(np.tanh, [lambda x, j=j: x ** (2 * j + 3) for j in range(deg)],
                     lambda x: x, 1e-4, x0)
        print('tanh on [0, %g], %d terms: max rel err %.3g (%.3f ulp)' % (x0, deg, e, e / 2 ** -24))
        for j, c in enumerate(t):
            print('  t%d = %a  (%.9e)' % (j, float(np.float32(c)), np.float32(c)))


if __name__ == '__main__':
    main()
