"""The gate of a recorded project's evaluation (block 34; specs/SPEC.md sec.
2.6, "The gate", equations (R5)-(R8), and block 34; decisions D-161, D-163
(4), D-164).

    role = ev_gate.assign_roles(donor, fracs, n_ref, n_rank, n_null,
                                rows_per_donor, seed)
    out = ev_gate.gate(z_ref, z_rank, z_null, z_test, factors)
    out["p"]            # (n,) the p-value of each tested window (R8)

Pure functions of arrays: numpy and `npe_misspec` (the kernel's bandwidths,
the reference's self term, the p-value's convention). No torch; no file is
read or written.

CONTRACT (wave A): the constants and the signatures are fixed by
specs/SPEC.md block 34. The functions are stubs.

Pure ASCII, LF only.
"""

import _evalpath  # noqa: F401

# a row's role in the gate (`gate_role` of block 35's <stem>_sim.npz)
ROLE_NONE, ROLE_REF, ROLE_RANK, ROLE_NULL, ROLE_CHECK = 0, 1, 2, 3, 4


def assign_roles(donor, fracs, n_ref, n_rank, n_null, rows_per_donor, seed):
    """(n,) int8: the gate's role of each row of D_sim.

    donor (n,) int: the simulated network of each row, in global row order.
    fracs: (f_ref, f_rank, f_null, f_check), four positive numbers summing
    to 1. n_ref, n_rank, n_null: the caps of the reference, rank and null
    sets. rows_per_donor: the most rows of one network in the rank, null
    and check sets. seed: of the permutation of the networks.

    The rule is specs/SPEC.md block 34's: the sorted distinct networks are
    permuted by numpy.random.default_rng(seed).permutation and dealt, in
    that order, floor(f_ref G), floor(f_rank G), floor(f_null G) and the
    rest; within a group the rows are ordered by (occurrence within their
    network, position in `donor`). ValueError when a group has no network.
    """
    raise NotImplementedError("block 34")


def member_statistic(z_ref, z, factors):
    """(t, bandwidths, median) for one member: t (n,) float64, the biased
    squared MMD (R6) between each point of z (n, d_z) and the reference
    z_ref (N_ref, d_z), with the kernel (R5); bandwidths (n_h,) float64 =
    factors * median; median the median pairwise distance among the
    reference embeddings. n may be 0. ValueError on a non-finite input."""
    raise NotImplementedError("block 34")


def gate(z_ref, z_rank, z_null, z_test, factors):
    """The gate of one ensemble on n tested windows.

    z_ref (M, N_ref, d_z), z_rank (M, N_rank, d_z), z_null (M, N_null, d_z),
    z_test (M, n, d_z), n >= 0: each member's embeddings of the reference,
    rank, null and tested rows. factors: the kernel's bandwidth factors.

    Returns a dict of float64 arrays: `t`, `r`, `p_member` (M, n); `T`, `p`
    (n,); `t_rank` (M, N_rank); `t_null`, `r_null` (M, N_null); `T_null`
    (N_null,); `bandwidths` (M, n_h); `median` (M,). (R6)-(R8): r by the
    count of rank values strictly below, T the mean over the members, the
    p-values by (1 + the count of null values at least as large) / (1 +
    N_null). The null rows are ranked by the expression that ranks a tested
    window. ValueError on a non-finite input.
    """
    raise NotImplementedError("block 34")
