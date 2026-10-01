#!/usr/bin/env python3
"""p2_numbers.py -- the constants and identities quoted as [RAN] in
P2_DSN_LOSS_AXES.md (S3.2, Table P2.1, S3.3) and in 00_INDEX.md S8.

Torch-free: numpy and the standard library only, so it runs in any
environment. It recomputes the angular-hinge constants 4 tan^2(alpha) and
the DSN's silhouette reading 1 - 4 sin^2(alpha), the margin conversion
m_sq = 2 m_cos, the two identities the loss rests on (squared Euclidean
versus cosine distance on unit rows; the midpoint identity), the hinge
boundary in the isosceles construction, the metric batch composition and
triplet pools at B_met = 32, the ETF targets, the lambda_dsn range, the
ramp horizons, and the count of legal and reachable (mining, loss, filter)
conditions under the legality projection, which is re-implemented here in
four lines rather than imported so the file has no dependency on the DSN
tree (the projection's two clauses are condition_space.project_condition's).

Run:
    cd hpc/joint/docs/tools
    python3 p2_numbers.py

Pure ASCII, LF only.
"""
import math
import numpy as np

print("== angular hinge constants ==")
for a in (2.0, 5.0, 10.0, 18.0, 20.0, 30.0):
    t = math.tan(math.radians(a)); s = math.sin(math.radians(a))
    print("alpha %5.1f deg  4tan^2 = %.6f  floor 1-4sin^2 = %.5f  tolerated a/b = %.5f"
          % (a, 4*t*t, 1-4*s*s, 4*s*s))

print("\n== margin conversion (unit rows): margin_sq = 2 m_cos ==")
for m in (0.1, 0.2, 0.3, 1.0):
    print("m_cos %.2f -> margin_sq %.2f  (band in cosine distance %.2f)" % (m, 2*m, m))
print("max squared distance between unit vectors = 4 (antipodal); max cosine distance = 2")

print("\n== identity ||u-v||^2 = 2 d_cos(u,v) on unit rows ==")
rng = np.random.default_rng(0)
u = rng.normal(size=(1000, 12)); u /= np.linalg.norm(u, axis=1, keepdims=True)
v = rng.normal(size=(1000, 12)); v /= np.linalg.norm(v, axis=1, keepdims=True)
lhs = ((u-v)**2).sum(1); rhs = 2*(1-(u*v).sum(1))
print("max |lhs-rhs| = %.2e" % np.abs(lhs-rhs).max())

print("\n== median-length identity D_nc = (2 D_an + 2 D_pn - D_ap)/4 (arbitrary vectors) ==")
xa, xp, xn = (rng.normal(size=(4000, 8)) for _ in range(3))
xc = 0.5*(xa+xp)
d_nc = ((xn-xc)**2).sum(1); d_ap = ((xa-xp)**2).sum(1); d_an = ((xa-xn)**2).sum(1); d_pn = ((xp-xn)**2).sum(1)
pred = (2*d_an + 2*d_pn - d_ap)/4
print("max rel residual = %.2e" % (np.abs(d_nc-pred)/np.abs(d_nc)).max())

print("\n== symmetric construction: hinge boundary at D_ap/D_an = 4 sin^2 alpha ==")
for a in (2.0, 18.0, 20.0):
    t = math.tan(math.radians(a)); w = t
    xa_ = np.array([1.0, w]); xp_ = np.array([1.0, -w]); xn_ = np.zeros(2)
    D_ap = ((xa_-xp_)**2).sum(); D_an = ((xa_-xn_)**2).sum(); xc_ = 0.5*(xa_+xp_); D_nc = ((xn_-xc_)**2).sum()
    hinge = D_ap - 4*t*t*D_nc
    print("alpha %4.1f: D_ap/D_an = %.6f  4sin^2 = %.6f  hinge = %.1e" % (a, D_ap/D_an, 4*math.sin(math.radians(a))**2, hinge))

print("\n== metric batch composition (b_met = 32, class-balanced, per = max(1, 32//K)) ==")
for K in (2, 3, 4):
    per = max(1, 32//K)
    rows = per*K
    n_trip_all = rows*(per-1)*(rows-per)   # anchors x positives (same class, not self) x negatives
    print("K=%d: per class %d, rows %d, all (a,p,n) triplets %d, BatchEasyHardMiner max %d triplets (one per anchor)" % (K, per, rows, n_trip_all, rows))

print("\n== ETF target -1/(K-1) and pairwise angle ==")
for K in (2, 3, 4):
    rho = -1.0/(K-1)
    print("K=%d: target cosine %.4f, angle %.1f deg" % (K, rho, math.degrees(math.acos(rho))))

print("\n== lambda_dsn range from log10 in [-3, 1]; runner default 0.1; inactive pin log10 = 0 ==")
print("lambda in [%g, %g]; runner default 0.1 = 10^%.0f; pin -> lambda 1.0 (never passed: dsn_on=0 -> --lambda-dsn 0)" % (10**-3, 10**1, math.log10(0.1)))

print("\n== planned horizons T for the separation ramp (tau = 0 in the joint space -> g = 1 regardless) ==")
print("joint loop: epochs*steps_per_epoch = 10*25 = %d; encoder-only (A0/A0s): --encoder-steps = 200" % (10*25))

print("\n== cell counts: 3 x 3 x 2 = 18 raw triples -> 13 legal; joint space strict fixed = 1 ==")
MIN = ("hard", "easy_positive", "easy_pos_semihard_neg"); LOSS = ("triplet", "joint", "joint_sep")
def proj(m, l, s):
    s = bool(s)
    if l == "triplet": s = False
    elif m == "hard": s = False
    return (m, l, s)
legal = []
for m in MIN:
    for l in LOSS:
        for s in (False, True):
            c = proj(m, l, s)
            if c not in legal: legal.append(c)
print("legal triples:", len(legal))
reach = sorted(set(proj(m, l, 1) for m in MIN for l in LOSS))
print("reachable from the joint space (strict fixed 1):", len(reach))
for c in reach: print("  ", c)
