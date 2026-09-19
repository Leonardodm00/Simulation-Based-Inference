# joint DSN+NPE -- implementation bundle

Stages 1, 2, 3, 3b and 3c of `JOINT_DSN_NPE_PLAN_v0_6.md`, as implemented and
verified in the sandbox. **Nothing here has run on the HPC.**

Read `HANDOFF_joint_dsn_npe_impl.md` first. It is the session handoff and it
lists, in order: what exists, the environment it was verified against, three
repository findings to act on before writing more code, two couplings to sbi
0.27 internals, every bug found and fixed, every deviation from the plan, and
the next actions.

## Layout

    stage1/   bench simulator, nuisance, realisation, gap, bank builder
    stage2/   joint model, three-stream batches, losses, training loop
    stage3/   the nine-arm hypothesis test, diagnostics, report
    stage3b/  domain/objective decomposition and the two encoder probes
    stage3c/  nuisance and realisation floors, aliasing, stratification, P10
    replicate_statistic.py            the numpy SPECIFICATION for stage2/joint_losses.py
    smoke_test_replicate_statistic.py

    JOINT_DSN_NPE_PLAN_v0_6.md        the plan this implements
    METRIC_REPLICATE_v1_1.md          the derivation behind plan S2.5
    HANDOFF_joint_dsn_npe_design.md   the design-session handoff (still current)
    HANDOFF_joint_dsn_npe_impl.md     this session's handoff

## Running the tests

Set the paths first. The DSN itself needs none: since migration step 2
(2026-09-19) it is the in-repo tree `hpc/dsn`, resolved by
`joint/dsn_locate.py`; `DSN_MAIN_DIR` is no longer read and is reported
as ignored if set.

    export SBIX_DIR=~/repos/Sbi-extractor
    export SBI_HPC_DIR=~/repos/Simulation-Based-Inference/hpc

Then, in this order:

    (cd stage1   && python3 smoke_test_latent_sbi.py)        # expect 47 passed
    (cd stage2   && python3 smoke_test_joint.py)             # expect 40 passed, 1 skipped
    (cd stage2   && python3 smoke_test_joint_losses.py)      # expect 27/27
    (cd stage3   && python3 smoke_test_joint_arms.py)        # expect 28 passed
    (cd stage3b  && python3 smoke_test_stage3b.py)           # expect 23 passed
    (cd stage3c  && python3 smoke_test_stage3c.py)           # expect 24 passed
    python3 smoke_test_replicate_statistic.py                # expect 14/14

The one skip is J5 (one host sync per epoch); it needs a CUDA device. Assert it
on the cluster with `torch.cuda.set_sync_debug_mode`.

If `hpc/dsn` is unusable (a broken checkout) the Stage 1 and 3b suites skip
the tests that need the generator and print the resolver's reason; nothing
fails silently.

## Dry runs

Every PBS job takes `DRYRUN=1` and prints the resolved plan plus the expected
output without allocating anything.

    DRYRUN=1 ARM=S OUT_DIR=/tmp/bank PBS_ARRAY_INDEX=0 \
        bash stage1/jobs/build_latent_bank.pbs
    DRYRUN=1 SIM_SHARDS='/path/S/*.npz' OUT_DIR=/tmp/arms PBS_ARRAY_INDEX=0 \
        bash stage3/jobs/joint_arms.pbs
    DRYRUN=1 RUNS_DIR=/tmp/arms bash stage3b/jobs/stage3b.pbs
    DRYRUN=1 CKPT=/tmp/arms/A1_seed0_ckpt.pt SIM_SHARDS='/path/S/*.npz' \
        OUT_DIR=/tmp/s3c bash stage3c/jobs/stage3c.pbs

## Known blockers

`bootstrap_paired.py` is in no repository, so every paired interval degrades
to a point estimate marked incomplete. `feat/joint-dsn-npe` does not exist on
origin, and `feat/misspec-gate` is merged into `main` -- branch off `main`.
Details in the handoff, S3.
