# P1 -- The encoder block: the six searched axes and the fixed knobs of the backbone

**Document P1 of the joint documentation set.** Owner of the `encoder` block
of `JOINT_KNOB_ORDER` (`depth_exponent`, `width_multiplier`, `block_family`,
`embedding_size`, `head_fusion`, `dropout`) and of the `BackboneConfig`
fields the joint stack never sets (`stem_width`, `in_channels`,
`group_width`, `l2_normalize`, `head_pool_ops`, `head_prenorm`,
`norm_target_cpg`, `norm_g_max`, `stem_kernel`, `stem_stride`,
`stage_kernel`, `downsampling_rate`). Master notation: E0. The chapter that
explains what the encoder is *for*: E3. **Date:** 2026-10-01 (v1).
**Applies to:** the repository `Simulation-Based-Inference` at `834eb41`,
`hpc/joint/` and the `hpc/dsn/backbone.py`, `config.py`,
`condition_space.py` it reads (D-037).

| date | change |
|---|---|
| 2026-10-01 | v1. Written from `hpc/dsn/backbone.py` (read in full), `hpc/dsn/config.py` (`SearchConfig`, `RegularizationConfig`, the architecture note), `hpc/dsn/condition_space.py` (the head codes), `hpc/dsn/Documentation/TUNING_1_searched_axes.md` S3.1-3.4, 3.8, 3.15 and `CONFIG_REFERENCE.md` S3.1-3.5, `joint/stage4/joint_space.py`, `joint/stage3/run_joint_arms.py` (`make_backbone`), `joint/stage3/probe_dsn_runtime.py`; the width schedule, stage layouts, output lengths and parameter counts recomputed in the sandbox from a torch-free copy of the backbone's pure helpers `[RAN]`, and the count reconciled with the cluster's 359708 `[KB]`. Finding F-s added. Grounding searches of S6 run and reported. |

**Abstract.** The six encoder axes of the joint search are the only axes
that change what the summary network *is*; everything else in the space
changes how it is trained or what is fitted on top of it. The question this
document answers is, for each of these six and for the twelve knobs held
fixed around them: where the value is set, what the value does to the
network that is built -- its depth, widths, stages, output length,
normalisation, head and parameter count -- what it changes in the objective,
how it interacts with the other blocks, how it fails, and which number in
the run record reveals the failure. **Covered:** the per-parameter fields of
plan S2.2 for all eighteen knobs; the architecture as an explicit function of
the knobs (S3.2), with the stage layouts and parameter counts over the whole
searched range `[RAN]`; the differences between the joint stack's ranges and
defaults and the standalone DSN's (D-036, S3.5); the findings this document
owns (F-b, F-c, F-g, F-m's encoder rows, F-s). **Deliberately excluded:** the
losses that train the encoder (P2), the flow it conditions (P4), the
optimiser (P5), and the pedagogical account of why a learned summary is
needed at all (E3). Nothing here is a measurement of training behaviour:
no job of `hpc/joint/` has run (`[KB]` usage v1.3 S9); every number is a
property of the network as built, read from the code or recomputed from it.

---

## 1. Notation and symbols

A subset of E0's master table, same types and units.

| Symbol | Name / Meaning | Type & domain | Units | First used in S |
|---|---|---|---|---|
| $x$ | one IFR window, the encoder's input | $x \in \mathbb{R}^{W}_{\ge 0}$ | Hz (cohort); counts per bin per unit (bench) | S3.1 |
| $W$ | window length in samples | $\mathbb{N}$; 18000 cohort, 3000 bench | samples | S3.2 |
| $h_\psi$ | the encoder, $h_\psi : \mathbb{R}^{W} \to S^{E-1}$ | map; weights $\psi$ | -- | S3.1 |
| $\psi$ | encoder weights | $\psi \in \mathbb{R}^{n_\psi}$ | -- | S3.2 |
| $n_\psi$ | number of encoder weights (the parameter count) | $\mathbb{N}$ | -- | S3.2 |
| $z$ | the embedding, $z = h_\psi(x)$ | $z \in S^{E-1} \subset \mathbb{R}^{E}$ | dimensionless | S3.1 |
| $\tilde z$ | the head's output before L2 normalisation | $\mathbb{R}^{E}$ | dimensionless | S3.2 |
| $S^{E-1}$ | the unit sphere in $\mathbb{R}^{E}$ | set | -- | S3.1 |
| $E$ | embedding dimension (`embedding_size`) | $\mathbb{N}$; searched in $[8, 16]$ | -- | S3.1 |
| $d_{\rm exp}$ | depth exponent (`depth_exponent`) | $\mathbb{N}$; searched in $\{3, \dots, 6\}$ | -- | S3.2 |
| $B_{\rm blk}$ | number of residual blocks, $2^{d_{\rm exp}}$ | $\mathbb{N}$ | -- | S3.2 |
| $b$ | block index | $b \in \{0, \dots, B_{\rm blk} - 1\}$ | -- | S3.2 |
| $w_{\rm m}$ | width multiplier (`width_multiplier`) | $\mathbb{R}_{>1}$; searched in $[1.5, 3.0]$ | dimensionless | S3.2 |
| $w_0$ | stem width (`stem_width`) | $\mathbb{N}$; 16 | channels | S3.2 |
| $s_b$ | real-valued width exponent of block $b$ | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.2 |
| $w_b$ | width of block $b$ after rounding and group snapping | $\mathbb{N}$ | channels | S3.2 |
| $g_{\rm w}$ | group width (`group_width`) | $\mathbb{N}$; 16 | channels | S3.2 |
| $n_{\rm st}$ | number of stages | $\mathbb{N}$ | -- | S3.2 |
| $s_{\rm tot}$ | total stride, $4 \cdot 2^{n_{\rm st}}$ | $\mathbb{N}$ | samples | S3.2 |
| $n_{\rm out}$ | length of the last feature map, $\lceil W / s_{\rm tot} \rceil$ | $\mathbb{N}$ | samples | S3.2 |
| $R_{\rm f}$ | receptive field of one output sample | $\mathbb{N}$ | samples | S3.2 |
| $G_{\rm gn}$ | GroupNorm groups of a layer | $\mathbb{N}$ | -- | S3.2 |
| $c_{\rm pg}$ | target channels per GroupNorm group (`norm_target_cpg`) | $\mathbb{N}$; 16 | channels | S3.2 |
| $n_{\rm head}$ | input features of the head's linear projection | $\mathbb{N}$ | -- | S3.2 |
| $p_{\rm drop}$ | dropout probability (`dropout`) | $[0, 1)$; searched in $[0, 0.3]$ | dimensionless | S3.2 |
| $r_{\rm eff}$ | participation-ratio effective rank of an embedding cloud | $[1, E]$ | -- | S3.7 |
| $p$ | latent dimension of the bank (the search's `--p` anchor) | $\mathbb{N}$ | -- | S3.6 |
| $n_{\rm hid}$ | hidden features of the flow's conditioner (`hidden_features`) | $\mathbb{N}$ | -- | S3.6 |
| $C$ | number of classes | $\mathbb{N}$ | -- | S3.3 |
| $\bar z_c$ | mean embedding of class $c$ | $\mathbb{R}^{E}$ | dimensionless | S3.3 |
| $c$ | class label | $c \in \{0, \dots, C - 1\}$ | -- | S3.3 |
| $\mathcal{L}^{\rm sim}_{\rm NPE}$ | the NPE term of the objective | $\mathbb{R}$ | nats/row | S3.2 |
| $q_\omega$ | the flow, $q_\omega(\theta \mid z)$ | conditional density | -- | S3.2 |
| $\theta$ | the inference parameters | $\theta \in \Theta$ | mixed | S3.2 |
| $\Theta$ | the prior box | set | -- | S3.2 |
| $\gamma_{\rm wd}$ | the AdamW weight-decay coefficient | $\mathbb{R}_{\ge 0}$ | dimensionless | S3.6 |
| $k_b$ | integer width exponent of block $b$, $k_b = \mathrm{round}(s_b)$ | $\mathbb{N}_0$ | -- | S3.2 |
| $w_b^{\rm raw}$ | the width of block $b$ before rounding and snapping | $\mathbb{R}_{>0}$ | channels | S3.2 |
| $g_{\rm eff}$ | effective group size of a block, $\min(g_{\rm w}, w)$ | $\mathbb{N}$ | channels | S3.2 |
| $w$ | a block's output width when the block index is immaterial: $w_b$ with $b$ dropped (an abuse of notation, used in eq. (P1.6)-(P1.8) only) | $\mathbb{N}$ | channels | S3.2 |
| $w_{\rm in}$ | a block's input width (the previous block's $w$, or $w_0$ for the first) | $\mathbb{N}$ | channels | S3.2 |
| $C_{\rm ch}$ | the channel count of a normalisation layer | $\mathbb{N}$ | channels | S3.2 |
| $n^{\rm res}, n^{\rm rx}, n^{\rm blk}$ | weights of one ResNet block, one ResNeXt block, one block of the chosen family | $\mathbb{N}$ | -- | S3.2 |
| $\mathbb{1}_{\rm proj}, \mathbb{1}_{\rm fusion}$ | indicators: the block carries a projection shortcut; the head fuses all stages | $\{0, 1\}$ | -- | S3.2 |
| $n_{\rm ops}$ | number of pooled statistics per stage, 1 or 3 | $\{1, 3\}$ | -- | S3.2 |
| $\Omega_{\rm st}$ | the set of stage widths | finite set of $\mathbb{N}$ | channels | S3.2 |
| $u_{\rm head}$ | the concatenated pooled vector entering the projection | $\mathbb{R}^{n_{\rm head}}$ | dimensionless | S3.2 |
| $A_{\rm head}, a_{\rm head}$ | weight matrix and bias of the head's linear projection | $\mathbb{R}^{E \times n_{\rm head}}$, $\mathbb{R}^{E}$ | dimensionless | S3.2 |
| $\rho_{\rm grad}$ | cosine between the NPE and DSN gradients with respect to $\psi$ | $[-1, 1]$ | -- | S3.3 |
| $\hat\Delta$, $\delta_{\min}$ | information gain; the shuffled-control floor | $\mathbb{R}$ | nats/row | S3.7 |

### 1.1 Conventions

- Code names in backticks are the knobs as the code spells them; the symbol
  of the same quantity is used only where a formula needs it. `depth_exponent`
  is $d_{\rm exp}$, not E0's $d$ (degrees of freedom), and TUNING_1's $d$,
  $w$ for the same two knobs are not used here.
- **Status** (the Provenance model): every knob of this document is
  `configured`; the quantities derived from them -- $w_b$, $n_{\rm st}$,
  $n_{\rm out}$, $G_{\rm gn}$, $n_\psi$ -- are `computed` (deterministic
  functions of the configuration and of $W$). $W$ itself is `measured` from
  the bank's sidecar.
- **Default** is per surface, as P0 S1.1: the runner's flag default is what a
  Stage 3 or Stage 4 run trains with; `BackboneConfig`'s default is what the
  standalone DSN or a direct caller gets; the runner passes its own values
  explicitly for the six axes and `stem_width`, and leaves every other
  `BackboneConfig` field at the dataclass default (S3.1).
- Equations are numbered (P1.n); plan equations are cited as "plan eq. (n)".
- "Fixed" has two senses here and both are used: *fixed by plan S5.1* (a knob
  the space declares at one value and carries in `spec.fixed`:
  `head_pool_ops`) and *fixed by omission* (a `BackboneConfig` field the
  runner never sets). S3.4 says which is which.

## 2. Glossary

Ordered by first appearance.

- **Backbone** -- the DSN's `OneDCNNBackbone`: stem, stages of residual
  blocks, head; the object `make_backbone` builds and the encoder $h_\psi$ of
  every arm except `A_ref`. S3.1.
- **Stem** -- the first strided convolution (kernel `stem_kernel`, stride
  `stem_stride`), from `in_channels` to $w_0$ channels, followed by GroupNorm
  and ReLU. S3.2.
- **Width schedule** -- the RegNet-style rule that assigns every block a
  width from $w_0$, $w_{\rm m}$ and its index, eq. (P1.1)-(P1.3). *Everyday
  meaning differs*: "width" is a channel count, not a kernel size. S3.2.
- **Stage** -- a run of consecutive blocks with the same width; the first
  block of each stage halves the length (`downsampling_rate`). The number of
  stages is emergent, not configured. S3.2.
- **Residual block** -- two convolutions with a skip connection, ResNet
  (`block_family` 0) or ResNeXt (`block_family` 1); the ResNeXt variant is a
  1-3-1 block whose middle convolution is grouped with $g_{\rm w}$ channels
  per group. S3.2.
- **Grouped convolution** -- a convolution whose input and output channels are
  split into groups convolved independently; parameters scale with the group
  width, not the layer width. S3.2.
- **GroupNorm** -- normalisation of each sample's activations over groups of
  channels and the whole length, so that training and evaluation use the
  same statistics and no batch statistic exists. S3.2.
- **Head** -- the pooling over length (mean, or mean/max/std), the optional
  per-stage LayerNorm, the concatenation of the selected stages, the linear
  projection to $E$ and the L2 normalisation. S3.2.
- **Head fusion** -- whether the head reads only the last stage
  (`head_fusion` 0) or concatenates the pooled statistics of every stage (1).
  S3.3.
- **Receptive field** -- the span of input samples one output sample depends
  on. S3.2.
- **Parameter count $n_\psi$** -- the number of trainable scalars of the
  backbone; the cost driver of the search. S3.2.
- **Fallback backbone** -- the small GroupNorm CNN `make_backbone` builds
  when the DSN tree is not found; it ignores the six axes. S3.1.
- **Effective rank $r_{\rm eff}$** -- the participation-ratio rank of the
  embedding cloud on the report split, recorded by the runner. S3.7.
- **Differences table** -- the D-036 table that sets the joint stack's range
  and default of each inherited axis against the standalone DSN's. S3.5.

---

## 3. Main body

### 3.1 Where the encoder's knobs live, and what the search can reach

This section establishes which of the eighteen knobs a Stage 4 trial can
move, which a Stage 3 arm can move, and which nothing moves.

One would expect the summary network's architecture to be set in one place.
It is set in three: six knobs travel as search axes through the runner's
flags into `make_backbone`, one (`stem_width`) is hard-coded in
`make_backbone`, and eleven are never named by the joint stack at all and
take `BackboneConfig`'s own defaults `[REPO]` `run_joint_arms.py:282-308`,
`dsn/backbone.py:56-90`. Regardless of what the ledger records, if a knob is
not in the `BackboneConfig(...)` call of `make_backbone`, its value is the
dataclass default -- which is how `head_pool_ops`, declared fixed at code 1
by the space, is built at code 0 (F-b).

| knob | surface(s) | runner default / what the runner passes | `BackboneConfig` default | searched range (joint) | status |
|---|---|---|---|---|---|
| `depth_exponent` | axis; `--depth-exponent`; `BackboneConfig.depth_exponent` | `3` | `4` | $\{3, \dots, 6\}$, integer uniform | configured |
| `width_multiplier` | axis; `--width-multiplier`; dataclass field | `2.0` | `2.0` | $[1.5, 3.0]$, real uniform | configured |
| `block_family` | axis; `--block-family`; dataclass field | `0` | `0` | $\{0, 1\}$, categorical | configured |
| `embedding_size` | axis; `--embedding-size`; `make_backbone(E)` | `10` | `16` | $[8, 16]$, integer uniform | configured |
| `head_fusion` | axis; `--head-fusion` (0/1); dataclass field (bool) | `0` | `False` | $\{0, 1\}$, categorical | configured |
| `dropout` | axis; `--dropout`; dataclass field | `0.0` | `0.0` | $[0.0, 0.3]$, real uniform | configured |
| `stem_width` | hard-coded in `make_backbone` | `16` | `16` | not searched | configured (by code) |
| `head_pool_ops` | `spec.fixed` = 1, never passed; dataclass field | not passed: built at `("mean",)` | `("mean",)` | fixed at code 1 = `("mean", "max", "std")` | configured; **F-b** |
| `in_channels` | dataclass field only | -- | `1` | -- | configured (by omission) |
| `group_width` | dataclass field only | -- | `16` | -- | configured (by omission) |
| `l2_normalize` | dataclass field only | -- | `True` | -- | configured (by omission) |
| `head_prenorm` | dataclass field only | -- | `True` | -- | configured (by omission) |
| `norm_target_cpg`, `norm_g_max` | dataclass fields only | -- | `16`, `32` | -- | configured (by omission) |
| `stem_kernel`, `stem_stride` | dataclass fields only | -- | `5`, `4` | -- | configured (by omission) |
| `stage_kernel`, `downsampling_rate` | dataclass fields only | -- | `3`, `2` | -- | configured (by omission) |

Ranges and priors are P0 Table A's `[RAN]`; defaults are `[REPO]`
`run_joint_arms.py:230, 236-242`, `dsn/backbone.py:56-90`. The six axes are
*always active*: no legality projection touches them (the DSN's
`project_condition` acts on the loss triple only, `dsn/condition_space.py`),
and every campaign searches all six (P0 Table A).

What the runner does with them (`make_backbone`, `run_joint_arms.py:282-308`):
it resolves the DSN tree, imports `backbone.py`, and builds

```
BackboneConfig(depth_exponent, width_multiplier, block_family,
               head_fusion=bool(int(head_fusion)), dropout,
               stem_width=16, embedding_size=E)
```

with every other field at its default. If the tree is not found it builds
`SmallBackbone`, prints a `[warn]`, and the record's `backbone` field reads
`"fallback"` instead of `"dsn"` (`:308-333`, `:468`, `:584`): the six axes
are then ignored and a search over them means nothing. Arm `A_ref` builds
`FixedStatsSummary` instead (`E = 8`, no knob of this document applies).

### 3.2 The architecture as a function of the knobs

This section establishes what network the six axes and the fixed knobs
build -- the load-bearing content, because every "impact" statement in S3.3
is a statement about these formulas.

**The stem.** Input $x \in \mathbb{R}^{W}$ is read as one channel
(`in_channels` 1); a convolution of kernel 5 and stride 4 (no bias) takes it
to $w_0 = 16$ channels, then GroupNorm and ReLU. Asymmetric zero padding
makes the output length $\lceil W / 4 \rceil$ (`backbone.py:190-231`). On the
cohort that is $4500$ samples, on the bench $750$.

**The width schedule, eq. (P1.1)-(P1.3).** The backbone stacks
$B_{\rm blk} = 2^{d_{\rm exp}}$ residual blocks. Block $b$ gets the
real-valued exponent and width

$$ s_b = \frac{\ln(b + 1)}{\ln w_{\rm m}}, \qquad k_b = \mathrm{round}(s_b), \qquad w_b^{\rm raw} = w_0 \, w_{\rm m}^{k_b}, \tag{P1.1} $$

the RegNet quantised-linear rule in the simplified variant where the initial
width and the slope coincide, $w_0 = 16$ (`compute_block_widths`,
`backbone.py:156-169`). The raw width is rounded to an integer and snapped to
a multiple of the group width,

$$ w_b = \max\big(g_{\rm eff}, \; g_{\rm eff} \cdot \mathrm{round}(\mathrm{round}(w_b^{\rm raw}) / g_{\rm eff})\big), \qquad g_{\rm eff} = \min(g_{\rm w}, \mathrm{round}(w_b^{\rm raw})), \tag{P1.2} $$

so that a grouped convolution with $g_{\rm w} = 16$ channels per group
always divides the layer (`adjust_width_to_group`, `:132-141`). Consecutive
equal widths are grouped into stages,

$$ n_{\rm st} = \#\{\text{distinct consecutive values among } w_0, \dots, w_{B_{\rm blk} - 1}\}, \tag{P1.3} $$

and each stage's first block carries stride 2 (`allocate_stages`,
`:172-187`; `OneDCNNBackbone.__init__`, `:403-426`). The stage count is
therefore *emergent*: it is set by how often $\mathrm{round}(s_b)$ changes
over $b$, which depends on $w_{\rm m}$ and $B_{\rm blk}$ and on nothing else.
Two consequences follow. Regardless of $d_{\rm exp}$, if $w_{\rm m}$ is
large then $s_b$ grows slowly in $b$ and few stages are formed, each with
many blocks; conversely if $w_{\rm m}$ is close to 1 then $s_b$ grows fast
and many narrow stages are formed. And because every stage halves the
length, the number of stages fixes the total stride and the output length:

$$ s_{\rm tot} = 4 \cdot 2^{n_{\rm st}}, \qquad n_{\rm out} = \lceil W / s_{\rm tot} \rceil. \tag{P1.4} $$

The layouts over the searched grid `[RAN]` (the torch-free replica of the
three helpers; `tools/` does not ship it, it is three functions copied from
`backbone.py`):

| $d_{\rm exp}$ | $w_{\rm m}$ | $n_{\rm st}$ | stage widths | blocks per stage | $n_{\rm out}$ at $W = 18000$ | at $W = 3000$ |
|---|---|---|---|---|---|---|
| 3 | 1.5 | 5 | 16, 32, 48, 80, 128 | 1, 1, 2, 2, 2 | 141 | 24 |
| 3 | 2.0 | 4 | 16, 32, 64, 128 | 1, 1, 3, 3 | 282 | 47 |
| 3 | 3.0 | 3 | 16, 48, 144 | 1, 4, 3 | 563 | 94 |
| 4 | 1.5 | 7 | 16, 32, 48, 80, 128, 176, 272 | 1, 1, 2, 2, 3, 4, 3 | 36 | 6 |
| 4 | 2.0 | 5 | 16, 32, 64, 128, 256 | 1, 1, 3, 6, 5 | 141 | 24 |
| 4 | 3.0 | 4 | 16, 48, 144, 432 | 1, 4, 10, 1 | 282 | 47 |
| 5 | 1.5 | 9 | 16, ..., 416, 608 | 1, 1, 2, 2, 3, 4, 7, 11, 1 | 9 | 2 |
| 5 | 2.0 | 6 | 16, 32, 64, 128, 256, 512 | 1, 1, 3, 6, 11, 10 | 71 | 12 |
| 5 | 3.0 | 4 | 16, 48, 144, 432 | 1, 4, 10, 17 | 282 | 47 |
| 6 | 1.5 | 10 | 16, ..., 608, 928 | 1, 1, 2, 2, 3, 4, 7, 11, 16, 17 | 5 | 1 |
| 6 | 2.0 | 7 | 16, ..., 512, 1024 | 1, 1, 3, 6, 11, 23, 19 | 36 | 6 |
| 6 | 3.0 | 5 | 16, 48, 144, 432, 1296 | 1, 4, 10, 31, 18 | 141 | 24 |

The runner's default row is $(3, 2.0)$: four stages of widths 16, 32, 64,
128 with 1, 1, 3, 3 blocks, $s_{\rm tot} = 64$, $n_{\rm out} = 282$ on the
cohort. The snapping of eq. (P1.2) is visible at $w_{\rm m} = 1.5$: the raw
widths $16 \cdot 1.5^k$ (24, 36, 54, 81, ...) become 32, 32, 48, 80, ..., so
the effective widths are multiples of 16 whatever $w_{\rm m}$ is, and
$w_{\rm m}$ acts on the *allocation* of blocks to widths more than on the
widths themselves.

**Normalisation, eq. (P1.5).** Every normalisation layer is a GroupNorm
whose group count is the largest divisor of the channel count $C_{\rm ch}$
not above $\min(\lfloor C_{\rm ch} / c_{\rm pg} \rfloor, 32)$, the 32 being
`norm_g_max` (`pick_G`, `:144-153`):

$$ G_{\rm gn}(C_{\rm ch}) = \max\{G \le \min(\lfloor C_{\rm ch}/c_{\rm pg} \rfloor, 32) : G \text{ divides } C_{\rm ch}\}, \quad G_{\rm gn} = 1 \text{ if none}. \tag{P1.5} $$

At the widths the schedule produces, $G_{\rm gn}$ is 1, 2, 3, 4, 6, 8, 16,
32, 32 at 16, 32, 48, 64, 96, 128, 256, 512, 1024 channels, and 27 at 432
and at 1296 `[RAN]`. A 16-channel layer therefore gets one group -- a
LayerNorm over channels and length -- and every layer normalises over the
whole length of the feature map and never over the batch: the backbone has
no batch statistic, so the forward pass in training and in evaluation is the
same function, and $B_{\rm met} = 32$ or the replicate stream's 4 pairs per
step (P0 Table B) carry no normalisation cost. The group count is a function
of the architecture, not a knob: `norm_target_cpg` and `norm_g_max` are the
two constants it is computed from.

**The blocks, eq. (P1.6)-(P1.7).** With kernel 3 (`stage_kernel`) and
input width $w_{\rm in}$, output width $w$:

- ResNet (`block_family` 0, `:235-278`): conv($w_{\rm in} \to w$, stride
  2 on the first block of a stage) + GN + ReLU, conv($w \to w$) + GN, a
  $1 \times 1$ projection shortcut with GN whenever the stride or the width
  changes, add, ReLU, dropout. Weights, with no biases on the convolutions:
  $$ n^{\rm res}(w_{\rm in}, w) = 3 w_{\rm in} w + 3 w^2 + 4 w + \mathbb{1}_{\rm proj}\,(w_{\rm in} w + 2 w), \tag{P1.6} $$
  with $\mathbb{1}_{\rm proj} = 1$ when the block carries the projection shortcut and 0 otherwise.
- ResNeXt (`block_family` 1, `:281-333`): conv $1 \times 1$ ($w_{\rm in} \to w$) +
  GN + ReLU, grouped conv kernel 3 ($w \to w$ with $g_{\rm eff} = \min(g_{\rm w}, w)$
  channels per group) + GN + ReLU, conv $1 \times 1$ ($w \to w$) + GN, the
  same shortcut rule, add, ReLU, dropout. No bottleneck is added (the
  three convolutions keep width $w$):
  $$ n^{\rm rx}(w_{\rm in}, w) = w_{\rm in} w + 3 g_{\rm eff} w + w^2 + 6 w + \mathbb{1}_{\rm proj}\,(w_{\rm in} w + 2 w). \tag{P1.7} $$

The last GroupNorm of each residual branch is initialised with zero gain, so
every block starts as the identity plus the shortcut (`_init_weights`,
`:428-446`); Kaiming initialisation elsewhere. The ResNeXt block's grouped
convolution is the split-transform-merge of the ResNeXt family -- each group
transforms a low-dimensional slice and the slices are summed -- which is
what makes its parameter count scale with $g_{\rm w}$ rather than with $w$
`[PubMed full text]` (Wu et al. 2020, Sensors, applied description of the
block, [DOI](https://doi.org/10.3390/s20061652)); at $w = 16$ the group
count is 1 and the "grouped" convolution is a plain one.

**The head, eq. (P1.8)-(P1.9).** Each selected stage output $(w, n)$ is
pooled over length by the chosen statistics (mean; or mean, max and standard
deviation with $\sqrt{\mathrm{var} + 10^{-5}}$), the pooled vectors are
LayerNorm-ed per stage when `head_fusion` is on and `head_prenorm` is `True`,
concatenated, and projected by one linear layer with bias
(`MultiScaleHead`, `:352-397`):

$$ n_{\rm head} = n_{\rm ops}\, w_{B_{\rm blk} - 1} \ \ (\text{`head_fusion` 0}), \qquad n_{\rm head} = n_{\rm ops} \sum_{w \in \Omega_{\rm st}} w \ \ (\text{`head_fusion` 1}), \qquad \tilde z = A_{\rm head}\, u_{\rm head} + a_{\rm head}, \tag{P1.8} $$

with $\Omega_{\rm st}$ the set of stage widths, $u_{\rm head} \in \mathbb{R}^{n_{\rm head}}$
the concatenated pooled vector, $A_{\rm head} \in \mathbb{R}^{E \times n_{\rm head}}$
and $a_{\rm head} \in \mathbb{R}^{E}$ the projection, $n_{\rm ops} \in \{1, 3\}$;
and, with `l2_normalize` `True`,

$$ z = \tilde z / \lVert \tilde z \rVert_2 \in S^{E-1}. \tag{P1.9} $$

At the runner's defaults $n_{\rm head} = 128$; with `head_fusion` 1 it is
$16 + 32 + 64 + 128 = 240$; with the three statistics 384 and 720
respectively `[RAN]`. The projection is the only place the six axes meet
$E$: $E$ sets the number of output rows of $A_{\rm head}$ and nothing else in
the backbone.

**Dropout.** `nn.Dropout(p_drop)` is applied to the output of every residual
block, after the final ReLU (`:262, :278`; `:316, :333`), and to nothing
else: not the stem, not the head. It is active in training mode only; the
runner's diagnostics and the posterior draws run under `torch.no_grad()` on a
model in whatever mode the loop left it (P5 states which).

**The parameter count, eq. (P1.10).** Summing eq. (P1.6) or (P1.7) over the
layout of eq. (P1.1)-(P1.3), with the stem ($5 w_0 + 2 w_0 = 112$) and the
head ($E (n_{\rm head} + 1)$, plus $2 n_{\rm head}$ of LayerNorm when fused):

$$ n_\psi = 112 + \sum_{b} n^{\rm blk}(w_{\rm in}(b), w_b) + E\,(n_{\rm head} + 1) + \mathbb{1}_{\rm fusion}\, 2 n_{\rm head}, \tag{P1.10} $$

with $n^{\rm blk}$ the count of eq. (P1.6) or (P1.7) by family and
$\mathbb{1}_{\rm fusion} = 1$ when `head_fusion` is on.

Evaluated in the sandbox `[RAN]` and reconciled with the cluster: at the
runner's defaults with $E = 12$ -- the configuration `probe_dsn_runtime.py`
builds (`:250-257`, `--embedding-size` default 12) -- eq. (P1.10) gives
**359708**, the number the probe reported on davinci `[KB]` (usage v1.3 S8;
`HPC_PATHS.md` 5d); at the runner's own $E = 10$ it is 359450 and at
$E = 16$, 360224. So the cluster's "359708 parameters" is the $E = 12$ count,
and 78 % of it (280320) sits in the last stage. Over the searched grid
(ResNet, last-stage head, mean pooling, $E = 10$; the other head and family
variants in S3.3):

| $d_{\rm exp}$ | $w_{\rm m} = 1.5$ | $2.0$ | $3.0$ |
|---|---|---|---|
| 3 | 0.30 M | 0.36 M | 0.40 M |
| 4 | 2.45 M | 2.56 M | 2.09 M |
| 5 | 17.7 M | 20.4 M | 20.0 M |
| 6 | 139 M | 159 M | 214 M |

Three readings. The count grows roughly eightfold per unit of $d_{\rm exp}$
-- depth doubles the block count and the widest stage widens -- so the search
spans three orders of magnitude, from 0.14 M (ResNeXt, $d_{\rm exp} = 3$,
$w_{\rm m} = 1.5$) to 214.5 M (ResNet, $d_{\rm exp} = 6$, $w_{\rm m} = 3.0$,
fused three-statistic head, $E = 12$) `[RAN]`. The count is *not monotone* in
$w_{\rm m}$ at fixed depth ($d_{\rm exp} = 4$: 2.56 M at 2.0, 2.09 M at 3.0),
because a larger multiplier makes fewer, longer stages and moves blocks out
of the widest one. And the ResNeXt family is 2.1 to 2.9 times cheaper than
ResNet at every corner `[RAN]`, because its kernel-3 convolution costs
$3 g_{\rm eff} w$ instead of $3 w^2$.

**The receptive field.** With the fixed kernels and strides, one output
sample of the last stage sees $R_{\rm f} = 1133$ input samples at the runner's
defaults (11.3 s at 100 Hz on the cohort, 22.7 s at 50 Hz on the bench) and
$18477$ at $(6, 3.0)$ -- the whole 180 s window `[RAN]`. The mean pooling of
the head then averages $n_{\rm out}$ such samples, so the embedding is a
window-wide summary at every corner; what the corner changes is how many
distinct positions are averaged (eq. (P1.4)).

### 3.3 The six searched axes

This section establishes, for each axis, the fields of plan S2.2. The source
tag of every row is `[REPO]` unless written otherwise; the analytic effect is
eq. (P1.1)-(P1.10) instantiated.

#### 3.3.1 `depth_exponent` ($d_{\rm exp}$)

- **Where it lives.** Axis 1 of `JOINT_KNOB_ORDER` (`joint_space.py:92-105`);
  `--depth-exponent` (`run_joint_arms.py:236`); `BackboneConfig.depth_exponent`
  (`backbone.py:58`); no job variable.
- **Type and domain.** Integer, $\ge 1$ (`__post_init__`); searched in
  $\{3, 4, 5, 6\}$, integer uniform prior (P0 Table A).
- **Defaults.** Runner `3`; `BackboneConfig` `4` (F-m). The standalone DSN's
  JSON configuration searches $[2, 5]$ (`[REPO]` TUNING_1 S3.1); the joint
  space copies the dataclass range $(3, 6)$ (`RANGE_PROVENANCE`, F-g).
- **Range provenance.** `DSN config.SearchConfig.depth_exponent_range`
  (`config.py:893`) -- correct.
- **Status.** Configured; derived $B_{\rm blk}$, $n_{\rm st}$, $n_\psi$
  computed.
- **What it represents, plainly.** How many residual blocks the network
  stacks: 8, 16, 32 or 64. More blocks mean more stages (eq. (P1.3)), a
  wider last stage, a shorter output map and a much larger parameter count.
- **What it changes, analytically.** $B_{\rm blk} = 2^{d_{\rm exp}}$ enters
  eq. (P1.1)-(P1.4) and (P1.10): at $w_{\rm m} = 2.0$, $n_{\rm st}$ goes 4, 5,
  6, 7 and the last width 128, 256, 512, 1024 as $d_{\rm exp}$ goes 3 to 6;
  $n_{\rm out}$ on the cohort goes 282, 141, 71, 36; $n_\psi$ goes 0.36 M,
  2.6 M, 20 M, 159 M `[RAN]`. In the objective, $d_{\rm exp}$ changes the
  capacity of $h_\psi$ and nothing in $q_\omega$: the NPE term
  $\mathcal{L}^{\rm sim}_{\rm NPE}$ can only improve with capacity on the
  training split, and whether it improves on the selection split is the
  search's question.
- **Legality and interactions.** Always active. Multiplicative with
  $w_{\rm m}$ and `head_fusion` in cost; with $W$ in the output length
  (eq. (P1.4)): at $d_{\rm exp} = 6$ and $w_{\rm m} = 1.5$ the last map has 5
  samples on the cohort and **1 on the bench** (S3.7). With `patience` 99 and
  `epochs` 10 (P0 Table C) no early stopping can compensate a model that is
  too large to converge in 250 steps.
- **Failure modes and the diagnostic.** Budgetary and silent: every value
  builds, and the wall clock scales with the mean of a right-skewed size mix
  (TUNING_1 S3.1's argument, `[REPO]` doc). The joint space includes
  $d_{\rm exp} = 6$, the corner TUNING_1 S3.1 says not to search without
  re-running its budget gate, and whose largest model is 215 M parameters
  (**F-s**). Diagnostics: no parameter count is recorded by the runner (the
  record carries `backbone`, `r_eff`, `L`, ...; `run_joint_arms.py:580-600`) --
  the count has to be recomputed from the configuration with eq. (P1.10);
  the job's walltime (`06:00:00`, P0 Table E) is the only guard.

#### 3.3.2 `width_multiplier` ($w_{\rm m}$)

- **Where it lives.** Axis 2; `--width-multiplier` (`:237`);
  `BackboneConfig.width_multiplier` (`:59`).
- **Type and domain.** Real, $> 1$ strictly (`__post_init__`); searched in
  $[1.5, 3.0]$, uniform prior.
- **Defaults.** Runner `2.0`; dataclass `2.0`. The standalone DSN's JSON
  searches $[1.5, 5.0]$ (TUNING_1 S3.2); the joint space takes the dataclass
  $(1.5, 3.0)$ (F-g).
- **Range provenance.** `SearchConfig.width_multiplier_range` (`config.py:894`)
  -- correct.
- **Status.** Configured.
- **What it represents, plainly.** How fast the channel count grows along the
  network: the base of the width schedule. A larger multiplier makes fewer,
  wider stages with more blocks each; a smaller one makes many thin stages.
- **What it changes, analytically.** Through $s_b$ of eq. (P1.1) it sets
  $n_{\rm st}$ (eq. (P1.3)) and hence $s_{\rm tot}$ and $n_{\rm out}$
  (eq. (P1.4)): at $d_{\rm exp} = 3$, $n_{\rm st}$ is 5, 4, 3 and
  $n_{\rm out}$ 141, 282, 563 for $w_{\rm m}$ = 1.5, 2.0, 3.0 `[RAN]`. The
  widths it produces are snapped to multiples of 16 (eq. (P1.2)), so its
  effect is quantised; the parameter count is non-monotone in it at fixed
  depth (S3.2).
- **Legality and interactions.** Always active; `__post_init__` refuses
  $w_{\rm m} \le 1$, and the lower bound 1.5 keeps the search clear of it.
  Interacts with $d_{\rm exp}$ in both cost and output length.
- **Failure modes and the diagnostic.** None internal; the budget failure of
  S3.3.1 is shared. A uniform prior on $[1.5, 3.0]$ puts a third of the mass
  on the region of fewest stages (largest $n_{\rm out}$, widest last stage),
  which at $d_{\rm exp} = 6$ is the 214 M corner.

#### 3.3.3 `block_family`

- **Where it lives.** Axis 3; `--block-family` (`:238`, choices 0/1);
  `BackboneConfig.block_family` (`:69`).
- **Type and domain.** Integer in $\{0, 1\}$; searched as a categorical (P0
  Table A), one surrogate column.
- **Defaults.** Runner `0`; dataclass `0`.
- **Range provenance.** `SearchConfig.block_family_choices` (`config.py:895`).
- **Status.** Configured.
- **What it represents, plainly.** 0 builds ResNet blocks (two kernel-3
  convolutions), 1 builds ResNeXt blocks (1-3-1 with the middle convolution
  grouped, 16 channels per group).
- **What it changes, analytically.** Eq. (P1.6) against (P1.7): at every
  corner the ResNeXt network has 2.1 to 2.9 times fewer parameters `[RAN]`
  (0.16 M against 0.36 M at the defaults), one more normalisation and one
  more nonlinearity per block, and the same layout, output length and
  receptive field (the layout of eq. (P1.1)-(P1.4) does not read the family).
- **Legality and interactions.** Always active. The group width
  $g_{\rm w} = 16$ is what the ResNeXt block divides by; eq. (P1.2) snaps
  every width to a multiple of 16 for *both* families, so the snapping acts
  even when the ResNet family is chosen.
- **Failure modes and the diagnostic.** None internal; a sampled 0.37 cannot
  happen because the axis is integer (TUNING_1 S3.3's caveat on NumPy
  integer scalars applies to the standalone driver; the joint driver's
  `_INT_AXES` casts to `int` before `build_argv`, `npe_tune_joint.py:200-203`).

#### 3.3.4 `embedding_size` ($E$)

- **Where it lives.** Axis 4; `--embedding-size` (`:230`); the `E` argument of
  `make_backbone` and `BackboneConfig.embedding_size` (`:73`); the flow's
  conditioning dimension through `build_joint_model` (P4).
- **Type and domain.** Integer $\ge 1$; searched in $[8, 16]$, integer uniform.
- **Defaults.** Runner `10`; dataclass `16` (F-m); the r2 encoder of the
  earlier pipeline had `E = 10` `[KB]` (deck 09); the search's shape anchor
  `--embedding-dim` is `12` (P0 Table D) -- the anchor resolves the flow's
  `hidden_features` range, it is not the value trained.
- **Range provenance.** `SearchConfig.embedding_size_range` (`config.py:896`).
- **Status.** Configured.
- **What it represents, plainly.** The dimension of the summary the flow
  conditions on: how many numbers describe a window.
- **What it changes, analytically.** Only the output rows of $A_{\rm head}$
  in eq. (P1.8) (a few hundred weights: 359450 to 360224 from $E = 10$ to 16
  `[RAN]`) -- and everything downstream: $z \in S^{E-1}$ is the flow's
  condition, so $E$ is the input width of every conditioner of $q_\omega$
  (P4), and the width rule of the search resolves `hidden_features` from
  $\max(p, E)$ (P0 S3.2; at $p = 26$ the anchor $E = 12$ is inert in that
  rule). On the sphere, $E$ bounds the effective rank: $r_{\rm eff} \le E$.
- **Legality and interactions.** Always active. With `head_pool_ops` at the
  three-statistic level the projection compresses three times more features
  into the same $E$ (TUNING_1 S3.4); with the mean-only head the stack
  actually builds (F-b) it compresses $w_{B_{\rm blk} - 1}$ features. The
  geometric floor for $C$ class means on a simplex ETF is $E \ge C - 1$
  (TUNING_1 S3.4) -- 1 at $C = 2$, 2 at $C = 3$ -- never binding at
  $E \ge 8$.
- **Failure modes and the diagnostic.** Collapse of the cloud onto a line or
  a few directions, which $E$ cannot prevent and which `r_eff` in the record
  measures (1.0 is a rank-one cloud; E3). An $E$ larger than the intrinsic
  dimension of the latent buys capacity to memorise (TUNING_1 S3.4's
  argument at $p = 6$; at $p = 26$ the whole range is below the latent
  dimension).

#### 3.3.5 `head_fusion`

- **Where it lives.** Axis 5; `--head-fusion` (`:240`, choices 0/1, cast to
  `bool` in `make_backbone`); `BackboneConfig.head_fusion` (`:75`).
- **Type and domain.** Integer in $\{0, 1\}$ on the surfaces, `bool` in the
  dataclass; searched as a categorical.
- **Defaults.** Runner `0`; dataclass `False` -- one value, two encodings
  (P0 Table F).
- **Range provenance.** `SearchConfig.head_fusion_choices` (`config.py:1014`).
- **Status.** Configured.
- **What it represents, plainly.** Whether the embedding is read from the
  last stage alone or from every stage, so that fine early-stage features
  (short time scales, fewer channels) reach the projection beside the
  coarse late ones.
- **What it changes, analytically.** $n_{\rm head}$ of eq. (P1.8): the sum of
  all stage widths instead of the last; a LayerNorm per stage (`head_prenorm`
  `True`) that equalises the scales of the concatenated parts at
  initialisation; a few hundred to a few thousand more weights (361050
  against 359450 at the defaults `[RAN]`). It does not change the layout or
  the cost of the body.
- **Legality and interactions.** Always active; the DSN's factorial treats
  it with `head_pool_ops` as a $2 \times 2$ head geometry (`condition_space.py`
  `cell_name`, `:294-312`), and the joint space searches one factor and
  declares the other fixed (F-b, S3.4). Interacts with $E$ through the
  compression ratio $n_{\rm head} / E$.
- **Failure modes and the diagnostic.** With `head_fusion` 1 the early stages'
  pooled means (over 4500 and 2250 samples at the defaults) enter the
  embedding: a window-mean of a shallow feature map is close to a rate
  statistic, which is the thing a learned summary is meant to improve on
  (E3); whether it helps is what the search measures. TUNING_1 S3.15
  reports the head geometries as differing chiefly in generalisation gap
  (`[REPO]` doc; a screening finding of the standalone DSN, not of this
  stack).

#### 3.3.6 `dropout` ($p_{\rm drop}$)

- **Where it lives.** Axis 6; `--dropout` (`:242`); `BackboneConfig.dropout`
  (`:90`); applied as `nn.Dropout` after every residual block.
- **Type and domain.** Real in $[0, 1)$; searched in $[0.0, 0.3]$, uniform
  prior.
- **Defaults.** Runner `0.0`; dataclass `0.0`; the standalone DSN pins it to 0
  through its architecture and loss phases and searches it last
  (CONFIG_REFERENCE S3.5, TUNING_1 S3.8).
- **Range provenance.** `RANGE_PROVENANCE` names
  `DSN config.SearchConfig.dropout_range`; `SearchConfig` has no such field --
  the range $(0.0, 0.3)$ is `RegularizationConfig.dropout_range`
  (`config.py:1183`): **F-c**, a wrong provenance string, right value.
- **Status.** Configured.
- **What it represents, plainly.** The probability that each activation of a
  block's output is zeroed during training (and the survivors rescaled), so
  that no feature can rely on a particular other one being present.
- **What it changes, analytically.** The training-time forward pass of
  $h_\psi$ becomes stochastic; the gradient of every term that reads $z$ --
  the NPE term, the DSN term, the replicate term -- is taken through the
  dropped network; evaluation uses the full network. It changes no count and
  no layout.
- **Legality and interactions.** Always active. With `weight_decay` (P5) it is
  the second regulariser of $\psi$; the DSN's staged design tuned the two
  together after the architecture (TUNING_1 S3.8), the joint space frees both
  from trial 0. With the replicate term it interacts through the posterior
  draws: if the model is in training mode when the draws are taken, the
  dropout noise enters $\hat m_g$ and $\hat C_g$ (P3, P5 state the mode).
- **Failure modes and the diagnostic.** At $p_{\rm drop}$ near 0.3 on a
  32-row metric batch and a 4-pair replicate batch, the per-step noise of
  the two real-domain terms grows; the diagnostic is the gradient-cosine
  probe $\rho_{\rm grad}$ and the validation curve, both in the record
  (`history`, `val_npe`). Zero is reachable and legitimate (TUNING_1 S3.8).

### 3.4 The twelve fixed knobs

This section establishes what is held still around the six axes, and which
of the twelve is held still *wrongly*.

| knob | value built | set where | what it does | changing it would |
|---|---|---|---|---|
| `stem_width` ($w_0$) | 16 | `make_backbone`, hard-coded | the first width and the base of eq. (P1.1); every width is a multiple of it | rescale every width; the only knob that changes widths without changing the allocation |
| `head_pool_ops` | `("mean",)` (code 0) | dataclass default; `spec.fixed` says code 1 | the pooled statistics per stage, eq. (P1.8); `n_ops` 1 or 3 | triple $n_{\rm head}$; **F-b**: the ledger of every Stage 4 trial records code 1, the network trains at code 0; D-038 patch |
| `in_channels` | 1 | dataclass default | the stem's input channels; 1 = one pooled IFR per window | a multi-channel window (the DSN's multichannel mode, `MULTICHANNEL_TECHNICAL_DOCUMENT.md`) -- the joint bank's `x` is one channel, so inert |
| `group_width` ($g_{\rm w}$) | 16 | dataclass default | the ResNeXt group width and the snapping unit of eq. (P1.2) | change the ResNeXt cost ($3 g w$) and the quantisation of every width, for both families |
| `l2_normalize` | `True` | dataclass default | eq. (P1.9): $z$ on the sphere | `False` would hand the flow an unnormalised $\tilde z$ and break the cosine geometry of the DSN losses (P2) |
| `head_prenorm` | `True` | dataclass default | per-stage LayerNorm before concatenation, fusion only | inert at `head_fusion` 0 |
| `norm_target_cpg` ($c_{\rm pg}$), `norm_g_max` | 16, 32 | dataclass defaults | eq. (P1.5): the GroupNorm group count | change $G_{\rm gn}$ at every layer; $c_{\rm pg}$ in $\{4, 8, 16, 24, 32\}$ is the intended set (CONFIG_REFERENCE S3.4) |
| `stem_kernel`, `stem_stride` | 5, 4 | dataclass defaults | the stem's receptive field and the first factor of $s_{\rm tot}$ | a stride other than 4 changes $n_{\rm out}$ everywhere (eq. (P1.4)) |
| `stage_kernel`, `downsampling_rate` | 3, 2 | dataclass defaults | the block kernel; the stride of each stage's first block | the factor 2 of eq. (P1.4) and the $3$ of eq. (P1.6)-(P1.7) |

Two senses of "fixed" (S1.1): `head_pool_ops` is fixed *by the space*
(declared, carried in `spec.fixed`, written to the ledger) and
simultaneously fixed *by omission* in the runner at a different value; the
other eleven are fixed by omission only, and the ledger does not mention
them. A reader of a Stage 4 ledger therefore sees `head_pool_ops: 1` in
every trial's spec and must know that the network had `("mean",)`.
Plan S5.1 chose code 1 as the fixed level; TUNING_1 S3.15 reports the
screening's four head geometries as differing chiefly in generalisation
gap, which is the kind of difference a mis-built head would hide.

### 3.5 The differences table (D-036)

This section establishes, for each inherited axis, the joint stack's range
and default against the standalone DSN's three surfaces, so that a reader of
TUNING_1 can translate. Columns: joint searched range (P0 Table A); joint
runner default; DSN JSON configured range (TUNING_1, "configured"); DSN
`SearchConfig` dataclass default (the range the joint space copies);
`BackboneConfig` default.

| axis | joint range | joint runner default | DSN configured range (TUNING_1) | `SearchConfig` default | `BackboneConfig` default | TUNING_1 section |
|---|---|---|---|---|---|---|
| `depth_exponent` | $\{3, \dots, 6\}$ | 3 | $[2, 5]$ | $(3, 6)$ | 4 | S3.1 |
| `width_multiplier` | $[1.5, 3.0]$ | 2.0 | $[1.5, 5.0]$ | $(1.5, 3.0)$ | 2.0 | S3.2 |
| `block_family` | $\{0, 1\}$ | 0 | $\{0, 1\}$ | $(0, 1)$ | 0 | S3.3 |
| `embedding_size` | $[8, 16]$ | 10 | $[8, 16]$ | $(8, 16)$ | 16 | S3.4 |
| `head_fusion` | $\{0, 1\}$ | 0 | $\{0, 1\}$ | $(0, 1)$ | `False` | S3.15 |
| `head_pool_ops` | fixed, code 1 (built at 0: F-b) | not passed | $\{0, 1\}$ | $(0, 1)$ | `("mean",)` | S3.15 |
| `dropout` | $[0.0, 0.3]$ | 0.0 | $[0.0, 0.3]$ (`regularization.dropout_range`) | `RegularizationConfig` $(0.0, 0.3)$ | 0.0 | S3.8 |

Where the two stacks differ in kind rather than in value:

- **The joint space searches the dataclass ranges, the standalone DSN
  searches its JSON's** (F-g). For depth that is $\{3..6\}$ against
  $\{2..5\}$: the joint search drops the cheapest models (8 blocks is its
  floor) and adds the most expensive corner (64 blocks), the one TUNING_1
  S3.1 warns against (F-s). For the width multiplier the joint range is the
  narrower one ($\le 3.0$ against $\le 5.0$).
- **The head is a $2 \times 2$ factor in the DSN and a $2 \times 1$ in the
  joint space**: `head_pool_ops` is searched there and fixed here (plan
  S5.1), and the fixed level is not what is built (F-b).
- **Dropout is free from trial 0 here, last in the DSN's staged pipeline**
  (TUNING_1 S3.8); the joint space's provenance string for it is wrong (F-c).
- **The base configuration differs**: the DSN trains its backbone with its
  `TrainConfig` and the standalone driver; the joint stack trains the same
  backbone inside the loop of P5 with the losses of P2 and P3. TUNING_1's
  cost and screening statements are about the standalone driver's budget and
  data and do not transfer; what transfers is the architecture, which is one
  file.

Pointers: TUNING_1 S3.1-3.4, 3.8, 3.15 for the standalone axes;
CONFIG_REFERENCE S3.1-3.5 for the `BackboneConfig` table; `TUNING_2` does
not cover the backbone's fixed knobs.

### 3.6 Interactions with the other blocks

This section establishes what the encoder axes touch outside the encoder.

- **With the flow (P4).** $z \in S^{E-1}$ is the flow's condition; $E$ is the
  conditioner's input width; `hidden_features`' searched range is resolved
  from $\max(p, E)$ by the width rule (P0 S3.2), with $p = 26$ dominating on
  the cohort and $p = E = 10$ on the bench.
- **With the DSN loss (P2).** The losses are cosine losses on the sphere;
  `l2_normalize` `True` is what makes the margin $m_{\cos}$ and the half-angle
  a geometry. $C$ class means need $E \ge C - 1$ for a simplex ETF (never
  binding).
- **With the replicate term (P3).** The term reads $z$ for two wells and draws
  from $q_\omega(\cdot \mid z_g)$; the encoder's mode (dropout on or off)
  during the draws is P5's statement; `n_posterior_draws` and the encoder
  do not otherwise interact.
- **With the optimiser (P5).** $\gamma_{\rm wd}$ decays $\psi$ including the
  GroupNorm gains; the gradient-clipping norm of 5.0 (P0 Table C) is taken
  over $\psi$ and $\omega$ together, so a 200 M-parameter encoder changes
  the clipping regime of the flow. `lr` is shared by both networks.
- **With the data (P7).** $W$ enters only through eq. (P1.4): a bench bank at
  $W = 3000$ reaches $n_{\rm out} = 1$ at the $(6, 1.5)$ corner and 2 at
  $(5, 1.5)$ `[RAN]`; a cohort bank at $W = 18000$ never goes below 5.
- **With the search driver (P6).** All six are free in every campaign; none
  is derived or pinned; `boundary_axes` reports a best value at an edge
  ($d_{\rm exp} = 6$ or $w_{\rm m} = 3.0$ would be the ones to widen -- or
  not, given F-s).

### 3.7 Failure modes and the diagnostics that reveal them

This section establishes what breaks, under what configuration, and which
field of the run record shows it.

| failure | configuration | mechanism | diagnostic | tag |
|---|---|---|---|---|
| budget: a trial that cannot finish in the walltime | $d_{\rm exp} \ge 5$, any $w_{\rm m}$; worst at $(6, 3.0)$ | $n_\psi$ 20 M to 215 M, eq. (P1.10) | none in the record; the job's exit; recompute $n_\psi$ from the spec (F-s) | `[RAN]` |
| degenerate pooling | $W = 3000$ with $(6, 1.5)$ or $(5, 1.5)$ | $n_{\rm out}$ 1 or 2, eq. (P1.4): with the mean-only head the "mean" is one or two samples; with the three-statistic head (if F-b is fixed) the std statistic is $\sqrt{10^{-5}}$, a constant | `r_eff` falls; a constant feature column in the head | `[RAN]` |
| the fallback backbone | DSN tree not resolved | `SmallBackbone` ignores all six axes | `backbone: "fallback"` in the record; a `[warn]` line | `[REPO]` `:308-311` |
| the head mismatch | every Stage 4 trial | code 1 declared, code 0 built | none: the ledger and the network disagree silently (F-b) | `[REPO]` |
| embedding collapse | any; made likelier by a strong DSN term (P2) | $z$ concentrates on $C$ points or a line; $E$ cannot prevent it | `r_eff` near 1 (`effective_rank`, `joint_diagnostics.py:107-121`) | `[REPO]` |
| noise from dropout in the real-domain terms | $p_{\rm drop}$ near 0.3 with $B_{\rm met} = 32$, $B_{\rm rep} = 4$ | stochastic forward pass on tiny batches | `rho_grad` per epoch; `val_npe` | `[reasoning]` |
| nothing learned | any | the search's control recipe | $\hat\Delta$ below $\delta_{\min}$ (P6, E7) | `[KB]` plan |

### 3.8 Findings this document owns

- **F-b** (`head_pool_ops`): S3.1, S3.4, S3.5. Resolution: D-038 patch.
- **F-c** (`dropout` provenance string): S3.3.6. Resolution: report; open
  whether it joins the D-038 stream (log, Open calls).
- **F-g** (two sources for the inherited ranges): S3.5. Resolution: report,
  both stated.
- **F-m**, encoder rows (`depth_exponent` 4 vs 3, `embedding_size` 16 vs 10):
  S3.1, S3.3.1, S3.3.4. Resolution: report.
- **F-s** (new): the joint space's `depth_exponent` upper bound 6 is the
  dataclass default, not the DSN's configured 5; TUNING_1 S3.1 says a bound
  of 6 doubles the block count to 64 and should not be searched without
  re-running the budget gate; at that bound the searched encoders have 49 M
  to 215 M parameters `[RAN]`, against a DSN-side measured maximum of
  31.6 M over its own ranges (`[REPO]` TUNING_1 S3.1, a documented
  measurement of the standalone study, not of this stack). No budget gate
  exists in the joint driver. Resolution: report; open whether the range is
  narrowed to $\{3, \dots, 5\}$ (a decision: it changes the space and every
  ledger built on it).
- **F-r** does not apply to this block: all six runner defaults lie inside
  the searched ranges.

---

## 4. Summary of results

- Six searched axes, twelve fixed knobs; the runner passes the six and
  `stem_width` and leaves the rest at `BackboneConfig` defaults (S3.1).
- The architecture is a closed-form function of the knobs: widths by eq.
  (P1.1)-(P1.2), stages by eq. (P1.3), output length by eq. (P1.4),
  GroupNorm groups by eq. (P1.5), blocks by eq. (P1.6)-(P1.7), head by eq.
  (P1.8)-(P1.9), parameter count by eq. (P1.10); the stage count is emergent
  from $w_{\rm m}$ and $B_{\rm blk}$ (S3.2).
- At the runner's defaults: four stages (16, 32, 64, 128; 1, 1, 3, 3 blocks),
  $s_{\rm tot} = 64$, $n_{\rm out} = 282$ on the cohort, $R_{\rm f} = 1133$
  samples, $n_\psi = 359450$ at $E = 10$ and 359708 at $E = 12$ -- the
  cluster's number, reconciled `[RAN]`, `[KB]`.
- Over the searched grid $n_\psi$ spans 0.14 M to 214.5 M; ResNeXt is 2.1 to
  2.9 times cheaper than ResNet; the count is non-monotone in $w_{\rm m}$ at
  fixed depth (S3.2).
- The differences from the standalone DSN (S3.5): dataclass ranges rather
  than JSON ranges (F-g; depth $\{3..6\}$ vs $\{2..5\}$, F-s), a $2 \times 1$
  head factor with the fixed level mis-built (F-b), dropout free from trial 0
  (F-c on its provenance string).
- No encoder default is outside its searched range (F-r does not apply).

## 5. Open points, caveats, assumptions

- **Nothing has trained.** Every statement is about the network as built;
  the effect of any axis on $L$ is the search's question, not this
  document's.
- **The parameter counts are a sandbox replica** of three pure functions of
  `backbone.py` plus a hand-derived count of the torch modules; the replica
  reproduces the cluster's 359708 exactly at the probe's configuration, which
  is the check on both the replica and the derivation. torch itself was not
  run here.
- **The receptive-field and output-length claims assume the fixed kernels
  and strides**; a `BackboneConfig` with other values (reachable by a direct
  caller, not by the runner) changes eq. (P1.4) and $R_{\rm f}$.
- **The dropout mode during posterior draws** is stated in P5 from the loop's
  code; this document assumes it without restating it.
- **F-s is a budget risk, not a correctness finding**; whether to narrow the
  range, or to add a parameter-count guard to the driver, is a decision.
- **The DSN's screening findings** (head geometries differing in
  generalisation gap; the size mix of its search) are quoted from TUNING_1
  as the standalone study's documented statements `[REPO]` doc, not as
  properties of this stack.

## 6. References / further reading

**Knowledge base (full text).** BayesFlow (Radev et al. 2022, IEEE TNNLS)
`[KB-PDF p.1453]`: the summary network reduces the observed data to a
fixed-size vector of learned statistics and is trained jointly with the
inference network; convolutional networks for data with temporal
dependencies; `[KB-PDF p.1456]`: the three sources of error, one of them a
summary network that does not capture the relevant information;
`[KB-PDF p.1457]`: a 1-D fully convolutional network as a summary network
for time series. Goncalves et al. 2020 (eLife) `[KB-PDF p.7 of 45]`:
amortised inference by one trained network. *Simulation Based Inference: A
Practical Guide* (the KB's PDF of that title) `[KB-PDF]`, the subsection on
embedding networks (page not resolved from the search snippet): a feed-forward
embedding network trained end-to-end with the inference network learns the
summary statistics; CNNs for images, RNNs for time series, ResNets as
baselines. `claude/deck_pack/09_NOTATION_AND_GLOSSARY.md` (`E = 10` for the
r2 encoder) `[KB]`; `JOINT_DSN_NPE_USAGE_v1.md` v1.3 S8, S9 `[KB]`;
`HPC_PATHS.md` 5d `[KB]`.

**Repository.** `hpc/dsn/backbone.py:56-121` (`BackboneConfig`), `:132-187` (the pure helpers), `:190-231` (padding, GroupNorm, stem), `:235-333` (the blocks), `:339-397` (the head), `:399-466` (the backbone and its initialisation),
`hpc/dsn/config.py:20-40, 888-896, 1000-1016, 1178-1192`,
`hpc/dsn/condition_space.py:110-113, 263-292`,
`hpc/dsn/Documentation/TUNING_1_searched_axes.md` S3.1-3.4, 3.8, 3.15,
`CONFIG_REFERENCE.md` S3.1-3.5, `hpc/joint/stage3/run_joint_arms.py:282-333,
440-468, 580-600`, `hpc/joint/stage3/probe_dsn_runtime.py:246-257, 324`,
`hpc/joint/stage4/joint_space.py:92-145, 220-246` -- all `[REPO 834eb41]`.

**PubMed (the searches of plan S6, run 2026-10-01, reported in full).**
According to PubMed: (i) `"group normalization" AND convolutional neural
network AND batch size`: 0 results; `group normalization neural network layer
normalization small batch`: 3 results, none the original GroupNorm paper
(Wu and He 2018, ECCV, named here from memory and not retrieved); the one
related hit, Luo et al. 2021, *Switchable Normalization*
([DOI](https://doi.org/10.1109/TPAMI.2019.2932062)), is **abstract only**
(no PMC full text) and is cited for nothing beyond the existence of the
group count as a hyper-parameter of GroupNorm, which is also what the code
shows. (ii) `(RegNet OR "designing network design spaces") AND (convolutional
OR CNN) AND (width OR depth)`: 81 results, the top five inspected
(iris segmentation, genetic U-Net, panoramic depth, wavefront coding, event
cameras) unrelated to the RegNet design-space rule; the original
(Radosavovic et al. 2020, CVPR, from memory) is not indexed. (iii)
`ResNeXt grouped convolution`: 23 results; Wu et al. 2020, Sensors
([DOI](https://doi.org/10.3390/s20061652)), **full text read (PMC7146509)**,
cited in S3.2 for the split-transform-merge description of the ResNeXt
block; the original (Xie et al. 2017, CVPR, from memory) is not indexed.
(iv) `dropout regularization "co-adaptation" neural networks overfitting
Srivastava`: 0 results; dropout's mechanism is stated from the code
(`nn.Dropout`) and tagged `[textbook, from memory]` (Srivastava et al. 2014,
JMLR, from memory). No number in this document comes from any of these
searches.

**bioRxiv.** The connector offers no keyword search; a 30-day
`neuroscience` slice (30 titles inspected, 2026-09-01) contained nothing on
network normalisation, width schedules or grouped convolutions. Nothing is
cited from bioRxiv.

**Data repositories.** No claim about a dataset is made here (the bank's
$W$ is read from the sidecar contract, P7).

**Named from memory, not retrieved** (flagged, used for attribution only,
no content taken from them): Wu and He 2018 (GroupNorm); Radosavovic et al.
2020 (RegNet design spaces); Xie et al. 2017 (ResNeXt); Srivastava et al.
2014 (dropout); He et al. 2016 (ResNet).

*Pre-send check (R1-R8): every symbol typed in S1; every "is" about the
network carries its hypothesis (the fixed kernels and strides, the runner's
call, the bank's $W$); the one transplant -- TUNING_1's cost and screening
statements -- is marked as the standalone study's and not applied to this
stack; one name per object ($d_{\rm exp}$, not $d$; $w_{\rm m}$, not $w$;
the two senses of "fixed" declared in S1.1); the map $h_\psi$ names its
domain and codomain, and $z$ is placed on the sphere; "width", "fixed",
"group" declared; the only borrowed phrasing (TUNING_1 on generalisation
gap) carries its scope; the one two-level object, $r_{\rm eff}$ of the
report split's cloud (computed) against the rank of the embedding
distribution (analytic), is named computed where it appears.*
