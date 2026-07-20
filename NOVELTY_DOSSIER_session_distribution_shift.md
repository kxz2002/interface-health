# Novelty Dossier v2: Session-Distribution-Aware Fusion for Per-Endpoint One-Class Anomaly Detection

## Revision note

This is a revised proposal after an earlier round of external review rejected the previous
design ("entity-conditioned residual decomposition fusion," framed via mixed-effects/fixed-random
effect language). That review's verdict, summarized: the mechanism was operationally just
per-entity z-score normalization feeding an unchanged existing gated fusion module — a
preprocessing change, not a fusion-architecture novelty; the mixed-effects framing was
inaccurate and should be dropped; and because endpoint→service is a strict 1:1 mapping in this
dataset, there was no real "hierarchy/broadcast" structure being exploited. That mechanism is
being demoted to a supporting/baseline preprocessing step (per-endpoint normal-only
standardization), not claimed as the paper's fusion-novelty contribution.

This dossier proposes a DIFFERENT fusion-mechanism idea, meant to actually change how the fusion
network operates (not just what is fed into an unchanged network), built on a newly confirmed
empirical signal in the data (see below).

## Project context

Research project on Train-Ticket microservice benchmark. Task: per-endpoint × time-window
**One-Class** anomaly detection (Deep SVDD, trained only on Normal-labeled data). Secondary
contribution: a feature fusion mechanism combining, per (endpoint, time-window) row:
- `endpoint_red` (10-dim): client+server RED features, genuinely endpoint-level.
- `service_metric` + `service_log` (8-dim): container metrics + log-derived stats, only
  collectible at service granularity, attached via join on `(service_name, timestamp_window_ms)`.

Existing baselines (all already implemented, NOT novel):
- L0: naive concat (0 params).
- L1: independent per-branch linear encoders, concatenated, no cross-branch interaction.
- L2: FiLM-style gate: `z = e_ep + sigmoid(gate([e_ep; e_svc])) ⊙ value(e_svc)`. Already judged
  in internal review as same family as FiLM (Perez et al. 2018)/GS-Fuse — not novel, strong
  baseline only. Empirical (multi-seed, temporal split): AUROC L0=0.617, L1=0.610±0.005,
  L2=0.630±0.011.

**Confirmed dataset fact**: endpoint→service mapping is strict 1:1 (8 endpoints, 8 distinct
services). No endpoint has a "sibling" endpoint sharing its service in this contract. Any fusion
story relying on multi-endpoint-per-service structure is invalid for this dataset as built.

**Newly confirmed empirical fact (measured directly from data, not hypothesized)**: the merged
training/eval dataset is assembled from **three separate data-collection sessions**, run on three
different dates roughly a week+ apart:
- `normal_v2` (2 pure-Normal cases, collected 2026-07-11) — current sole source of One-Class
  training data.
- `anomod_v1` (11 service-level fault-injection cases, collected 2026-06-28).
- `endpoint_raw2` (16 endpoint-level fault-injection cases, collected 2026-07-04).

Comparing the SAME endpoint's SAME feature (`trace_request_count`, already scaled to [0,1] by the
pipeline's Normalizer) between `normal_v2` and the **baseline-phase** (pre-injection, so not
confounded by any actual fault) windows of the fault-injection sessions, per-endpoint, revealed a
clear, systematic, endpoint-specific shift for 2 of 8 endpoints:
- `inside_pay_service/inside_payment`: normal_v2 mean=0.506 (n=185) vs. fault-session-baseline
  mean=0.975 (n=746) — **1.9x** higher traffic level in the fault-collection sessions.
- `preserveservice/preserve`: normal_v2 mean=0.582 (n=174) vs. fault-session-baseline mean=1.031
  (n=766) — **1.8x** higher.
- The other 6 endpoints show only mild (≤40% relative, mostly <1 std) differences between
  sessions.

This is a real, measured **cross-session distribution shift** — not simulated, not assumed. Likely
cause: each collection session used its own independently-configured synthetic load generator, so
"baseline traffic level" is session-specific for at least some endpoints, on top of being
endpoint-specific. Currently NONE of L0/L1/L2 (nor the demoted per-endpoint-normalization idea)
model or condition on which session/collection-run a row came from — every row is treated as an
i.i.d. draw from one shared distribution per endpoint, which this measurement shows is false for
at least 2 of 8 endpoints.

Separately (already scoped as an infra change, not part of this fusion-novelty question): the
One-Class training pool will also be expanded by incorporating baseline-phase (pre-injection, so
label-safe as Normal) windows from the fault-injection sessions — from 838 rows to ~7,359 rows.
Because that expansion pulls training data from the SAME two additional sessions used to compute
the shift statistic above, it does not resolve the shift — after expansion, the training pool
itself would then also contain multiple sessions with materially different endpoint-specific
baselines for `inside_payment`/`preserve`. This makes the shift MORE relevant, not less: a model
learning a single "normal" boundary for `inside_payment` risks under-fitting to a bimodal
(or the-more-common-of-two) traffic regime rather than the true underlying operating envelope.

## Proposed method under review

**Session-distribution-aware fusion**: the fusion/gating mechanism explicitly conditions on which
collection session a row belongs to (a small, closed, known categorical variable — 3 known
sessions currently, could grow if more data is collected later), not just which endpoint. Two
candidate mechanism designs under consideration (neither implemented yet, both plausible, asking
the reviewer to weigh in rather than picking one blind):

**(a) Two-level reference/residual, both fed into the SAME existing gate:** compute a per-endpoint
normal-only baseline (demoted mechanism from the prior round, kept only as an input feature, not
claimed as novel) AND a per-(endpoint, session) baseline, both as fixed statistics estimated from
Normal training data. Feed the gate/value network both residuals (endpoint-relative and
session-relative) so it can use whichever is informative, rather than assuming session
membership is irrelevant. This is architecturally still "compute some reference statistics,
concatenate into existing L2," so it inherits much of the previous critique (still no change to
how the gate/value layers themselves operate) — flagged here for the reviewer to judge whether
adding a session axis changes that verdict.

**(b) Learned session-conditional affine correction before the existing gate** (closer to domain
adaptation / batch-effect correction literature than to (a)): learn a small per-session affine
transform `(scale_s, shift_s)` — analogous to a domain-adaptation "batch effect correction" or a
conditional-BatchNorm-style per-domain affine, but conditioned on collection-session identity
rather than a learned/inferred domain code — applied to the service-level features before they
enter the SAME existing gate/value computation, trained end-to-end (backprop through the affine
params) rather than fit as a fixed closed-form statistic. This is a genuine (if small) learned
parametric component reacting to a real, measured distribution-shift signal, which is a different
category of claim than (a)'s "just another reference-subtraction feature."

Motivation for why this differs from the previous (rejected) proposal: the prior proposal's
grouping variable was endpoint identity, used to normalize away *within-endpoint-across-time*
variation, with no learned parameters and no clear held-out shift signal to justify itself beyond
"different endpoints have different baselines" (true but not surprising, and easily read as
"just z-score by group"). This proposal's grouping variable is collection-session identity, and
the mechanism REACTS to a specific, measured shift (1.8-1.9x on 2 endpoints) that plain per-endpoint
normalization does NOT fix, since per-endpoint stats fit across all sessions combined would blend
the two session-modes together for those 2 endpoints, potentially averaging away exactly the
signal that matters.

**Explicitly acknowledged limitation to raise with the reviewer**: only 3 sessions currently
exist, and only 2 of 8 endpoints show a strong shift. A mechanism built around 3 discrete session
categories, only 2 of which show the phenomenon it's designed to address, may be seen as
overfit to this dataset's specific collection history rather than addressing a general,
recurring phenomenon in the target domain (production microservice monitoring). Need the
reviewer's judgment on whether "distribution shift across independently-collected data-batches in
the same benchmark" is (i) a recognized, named problem in the literature with an established
name/framing we should adopt or compare against, and (ii) whether it's a strong enough
motivating phenomenon to build a fusion novelty claim on, given it's currently observed in a
minority of endpoints in a research benchmark rather than demonstrated as a persistent production
phenomenon.

**Fallback plan already discussed internally if this direction doesn't pan out empirically**:
objective-coupled fusion — let the fusion gate's strength/behavior be informed by feedback from
the downstream Deep SVDD hypersphere (e.g., current distance-to-center), rather than training the
fusion module and the SVDD encoder as two independently-optimized, mutually unaware components as
today. Not elaborated in this dossier (higher engineering cost, not yet detailed), but relevant
context: if the reviewer judges the session-distribution angle weak, whether pursuing an
objective-coupled fusion (fusion mechanism aware of and reactive to the one-class detection
objective, rather than purely feature-side) is a more promising axis is a useful thing for the
reviewer to weigh in on too, even briefly.

## Candidate prior work already found in the previous review round (context, not re-verified)

- RevIN (Kim et al., ICLR 2022) — per-instance/per-sequence normalize-then-restore, not
  entity/session-conditioned, not evaluated in previous round as directly applicable here.
- Domain-specific/conditional BatchNorm, groupwise standardization — previous reviewer's suggested
  correct framing for the demoted per-endpoint mechanism.
- Mixed-effects neural network family (MeNets, ARMED, NGMM, TabMixNN) — ruled inapplicable
  terminology in the previous round; not being invoked again here.
- AIOps multimodal fusion papers found (AnoFusion, GAL-MAD, MODIFy, Grassmann-manifold fusion,
  Twin Graph attentive fusion, missing-aware fusion) — all pursue graph/diffusion/manifold/
  attention sophistication; none were found (in a non-exhaustive search) to condition fusion on
  collection-batch/session identity or to do domain-adaptation-style batch-effect correction in
  this specific application.
- NOT yet searched in this round (flagging honestly rather than fabricating a search that wasn't
  done): domain adaptation / covariate shift / batch effect correction literature broadly (e.g.,
  batch-effect correction in genomics/bioinformatics, which is a well-established named problem
  for exactly this "same measurement, different collection run, systematic shift" pattern) — this
  is the single most important gap for the reviewer or a follow-up search to fill, since if this
  proposal is just "batch-effect correction, applied to microservice telemetry, feeding a gate,"
  that framing needs to be surfaced and compared against explicitly, not discovered by a reviewer
  after submission.

## Questions for the reviewer

1. Is (a) or (b) — or neither — likely to read as a genuine fusion-mechanism novelty rather than
   another preprocessing/normalization change in different clothing? Be specific about which
   parts of each design would or would not change the verdict from the previous round's critique.
2. Is "cross-session/batch distribution shift correction" a well-established named problem
   (e.g., batch-effect correction, covariate shift adaptation, domain-conditional normalization)
   that this proposal should explicitly position itself relative to, rather than presenting as a
   new observation? If so, name the most relevant established framing/prior work area.
3. Given only 2 of 8 endpoints show a strong (>1.5x) session-level shift, and there are currently
   only 3 known sessions total — is this phenomenon substantial and general enough within THIS
   dataset to motivate a fusion mechanism, or is it too narrow/idiosyncratic to the specific
   collection history to be a defensible motivating signal for a paper contribution?
4. Between (a) (extra reference/residual features into the unchanged gate) and (b) (learned
   session-conditional affine correction, trained end-to-end, feeding the unchanged gate) — which
   is the stronger design if one must be chosen, and why? Is there a third design the reviewer
   would suggest instead that better addresses the actual gap (gate/value mechanism itself
   unchanged across every version of this proposal so far)?
5. Given this is one of two contributions in a mid-tier-venue-targeted paper (not a top venue),
   is the objective-coupled fusion fallback (fusion mechanism reacting to SVDD hypersphere
   distance feedback, not independently trained) worth prioritizing over the session-distribution
   angle instead, purely on novelty-ceiling grounds — even before considering its higher
   engineering cost? A brief comparative judgment is enough; full design of that fallback is out
   of scope for this dossier.
