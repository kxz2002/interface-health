# Novelty Dossier: Reliability-aware Weakly-supervised Multimodal Representation Learning for Endpoint-level Anomaly Detection

## Project context

Research project on Train-Ticket microservice benchmark. Task: per-endpoint x time-window
**One-Class** anomaly detection (Deep SVDD, trained only on Normal-labeled data). 4 data
modalities feed the pipeline: `endpoint_red` (client+server RED, genuinely endpoint-level, 10
features), `service_metric` (Prometheus container metrics, service-level only, 5 features),
`service_log` (Drain3-derived log stats, service-level only, 3 features). Service-level fault
injection labels every endpoint inside the affected service as anomalous during the inject
window, even though only a subset of those endpoints are actually causally affected — this is the
"weak label" problem referenced below. Existing internal baselines (all non-novel, already
implemented): L0 naive concat (AUROC 0.617), L1 independent per-branch encoders (AUROC
0.610+/-0.005), L2 FiLM-style sigmoid gate (AUROC 0.630+/-0.011) — all multi-seed, temporal split.

Empirical evidence already gathered internally (not hypothesized) that motivates this direction:
- All 18 handcrafted features have single-feature AUC in [0.44, 0.56] (near chance) when evaluated
  on the POOLED dataset (all fault types mixed).
- When split BY fault type (26 fault types with usable sample size), per-fault-type discriminative
  power is much higher for some modality/fault-type pairs: e.g. `service_metric` mean single-feature
  AUC reaches 0.699 on `Lv_E_HTTPABORT_travel` (vs. pooled 0.516), and drops to 0.343 (worse than
  random direction) on `Lv_D_TRANSACTION_timeout`. `service_log` PC1 AUC reaches 0.876 on
  `Lv_S_KILLPOD_order` (vs. pooled 0.540). Best-modality-by-fault-type ranking: `service_metric`
  wins on 16/26 fault types by mean feature AUC; `service_log` wins on 19/26 by PC1 AUC.
  This is a single train/eval pass with no cross-validation or multiple-comparison correction —
  directional signal only, not a confirmed statistical result.
- Endpoint-precise labels (`is_endpoint_anomaly`) exist for 16/29 cases (those with a
  `target_endpoint` field in metadata); the other 13 cases fall back to case-level (service-wide)
  approximate labels, i.e. labels are genuinely of mixed/uncertain precision, not just "noisy" in
  the abstract sense.

## Proposed research direction under review

**Working title**: "Reliability-aware Weakly-supervised Multimodal Representation Learning for
Endpoint-level Anomaly Detection in Microservice Systems"

**Core reframing claim**: instead of asking "how should multiple modalities be fused" (a fusion-
architecture question), ask "how can we learn a reliable endpoint representation from weak
supervision and dynamically changing multimodal observations" (a representation-learning
question). Fusion becomes an intermediate step toward a representation with 4 target properties:
(1) encodes endpoint operational state rather than raw concatenated observations; (2) robust to
weak/noisy anomaly labels — similar operational behaviors should stay close in latent space even
when labels are locally wrong; (3) reliability-aware — modality contribution should be a *learned
latent property*, not a manually assigned weight, and should vary by (fault type / operating
condition); (4) detector-agnostic — the representation should be reusable by multiple downstream
consumers (anomaly detection, classification, failure prediction, root cause analysis), not
tightly coupled to Deep SVDD specifically.

**Explicitly NOT yet decided** (this is the point of this novelty check — mechanism is
deliberately left open pending this assessment): candidate realizations mentioned only as
possibilities, not committed to: weakly-supervised contrastive learning, prototype learning,
uncertainty estimation, reliability-aware latent fusion, teacher-student distillation.

**Author's own stated top risk** (should be weighed heavily): "this direction may currently be
just a combination of two existing hot topics (weak supervision + reliability-aware fusion)
rather than a genuinely new problem formulation." The author explicitly does NOT want mechanism
feedback (encoder/gate design) at this stage — only whether the *problem formulation* itself is
novel enough to be worth pursuing, or whether it needs to be sharpened into a causal claim first
(e.g., "why do weak labels and dynamic modality reliability *jointly* cause existing representation
learning to fail in the microservice-endpoint setting, in a way neither problem causes alone?").

## Candidate prior work found in literature search (Phase B, this session)

All found via WebSearch in this session; abstracts/summaries fetched where possible via WebFetch
(some fetches failed due to paywall/redirect/binary-PDF issues, noted below). None of these were
run through a dedicated citation-verification script (`verify_papers.py` was not found/available in
this environment) — treat entries below as **[UNVERIFIED]** in the sense that titles/venues/years
were read directly from search-engine snippets and (where noted) fetched abstracts, not
cross-checked against arXiv/CrossRef/Semantic Scholar APIs. Do not treat arXiv IDs below as
confirmed; flag any suspicious ID back to the user rather than trusting it.

1. **RuntimeSlicer: Towards Generalizable Unified Runtime State Representation for Failure
   Management** (arXiv 2603.21495, appears to be 2026) — [UNVERIFIED]. Abstract confirms: explicitly
   detector-agnostic / task-agnostic unified embedding across metrics+traces+logs; trained via
   "Unified Runtime Contrastive Learning" (cross-modality alignment + temporal consistency) plus a
   "State-Aware Task-Oriented Tuning" adaptation layer; motivated by exactly the same critique used
   in the proposal ("existing approaches tightly couple modality-specific preprocessing,
   representation learning, and downstream models, resulting in limited generalization"). Does NOT
   appear (from abstract) to address weak/noisy labels or dynamic per-fault-type modality
   reliability — reliability/weighting not mentioned in the fetched abstract.

2. **TVDiag: A Task-oriented and View-invariant Failure Diagnosis Framework for Microservice-based
   Systems with Multimodal Data** (arXiv 2407.19711 / ACM TOSEM, 2025/2026) — [UNVERIFIED]. Learns
   cross-modal "view-invariant" failure representations via contrastive learning across
   logs/metrics/traces, explicitly critiques prior work for "treating modalities equally." Targets
   two downstream tasks (culprit localization + failure-type identification) — partial
   detector-agnosticism (multi-task, not shown to be detector-agnostic beyond its own two tasks).
   Abstract does not mention weak/noisy label handling.

3. **Giving Every Modality a Voice in Microservice Failure Diagnosis via Multimodal Adaptive
   Optimization** ("Medicine", ASE 2024, ACM 3691620.3695489 / IEEE 10764924) — [UNVERIFIED]. Title
   and framing directly target *dynamic per-modality reliability/contribution* during training
   (critiques modalities being weighted unequally / some modalities being "silenced" during joint
   optimization) — this is the single closest hit to "modality reliability changes and must be
   actively rebalanced," though the mechanism (per search snippets) reads as *training-dynamics /
   optimization-balancing* (akin to gradient-modulation multimodal learning), not explicitly framed
   as a representation-learning contribution decoupled from a downstream detector. Full text not
   successfully fetched this session (fetch attempts failed) — mechanism understanding here is
   lower-confidence than for RuntimeSlicer/TVDiag.

4. **ARMOR: Missing-Aware Multimodal Fusion for Unified Microservice Incident Management** (arXiv
   2603.25538, 2026) — [UNVERIFIED]. Self-supervised, missing-modality-robust fusion via learnable
   placeholders + dynamic bias compensation + mask-guided reconstruction. Framed as fusion
   architecture (per fetched summary), addresses missingness rather than reliability/weak
   supervision per se — related but distinct failure mode (absent modality vs. present-but-
   unreliable modality).

5. **AnoFusion: Robust Multimodal Failure Detection for Microservice Systems** (KDD 2023, arXiv
   2305.18985) — [UNVERIFIED]. Established prior baseline in this space (graph attention + GRU
   fusion of metrics/traces/logs); cited internally already as a prior-round "not novel, same
   family" reference point for gated fusion.

6. **Uncertainty-Aware Multimodal Anomaly Detection for Microservice Systems With Active Learning**
   (IEEE TSC 2026, doi 10.1109/tsc.2026.3672587) — [UNVERIFIED], abstract not fetched this session
   (found via search snippet only). Title suggests uncertainty-quantification framing close to
   "reliability," combined with active learning (a different mechanism for handling label
   scarcity/noise than weak supervision, but same underlying motivation — labels are imperfect).

7. **CAPAD: Microservice Anomaly Detection with Pseudo-Labeling and Contrastive Training** (ISPA
   2025) — [UNVERIFIED], snippet only. Pseudo-labeling = a common technique for handling weak/noisy
   supervision, combined with contrastive representation learning — relevant as a mechanism
   precedent even if not endpoint-level or reliability-aware.

8. **Weakly-Supervised Log-Based Anomaly Detection with Inexact Labels via Multi-Instance
   Learning** (ICSE 2025, doi 10.1109/icse55347.2025.00189) — [UNVERIFIED], abstract fetch failed
   (redirect to IEEE paywall). Title confirms: weak/inexact-label handling via multi-instance
   learning is an active, published SE-venue technique — but appears log-only (unimodal), not
   multimodal, and not framed around endpoint-level granularity or modality reliability.

9. **ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining**
   (NeurIPS 2025) — [UNVERIFIED], snippet only. General precedent for "representation pretraining
   as the contribution, detector as a downstream consumer" framing, outside the microservice
   domain (industrial/vision anomaly detection) — relevant as a cross-domain analogy for Property 4
   (detector-agnostic) rather than a direct competitor.

10. **General weak-supervision + multimodal reliability combination literature** (multiple hits,
    none microservice/AIOps-specific): UniS-MMC (unimodality-supervised multimodal contrastive
    learning), "A General Framework for Learning from Weak Supervision" (arXiv 2402.01922),
    "Meta-Learn Unimodal Signals with Weak Supervision for Multimodal Sentiment Analysis" (arXiv
    2408.16029). These establish that "weak supervision + per-modality reliability/contribution"
    as a *general ML combination* is already a populated area outside the microservice domain
    (mostly sentiment analysis / vision-language) — relevant to Risk 2 in the proposal (the
    combination itself, decontextualized from microservices, is not new).

**Search coverage note**: ~15 distinct queries run across 5 phases (general multimodal AIOps
survey, detector-agnostic representation, reliability-aware fusion AIOps, dynamic modality
weighting general ML, weak-supervision-microservice, cross-checks). Not exhaustive — did not
search Google Scholar directly (tool unavailable, used WebSearch which mixes sources), did not
check ICLR/NeurIPS/ICML 2026 proceedings pages directly by venue browse, relied on search-engine
ranking rather than systematic venue-by-venue enumeration. Several WebFetch calls failed (paywall
redirects, binary PDF parsing failures, one "socket closed") — abstracts for items 4, 6, 7, 8 above
are based on search-snippet text only, not the fetched abstract itself; treat those 4 as lower
confidence than items 1-3.

## Questions for the reviewer

Please answer these directly and specifically — the user does NOT want mechanism/architecture
feedback (encoder design, gate design, etc.) at this stage. Only the problem-formulation-level
novelty question matters right now.

1. **Novelty mapping**: Given candidate prior work #1 (RuntimeSlicer) and #2 (TVDiag) — both
   already explicitly pursue "detector/task-agnostic unified representation across
   metrics+traces+logs for microservice failure management" — does the proposed direction have any
   remaining novelty once these two are accounted for? Be specific: is "endpoint-level granularity"
   (vs. their apparent service/instance-level granularity, if confirmed) alone enough differentiation,
   or does it need the weak-supervision + reliability angle on top to survive?

2. **Threat analysis** from three angles:
   - Representation learning: does RuntimeSlicer's "Unified Runtime Contrastive Learning +
     State-Aware Task-Oriented Tuning" already deliver Properties 1 and 4 (operational-state
     encoding, detector-agnostic) as claimed goals of this proposal? If so, what's left to claim?
   - Weak supervision: is "labels are locally wrong because service-level fault injection
     over-labels endpoints" different in kind from generic label noise already handled by
     multi-instance learning (item #8) or pseudo-labeling (item #7), or is it the same problem in a
     new setting?
   - Multimodal fusion / reliability-aware: does "Giving Every Modality a Voice" (item #3) already
     claim dynamic, fault-condition-dependent modality reliability as a learned property? If its
     mechanism is training-dynamics rebalancing rather than a latent representation property, is
     that distinction meaningful enough for a reviewer, or a distinction without a difference?

3. **Contribution boundary**: the author's own instinct is to claim "reliability-aware latent
   representation" as the paper's contribution, and treat the specific mechanism (gate, MoE,
   distillation, whatever gets implemented) as an implementation detail, not the contribution
   itself. Is that boundary defensible given prior work #1-#3, or does the *representation itself*
   already lack novelty once RuntimeSlicer exists, forcing the mechanism back into being the real
   claim (which the author considers a weaker fallback)?

4. **Is this a genuine new problem formulation, or a combination of two existing hot topics?** The
   author explicitly suspects the latter and wants brutal honesty. If it is "just" a combination:
   what is the minimal reframing that would turn it into a real new problem? The author suggests
   as an example: "why do weak labels AND dynamic modality reliability jointly break existing
   representation learning in the microservice-endpoint setting, in a way neither one causes
   alone?" — evaluate whether this causal-coupling framing is (a) actually novel, (b) empirically
   testable given the dataset (29 cases, 8 endpoints, per-fault-type AUC data already gathered),
   and (c) worth pursuing over simply abandoning the direction.

5. **Publication positioning**: given items #1-#10, does this read more like a software-engineering
   contribution (ICSE/ASE/FSE — closest neighbors are TVDiag/RuntimeSlicer/Medicine, all SE-adjacent
   AIOps venues), a systems contribution (Middleware/SoCC), or a general ML contribution
   (representation learning workshop, AAAI/IJCAI)? Does the answer change depending on whether the
   final contribution ends up being the representation-learning framing (favors ML venue framing,
   competing directly with RuntimeSlicer-style work) vs. the reliability-aware-fusion framing
   (favors SE/AIOps venue framing, competing directly with Medicine/TVDiag-style work)?

6. **Overall recommendation**: PROCEED / PROCEED WITH CAUTION (name the specific narrowing needed)
   / ABANDON, with the single strongest reason.
