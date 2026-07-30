# Literature synthesis — reducing LLM over-assertion in subsumption matching (2026-07-21)

Consolidation of four parallel literature reviews, framed on our two-stage matcher
(Stage 1 embedding retrieval top-K → Stage 2 LLM classifies each candidate as
`<`/`>`/`=`/none). Failure mode: the LLM asserts a relation on Stage-1 near-misses
whose gold is `none`, huge FP mass, F1 collapses. Question asked: is this a known
problem, what are we missing, what is the best fix.

## 0. Is this known? — YES, definitively, in three literatures (one is ours)
The exact failure is named and studied in: (i) **logical alignment repair**
(LogMap, ALCOMO, LogMap-C, conservativity — ALCOMO is Meilicke's PhD at **Uni
Mannheim**), (ii) **selective prediction / calibration / abstention**, (iii)
**hard-negative-aware subsumption learning + incomplete-reference evaluation**
(BERTSubs, Bio-ML, OntoLAMA). Our over-assertion is catalogued as the canonical
LLM-OM hallucination in **OAEI-LLM** (Qiang et al. 2024/25, arXiv:2409.14038 /
2503.21813): "classify subclass/related pairs as equivalence," FP rate rises with
one-to-many mappings.

## 1. Why our three attempts fell short — all three are documented, fixable flaws
- **H1 confidence abstention failed → EXPECTED.** Raw `P(yes)/(P(yes)+P(no))`
  measures an input-independent positive prior (surface-form competition;
  "none" is dispreferred in forced choice), not evidence. Holtzman et al.
  *Surface Form Competition* (EMNLP 2021); Zhao et al. *Calibrate Before Use*
  (ICML 2021); Zheng et al. *LLMs Are Not Robust MC Selectors* (ICLR 2024).
  **Fix:** subtract the prior (contextual/batch calibration on the top-20
  logprobs we already read), reframe `none` as the *residual* of calibrated
  binary decisions, use a **score margin** (top vs runner-up) not an absolute
  threshold, and set the cut by **risk–coverage** per relation (Geifman &
  El-Yaniv NeurIPS 2017; Guo et al. temperature scaling ICML 2017; Chow 1970).
- **gpt-oss verifier collapsed to "no" → KNOWN format bug, not a capability
  limit.** gpt-oss uses the **Harmony** format (reasoning in an `analysis`
  channel, answer only in the `final` channel). With `max_tokens=1` we read the
  first *reasoning* token, never the decision. Tam et al. *Let Me Speak Freely?*
  (EMNLP 2024) — forcing the answer before the reasoning span completes is the
  worst case. **Fix:** let it reason, then read the verdict via
  `guided_choice=["Yes","No"]` on the final field (logprob stays readable) or
  self-consistency majority vote (Wang et al. ICLR 2023).
- **Same-model self-verification only worked for 3/4 → EXPECTED self-preference /
  sycophancy.** Panickssery et al. *LLM Evaluators Favor Their Own Generations*
  (NeurIPS 2024); Zheng et al. *LLM-as-a-Judge* (NeurIPS 2023, self-enhancement
  bias); Huang et al. *LLMs Cannot Self-Correct Reasoning Yet* (ICLR 2024) —
  intrinsic self-correction without an external signal can *degrade*. **Fix:**
  an **independent** verifier (a different model, or an off-the-shelf NLI model
  as an entailment gate) and **direction symmetrization** (a false subsumption
  often reads "yes" in *both* directions → cheap high-precision reject).

## 2. The lever we have NOT pulled — and every agent ranks it #1
**Logical consistency / coherence post-filter.** Our predicted `<` graph + the
two ontologies' own hierarchies is a mixed-relation alignment; a false edge, added
as a bridging axiom, tends to create an **unsatisfiable class** (consistency
violation) or a **novel named subsumption** inside one ontology (conservativity
violation). Deleting the minimal, lowest-confidence offenders removes FPs with
near-zero recall loss. Free, model-independent (works for gpt-oss with no LLM
call), provably only removes errors, and **generalizes our transitive-reduction
result** from "clean but small" to actually deleting contradiction-inducing edges.
Cheap version over `RDFGraphWrapper`/NetworkX: (a) antisymmetry/exclusivity —
drop both when `A<B` and `B<A`, or when a pair is guessed into conflicting
relations; (b) `<`-cycle detection and break the lowest-confidence edge;
(c) sibling-exclusion (ancestor-safe!): reject `s<t` when `t` is a *sibling/
descendant* of an already-accepted target, **never** when `t` is an *ancestor*
(those are our 38 % gold gaps). Full version: LogMap-C / ALCOMO conservativity
repair. **Cite:** Meilicke, *Alignment Incoherence in Ontology Matching* (PhD,
Uni Mannheim, 2011 — ALCOMO); Solimando, Jiménez-Ruiz, Guerrini, *Conservativity
Principle Violations* (ISWC 2014; KAIS 2017 — LogMap-C); Jiménez-Ruiz & Cuenca
Grau, *LogMap* (ISWC 2011).

## 3. Our 38 % gold gaps are a NAMED measurement problem, not (only) model error
Subsumption references are inherently incomplete (a class has many valid
transitive superclasses no curator enumerates), so "false positives" are often
un-annotated true subsumptions. Community responses, all citable:
- **Semantic P/R** — credit any predicted edge entailed by the reference's
  transitive closure (Euzenat, *Semantic Precision and Recall*, IJCAI 2007) — the
  formal provenance of our hierarchy-credited metric.
- **Hierarchical hP/hR/hF** — augment predicted+true labels with all ancestors,
  set-based P/R (Kiritchenko et al., Canadian AI 2006) — our credit rule's origin.
- **Ranking instead of global precision** — Hits@K / MRR over 1 positive + hard
  negatives, ancestors excluded (Bio-ML, He et al., ISWC 2022) — exactly our
  Stage-1 `evaluation_recall` protocol.
- **Three-bucket reporting** — TP (confirmed) / FP (contradicts reference) /
  **`?` unknown** (plausible, not contradicted) excluded from the precision
  denominator (Bio-ML "unknown vs negative"; silver-standard / golden-hammer
  caveat, Hertling & Paulheim, ESWC 2020 — our supervisors).
- **isAmong** descendant-set Precision*/Recall*/F1* — the actual OAEI 2025
  BeyondEquivalence metric (rewards containment/partial overlap). NB: the track is
  a **2025 first edition**; its metric is isAmong, NOT Recall@K/MRR (that is
  Bio-ML). Keep them distinct when citing.
- Label `none` dev pairs with OntoLAMA's **disjointness gate** ("stays satisfiable
  + no common descendants") so we never mislabel an undeclared true subsumption as
  negative (He et al., *OntoLAMA*, Findings of ACL 2023; BERTSubs neighbourhood
  negatives, WWW 2023).

## 4. What the working LLM-OM systems do for precision (transferable, model-indep.)
OLaLa (Hertling & Paulheim, K-CAP 2023 — our namesake; source of
`get_confidence_first_token`): explicit **"none" option** in an MC prompt +
tunable confidence threshold + 1:1 cardinality. LLMs4OM (Giglou et al., ESWC
2024): **dual gate** = LLM-confidence AND retriever-similarity floor; Concept /
Parent / Children views in the prompt. MILA / Agent-OM (2024/25): **mutual-best /
bidirectional confirmation** (accept only if top-ranked in *both* directions) —
generalizes our Both-Order-Voting (B2); escalate only borderline cases to the LLM.
MapperGPT / "LLMs as Oracles" (EACL 2026): LLM as a **high-precision post-filter
on the uncertain subset only**.

## 5. Prioritized recommendation (clean precision per cost, model-independence)
1. **Logical consistency post-filter** (§2) — free, provably precision-only,
   extends our work, our department's tradition, works for gpt-oss with no LLM
   call. Highest-value, cleanest thesis result.
2. **Fix + harden the verifier** (§1): independent verifier / NLI entailment gate,
   Harmony-aware decoding for gpt-oss, direction symmetrization, margin gate.
   Verification already separates for 3/4 (llama +0.383, gemma +0.393, mistral
   +0.310); these fixes target the 4th and lift precision further.
3. **Recalibrate H1 properly** (§1): per-relation temperature/Platt on
   hard-negative dev, threshold by risk–coverage, compare logprob vs P(True) vs
   verbalized. Turns "abstention failed" into "threshold was miscalibrated —
   fixed."
4. **Honest incomplete-reference reporting** (§3): three-bucket + semantic/
   hierarchical credit (+ isAmong as a second metric). Converts much of the 38 %
   from "error" into an explicitly-reported unknown class.
5. **Hierarchy-context prompts** (§4): ancestor/descendant/sibling context in
   Stage-2 (BERTSubs Path/BC; LLMs4OM Parent/Children) to curb similarity-driven
   over-prediction.

**Single highest-value, model-independent lever:** the logical consistency /
coherence post-filter (#1).

## Citation corrections flagged by the reviewers
- BERTMap authors: He, Chen, Antonyrajah, Horrocks (AAAI 2022); Bio-ML is ISWC
  2022 (not NeurIPS). OAEI BeyondEquivalence = 2025 first edition, isAmong metric.
- A handful of 2026-dated arXiv IDs surfaced by search are unverified and were
  demoted; all load-bearing claims rest on established venues (ICML/ICLR/ACL/
  ISWC/NeurIPS 2017–2024).
