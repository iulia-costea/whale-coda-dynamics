# Data dictionary, provenance, and assumptions

This project uses the Assayag release as an observational source, although it
already contains derived features. Normalized output is JSON Lines: one JSON
object per coda, with typed numbers, nulls, and arrays. Every original column
and value is retained in `raw_values`. Only fields needed by RQ1 or its audit
are also normalized. The executable column mapping is `FIELDS`
in `src/whale_coda_dynamics/preprocessing/audit.py`.

## Core fields

| Source | Normalized | Meaning and limits |
|---|---|---|
| codaNUM2018 | source_coda_id | Original coda identifier, retained as text |
| Tag | tag_id | Tag label; used for ordering in the authors' notebook |
| REC | recording_id | Source recording/segment label; continuity between labels unverified |
| guy_coda_type | coda_type | Authors' preferred label; preserved without reconciliation |
| decision_tree_coda_type | alternative_coda_type | Alternative label; disagreements flagged |
| Focal | focal_status | 1 tagged caller, 0 non-focal, null missing; not individual identity |
| Whale | source_whale_code | Caller code; scope and reset rules unverified |
| IDN | source_individual_id | Source ID; zero remains a flagged, unresolved value |
| Name | raw_values only | Constant within tags in this release; do not assign to every caller |
| Bout | source_bout_code | Source annotation; definition and reset rules unresolved |
| Coda | raw_values only | Source numbering, not a globally unique identifier |
| TsTo | source_onset_s | Numeric onset used in seconds by notebook; origin/timezone unresolved |
| Duration | duration_s | Coda duration in seconds |
| Date | date_candidate | ISO date parsed assuming day-month-year; original retained |
| TagOnTime | tag_on_time_candidate | Parsed clock time; origin/timezone not established |
| nClicks | click_count | Number of clicks |
| Unit, UnitNum | raw_values only | Source social-unit labels |
| ICI1–ICI40 | raw_values only | Inter-click measurements; retain padding zeros without imputation |
| IPI1–IPI40 | raw_values only | Inter-pulse measurements; units and numeric sentinel rules need confirmation |
| ELKI2, ELKI2name | raw_values only | Empty in downloaded release; retained |

Date conversion currently supports numeric months and Hebrew `מרץ` (March)
and `מאי` (May).
Other unknown formats produce flags, not guessed dates. Numeric sentinel values
such as 65535 in IPI columns remain unchanged; configured missing counts do not
include these undocumented codes. We do not add TagOnTime to TsTo yet.

## Existing derived fields

| Source | Interpretation verified from notebook |
|---|---|
| past_codas | Zero-based row indices, newest first, at most 10; matches Focal and Tag, not Whale; rolling 7-second onset rule |
| codas_history | Same traversal, both focal statuses included |
| overlap_pair / non_overlap_pair | Index of the next row for focal-to-non-focal pairs |
| reverse_overlap_pair / reverse_non_overlap_pair | Index of the next row for non-focal-to-focal pairs |
| pair_category | Category of the outgoing pair; null if none selected |
| CB | End-to-start gap to the selected next interlocutor coda; negative for overlap |
| CI | End-to-start gap to next eligible coda matching Tag, Focal and Whale |

Pair selection uses adjacent rows sorted by Tag and TsTo, opposite binary Focal
statuses, and onset difference <=7 seconds. CI searches forward within seven
seconds of onset. Neither history function enforces Bout or REC boundaries.
Only the two history lists are parsed for audit checks. Other precomputed pair
fields remain in `raw_values`; RQ1 recomputes its own features.

## Provenance and intentionally unresolved fields

`coda_id` is `assayag:row:<zero-based index>` and is stable only for the exact
source version identified by `source_sha256`. `source_csv_line` is the nominal
line number for this single-line-record CSV. No rows are removed or reordered.
Absolute timestamps and caller/exchange IDs are not created by normalization.
Candidate duplicates and shared onsets are flagged, never silently removed.

## Feature contract

`build_features` requires explicit confirmed definitions or explicit acceptance
of provisional assumptions, globally unique exchange
IDs and coda IDs, comparable onsets/offsets within each exchange, and exactly two
callers per exchange. Caller continuity and recording boundaries must already
have been resolved by its caller. Tied onsets are rejected pending a policy.

- Ct: target coda type.
- A_history/B_history: completed prior codas by self/other, oldest first, at most
  10 each. Histories cannot cross exchanges. Partially overlapping codas are
  excluded from these histories because their complete type may not yet be available.
Interactional state S (PLAN [D5], DEC-1) holds only what is known at target
onset. It is defined relative to the immediately preceding coda of either caller
(latest earlier onset in the exchange; ID exported as `turn_predecessor_id`) and
never contains a coda's duration or content.

- turn_category: exchange_start (no earlier coda), overlap (target starts before
  the preceding coda ends), otherwise self_continuation (preceding coda is the
  target caller's own) or switch.
- gap_s: target onset minus the end of the preceding coda, for self_continuation
  and switch; 0 otherwise.
- overlap_elapsed_s: target onset minus the onset of the preceding coda, for
  overlap; 0 otherwise.

Sensitivity and diagnostic fields, in no S input list:

- retrospective_delay_s: target onset minus the end of the most recently started
  earlier interlocutor coda (ID exported as `retrospective_predecessor_id`).
  Negative values use an endpoint after target onset; absent predecessors yield
  null. For continuations it includes the caller's own previous coda, so it is
  used only for the sensitivity run with the old delay.
- overlaps_prior_interlocutor: any earlier interlocutor coda still active.

These are explicit proposed RQ1 choices, not a reproduction of the authors'
history algorithm. The audit command does not construct them; `prepare.py` applies
them to retained provisional exchanges and exports their identity status.

## Provisional exchange outputs

`exchanges.py` operates on normalized observations separately from the
feature constructor. It uses hard boundaries `(Tag, REC, Bout)`
and sorts by onset inside each group. A gap greater than the configured limit
from the immediately preceding onset-ordered coda's end starts a new segment.
Negative gaps retain overlaps; no running maximum of prior end times is used.

A whole segment is excluded if caller codes are missing/zero, the distinct caller
count is not exactly two, onsets tie, or coda types are missing. Invalid onset or
duration excludes its entire hard-boundary group to avoid joining across an
event with unknown timing. Missing Focal alone does not exclude primary segments:
the primary local identity assumption uses Whale. Unknown absolute timing remains
unresolved, and only within-segment differences are used.

`exchange_assignments.jsonl` has one row per original coda. `retained` and
`exclusion_reasons` express analysis inclusion. `exchange_id` contains threshold
and first source-row index; it is source-version-specific. `local_caller_id`
combines this segment ID with the source Whale code. Its status is explicitly
`provisional_local_code`. It must not be interpreted as identity across segments.

`exchange_segments.jsonl` summarizes all candidate segments. History eligibility
counts use completed prior events, matching the proposed feature contract.
`strict_focal_subset` requires both binary Focal values, no missing values, and
a one-to-one relationship with the two Whale codes. This provides a stricter
comparison sample without establishing actual individual identities.

The 5/7/10-second runs quantify sensitivity of sample construction only; they
are not tests of robustness of model results. Exclusion reason counts overlap.

## Final target tables

`rq1_targets` includes each retained coda. `rq1_comparison` selects rows with at
least one completed prior coda per caller. Both are exported as CSV and JSONL.
Additional fields are `A_history_length`, `B_history_length`,
`comparison_eligible`, `comparison_exclusion_reasons`, `strict_focal_subset`,
`split`, source row/coda identifiers, tag/recording/bout codes, onset/offset,
and source checksum. `whale_id` is the provisional segment-local caller ID;
it is not a photo-identified animal. History IDs refer to rows in `rq1_targets`,
which can include earlier codas not themselves eligible comparison targets.

`recording_date` is the normalized source Date carried through assignments and
final tables. Split assignment uses SHA-256 of seed + date, with 15% of days each
allocated to test and validation (rounded, at least one each), and the remainder
to training. `rq1_day_splits` replaces the old tag manifest. Every retained coda
on a date, including context-only rows and other tags, shares its split. Missing
dates or tags spanning multiple date labels stop preparation for review. Dates
remain recording labels; absolute onset times and midnight rollover are unresolved.

The eight input conditions are declared in `config/rq1.json`: none, S, A, B,
A+B, A+S, B+S, A+B+S. S uses `turn_category`, `gap_s` and `overlap_elapsed_s`. The main
RQ1 contrast is A+B+S versus B+S, evaluated on exactly the same comparison targets.
A_history/B_history lengths and IDs are support metadata, not extra unlisted
predictors. Context-free and S-only conditions must not receive history masks or
lengths as inputs. For each condition, provide only its declared input columns.

Timing is onset-available (`timing_convention` = `onset_available`). The only
other accepted value, `retrospective_signed_end_to_start`, is the sensitivity run
and requires S = `turn_category`, `retrospective_delay_s`.
`delay_uses_post_onset_endpoint` is true when the retrospective predecessor ends
after target onset. The flag itself is diagnostic metadata, not an extra predictor. Completed-prior
history eligibility remains unchanged. Held-out histories use observed previous
codas, not recursively generated predictions.

## Audit output

The only generated report is `reports/rq1.md`. Row-level audit findings are
embedded as `audit_issues` in each normalized record; exchange exclusions remain
in the exchange assignment table. There are no separate audit report files.

## Sources inspected

1. [Assayag et al. (2026)](https://doi.org/10.1080/09524622.2026.2684665)
   and [dataset/code](https://github.com/MaorAssayag/coda-sequences-relations-and-prediction).
   The notebook reads an earlier `Shane_Annotations.csv` path, filters nClicks >2,
   and constructs the features supplied in our release. That filename alone does
   not establish a public upstream version.
2. [Sharma et al. (2024)](https://www.nature.com/articles/s41467-024-47221-8)
   and [data](https://github.com/pratyushasharma/sw-combinatoriality/tree/main/data).
   Methods describe temporal order and speaker annotation for 3,948 tag-recorded
   codas. The inspected `sperm-whale-dialogues.csv` release has 3,840 rows.
   Source authors attribute original collection and annotation to Shane Gero/DSWP.
3. [Gubnitsky et al. (2025)](https://www.nature.com/articles/s41598-025-97009-z)
   and [detector code/examples](https://github.com/Project-CETI/Coda-detector).
   Describes manual 2014–2016 annotations and additional automatic annotation.
   Explains focal/non-focal separation, but is not a complete CSV dictionary.

The earlier public release shares recording/onset combinations with 2,788 rows
of the Assayag file. That comparison is preliminary: keys are not unique and
some annotations differ. It supports common recordings, not a validated merge
or proof of direct file ancestry. The temporary reference file was not merged.

## Local checks performed before implementation

All 3,933 rows, both supplied histories, all 1,508 pair categories, and populated
CB values were checked against the authors' notebook rules. Pair category counts:
340 overlap, 412 non-overlap, 316 reverse overlap, 440 reverse non-overlap.
The notebook's saved output agrees. The paper reports 1,538 pairs. Replacing only
the onset cutoff with an end-to-start cutoff gives 1,518 pairs; this is not a
complete reconciliation. Do not force the data to reproduce a published count.

946 rows' past_codas include a different Whale code. This could reflect code
scope or caller mixing; it is not by itself evidence of 946 annotation errors.
The generated report contains current counts of all implemented flags.

## Decisions still needed

| Question | Status / consequence |
|---|---|
| Is Whale stable within REC, Bout, Tag, or a different unit? | Unresolved; exported whale_id is provisional and segment-local |
| What does Bout mean, and where does it reset? | Unresolved; no definitive exchange_id |
| Is TsTo relative to tag-on, recording start, or midnight? Which timezone? | Unresolved; no absolute timestamp |
| Are different REC labels continuous segments? | Unresolved; do not concatenate by label alone |
| What do IDN=0 and missing Focal mean? | Preserve and flag; no automatic caller imputation |
| How were 3,933 codas/1,538 pairs selected? | Codas match release; pairs differ |
| Which coda classification is authoritative for RQ1? | Preserve both; preferred source label is provisional |
| What do IPI numeric sentinels mean? | Preserve, not used in RQ1 features |

The current pipeline generates observational features using the provisional
definitions below. These assumptions remain limitations of the analysis.

## Short-term operational decision

The project proceeds without waiting for author responses. The provisional
grouping in `config/exchanges.json` restricts sequences to Tag/REC/Bout groups,
uses local Whale codes, splits on an end-to-start gap above seven seconds, and
retains exactly-two-caller segments without tied onsets. The source definitions
above remain unresolved. Hard boundaries may truncate real exchanges; local
caller codes may not track identities perfectly. Claims must be conditional on
these assumptions and evaluated with the provided threshold and focal-subset
sensitivity analyses. No author message has been sent.

## Pair-category missingness audit

Recomputed from all 3,933 Assayag source rows on 2026-09-23, using the configured missing tokens: 2,425 have no `pair_category`, and 1,508 are labelled. Among unlabelled codas, 1,597/2,425 (65.9%, approximately 66%) are non-focal; among labelled codas, 756/1,508 (50.1%, approximately 50%) are non-focal. All 157 rows with missing `Focal` are unlabelled.

Missing pair labels are therefore associated with focal status. Turn category is derived from event timing and caller order; missing `pair_category` alone is not grounds for removing codas. Existing dyadic-exchange and timing eligibility rules still apply.

## Sharma long-coda flag

`long_coda` records whether source `nClicks > 10`; null means the count is missing or invalid. It is present in `codas_normalized_rq2.jsonl`, `rq2_targets`, and `rq2_comparison`. This diagnostic flag does not filter rows or enter model inputs. The generated RQ2 report gives source, retained, and comparison-target counts so later sensitivity analyses can distinguish their populations.
