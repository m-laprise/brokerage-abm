# To-do memo: high-value follow-up experiments

Status: future work, not part of the paper  
Recorded: 2026-09-01

This memo preserves three focused experiments that could strengthen the results
without requiring another broad parameter sweep. None is currently part of the
manuscript or an established result.

## Priority 1: Principal learning by tenure

### Question

Is the broker's information advantage greatest for recent entrants with little
private experience, and does it narrow as principals accumulate their own match
histories?

### Value

This is the strongest addition per unit of effort. It would connect the broker's
pooled information directly to which principals benefit most. The resulting claim
would also be intuitive for an organizational-behavior audience: inexperienced
actors may depend most strongly on an information-rich intermediary.

### Design

- Preserve the existing model behavior.
- Extend measurement so the common holdout evaluation records broker rank
  correlation, principal rank correlation, and their difference by principal
  tenure.
- Also record outsourcing, match output, and active fitting-history size by
  tenure.
- Treat tenure as the primary grouping variable because exit and entry are
  exogenous. Treat history size as a secondary mechanism measure because it is
  affected by prior matching.
- Focus on entrants observed after the initialization-transient display cutoff.
  Pre-specify tenure bins only after confirming that each bin has adequate support.
- Compute within-seed summaries first, then ensemble means and 95% Monte Carlo
  intervals across seeds.

### Initial run

Rerun only the default neural-network baseline with 50 seeds. If a clear tenure
gradient appears, confirm it at a small number of matching-composition and turnover
conditions. Do not begin with a full sweep.

### Required safeguards

- The new diagnostics must not consume model-event random draws.
- A fixed-seed regression comparison must establish that recording the diagnostics
  leaves every existing trajectory unchanged.
- Initial agents and later entrants have different initial histories. The primary
  analysis should therefore follow entrants rather than interpreting all agents of
  a given tenure as exchangeable without qualification.

## Priority 2: Friction-free outsourcing robustness

### Question

Does high outsourcing persist when the different incidence of self-search costs
and broker fees is removed?

### Motivation

The default model gives self-search a cost for every requested position, whereas
the broker fee is paid only for a successful placement. The nominal amounts are
equal and small, but their incidence differs. Because channel choice uses
satisfaction net of these costs, this rule may favor outsourcing.

### Design

- Compare the default baseline with a friction-free condition setting
  `search_cost_rate=0`.
- Keep prediction, candidate sets, reservation thresholds, matching, and all other
  mechanisms unchanged.
- Use identical seeds and paired Monte Carlo intervals.
- Report outsourcing, channel satisfaction, fill rate, total realized output per
  demanded position, broker and principal holdout ranks, and the output gap.

### Initial run

Run 50 common seeds at the baseline. If the substantive conclusions remain the
same, retain this as a compact robustness check. Extend it only if removing costs
materially changes outsourcing or other central outcomes.

### Interpretation

Persistence of high outsourcing would show that the result is not produced by the
default cost-incidence rule. A large change would instead require deciding whether
to revise the economic rule or present outsourcing as conditional on that rule.

## Priority 3: No-broker market counterfactual

### Question

How does the market as a whole perform with brokerage relative to fully
decentralized self-search?

### Value

The current broker-minus-self-search output gap compares endogenously selected
matches within a market that contains a broker. It is descriptive rather than a
causal estimate of brokerage's contribution. A no-broker condition would provide
the missing system-level counterfactual.

### Design decision required before implementation

The no-broker condition should remove the broker as an institution, not merely
force channel choices to self-search while leaving broker edges in the graph. It
should therefore:

- remove the broker node and roster edges from the network used for matching and
  structural measures;
- omit broker seed history and reputation;
- route all active demand through the unchanged self-search mechanism;
- preserve the same principal types, initial principal network, demand process,
  matching function, turnover, and common seeds as the broker condition.

Broker centrality is undefined in this condition and should not be treated as an
outcome.

### Primary estimands

- cumulative and late total realized output per demanded position, net and gross
  of search costs;
- fill rate and total match count;
- principal holdout rank correlation;
- principal-network degree and other applicable structural measures;
- outcome dispersion across principals, if an agent-level measure is added.

The primary economic comparison should use all demand or all principals, not
channel-conditioned match means.

### Initial run

Run the default broker and no-broker baseline with 50 common seeds from the same
clean code commit. If the comparison materially changes the paper's understanding
of brokerage, extend it to a focused $\rho\times\eta$ design. Do not add it to the
full sweep before seeing the baseline result.

## Recommended order

1. Add the tenure diagnostics and verify that they are behavior-neutral.
2. Run one common-seed baseline array containing the default control, the
   friction-free condition, and, if its design has been approved, the no-broker
   condition.
3. Analyze the baseline results before authorizing any additional regimes.
4. Extend only findings that materially sharpen or challenge the paper's central
   claims.

The explore-exploit hypothesis and its more involved experimental design are
preserved separately in
`output/ridge/ablations/explore_exploit_research_note.md`.
