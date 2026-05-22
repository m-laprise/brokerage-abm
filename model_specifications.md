# Appendix A: Specifications for an ABM of Transient Brokerage in Matching Markets

This appendix details the specifications of an agent-based model of brokered matching meant to formalize and demonstrate the theory of brokerage put forward in its companion article.

## Theory Overview

The structural-hole theory of brokerage (Burt, 1992) locates the broker's value in its network position, bridging disconnected parties. Structural-hole brokerage, when performed at scale, can be self-liquidating: each successful match creates a direct tie that densifies the network and closes the holes that created bridging opportunities in the first place, unless the broker aggressively recruits new, distant clients.

I propose a complementary view of brokerage. Brokerage is outsourced relational work: the broker constructs viable matches between parties who cannot easily evaluate each other. This relational work generates an informational byproduct that the broker can leverage. The broker accumulates knowledge about heterogeneous parties and how to match them successfully. Structural position provides the access that feeds learning, but while each successful match erodes the broker's structural advantage, it also strengthens its informational position (by adding an observation to its experience of the matching function).

The broker converts structural capital into informational capital through the act of brokering. When the matching problem is sufficiently complex, the informational capital compounds faster than structural capital erodes, and this compounding advantage can support a transition from intermediation to capture, transforming the broker into a principal selling the resource it was formerly intermediating or data and analytics. This is *transient brokerage*, a process that highlights the broker's power rather than its fragility.

This project develops an agent-based model of brokered matching to formalize and demonstrate the theoretical framework. In the model, agents seeking pairwise matches either search their own network or outsource the search to a broker. Capture can take two forms: resource capture, where the broker becomes a principal and locks clients out of learning, or data capture, where the broker sells its predictions as an analytics service while clients continue matching directly. All agents use heuristic decision rules. No agent solves an optimization problem or holds beliefs about other agents' strategies.

## Questions

1. Under what conditions does the broker develop, maintain, or lose an informational advantage over its clients?

2. Under what conditions can the broker leverage its advantage for capture?

3. What form does capture take and how does capture impact the dynamics of the broker's advantage?

## Main Propositions

The simulation is designed to demonstrate the following propositions.

### Premise

**A broker provides value in a matching market because of its structural or informational advantages.** A broker helps create a match between principals who, without the broker's intervention, could not easily find each other (access; structural advantage) or were unaware that they would benefit from a match (matchmaking; informational advantage). In other words, the broker's service is valuable both because it can find counterparties that clients cannot reach and because it can assess match quality better than its clients can.

- ***The existence of a structural advantage depends purely on network topology.*** It can be measured using betweenness centrality, constraint, and effective size.

- ***The emergence of an informational advantage depends on the value of the data a broker and its clients accumulate, which in turn depends on the form and difficulty of the matching problem.*** When the matching problem is hard to solve, local or limited experience can be insufficient relative to a broker's high volume of cross-market data.

### 1. Advantage

#### Proposition 1.1

**A broker's structural and informational advantages exhibit distinct dynamics over repeated brokerage activity.**

**1.1a. Structural advantage tends to be self-liquidating.** A broker that bridges a gap between two principals, and successfully matches them, creates a direct relationship between them. With each match, the broker's network position weakens. Direct ties accumulate between principals and structural holes close. This is particularly the case when brokerage occurs at scale. A broker can counteract the self-liquidating tendency by aggressively recruiting new candidates, continuously expanding its reach into parts of the network that principals cannot yet access. However, when the broker's pool of candidates is stable or slowly evolving, placements create direct ties faster than new structural holes are bridged, and the erosion of structural advantage dominates.

**1.1b. Informational advantage tends to be self-reinforcing.** Each match generates information about what makes pairings succeed or fail in a given market. The broker's cross-market experience, whether it helps assess general counterparty quality or understand pairing complementarities, generates an informational advantage over each client's limited within-agent perspective. This advantage grows with the volume and diversity of the broker's placement history.

#### Proposition 1.2

**The broker's informational advantage arises primarily from understanding pairing complementarities (the relational channel) rather than from better assessing counterparty quality (the attributional channel), and the extent of the dominance of the relational channel depends on the structure of the data-generating process.**

The broker's advantage will be largest when the interaction component dominates (relational channel), rather than when the general quality component dominates (attributional channel). This difference is more or less pronounced depending on the structure of the data-generating process.

#### Proposition 1.3

**Network topology influences whether an active broker's structural or informational advantages dominate.**

**1.3a. In sparse or small-world networks, an active broker's value depends primarily on its structural advantage, which is correlated with its informational advantage.** Principals cannot easily find each other in such a network. Standard structural-hole measures (betweenness centrality, constraint, effective size) predict broker value well in this regime. At the same time, structural holes provide the access that feeds the broker's learning, so its structural and informational advantages are positively correlated.

**1.3b. In dense (or densifying) networks, an active broker's value depends primarily (or increasingly) on its informational advantage, which can become decoupled from its structural advantage.** Even if principals could find each other, a broker provides value if it can predict match quality better than principals can. A broker that started in a strong structural position may possess a lot of accumulated cross-market knowledge, regardless of its current structural advantage. As a broker's structural position weakens as a result of its brokerage activity, it accumulates more data, and the correlation between the two types of advantages weakens. In the limit, the broker has no structural advantage but retains its informational advantage.

### 2. Conditions for capture

#### Proposition 2.1

**A broker can leverage its informational advantage to transition from intermediary to principal (capture by the broker). The informational channel, not the structural one, drives capture.** A broker's information advantage translates into higher quality predictions of matching outputs and matches providing higher value to clients. Instead of continuing to facilitate these matches, the broker can leverage its predictive ability to capture the resource it was intermediating and become a principal.

At the point of capture, the broker's structural advantage may be declining (it has closed many of the gaps it once bridged). Standard structural-hole measures would predict that the broker is losing power exactly when it is consolidating it.

#### Proposition 2.2

**Capture requires specific conditions and does not occur universally. It is more likely when matching is complex and markets are opaque.**

High matching complexity make the principals' learning problem harder, widening and preserving the broker's advantage.

In markets with simple matching problems, principals learn fast and well enough that the broker does not accumulate a decisive informational advantage. Brokers persist as commodity intermediaries earning thin margins and may attempt capture but do not consolidate into dominant principals. This is the no-capture region of the parameter space.

Within the capture region of the parameter space, if the broker does not capture, its information advantage may start to erode over time.

### 3. Forms of capture

#### Proposition 3

**Capture can occur in two forms with qualitatively different dynamics.**

**3a. Under resource capture, the transition is abrupt, and the broker suddenly starts taking inventory risk and acting primarily as a principal.** Resource capture creates a triple lock-in: the client's information state freezes (it doesn't learn from new matches like it did when the broker acted as an intermediary), the client's network no longer grows (the broker is everyone's counterparty, so no direct ties form between principals), and the open market thins as the broker acquires counterparty positions (supply-side scarcity). The self-liquidating dynamic of structural advantage is suspended, because the broker no longer creates direct ties between clients. This produces a steep capture trajectory.

**3b. Under data capture, the transition is gradual, and the broker progressively monetizes its informational advantage by acting as a principal in subscription contracts.** Clients continue making new matches, learning from outcomes, and growing their networks. The self-liquidating dynamic of structural advantage continues operating. This produces a smooth capture trajectory.

## Illustrative Domains

The model is domain-agnostic: it formalizes brokered matching between heterogeneous agents in a single population. It represents any market in which a broker facilitates pairwise matches, and accumulates cross-market data from doing so while facing structural erosion from the direct ties it creates.

Because of its level of generality, the model can equally be taken to represent a variety of empirical domains in which real life brokers sometime transition to become principals; here I describe three of them, which I will refer to as illustrative examples throughout this section, to make things more concrete.

**Interdealer brokerage in OTC financial markets.** Dealers in over-the-counter markets (interest rate swaps, foreign exchange, corporate bonds) need counterparties for trades. Interdealer brokers (IDBs) sit between dealers, matching buy and sell interests across the market. Each successful brokered trade creates a direct relationship between two dealers who can subsequently trade bilaterally. The IDB accumulates cross-market knowledge of which dealer pairings clear efficiently. The well-documented transition from voice brokerage to electronic trading platforms (ICAP → NEX/CME, BGC → Fenics) is an instance of data capture; IDBs that became principal traders illustrate resource capture.

**Dealer networks in collectible markets.** Collectors of art, wine, rare books, or similar specialty goods seek trades or sales through dealers who know the market. Each collector has distinct tastes and holdings; match quality depends on multidimensional complementarity between what one party has and what another wants. Dealers accumulate knowledge of collector preferences across transactions. A dealer who transitions from pure intermediation to holding inventory (gallery, wine merchant) illustrates resource capture; one who builds a valuation database or subscription advisory service illustrates data capture.

**Import-export trading companies.** Producers and buyers across international markets rely on trading intermediaries to find counterparties they cannot easily reach or evaluate. Trading companies (*sōgō shōsha*, commodity brokers, Hong Kong trading houses) bridge geographically and informationally separated markets, matching exporters' goods with importers' needs. Match quality depends on multidimensional compatibility of product specifications, volumes, timing, and quality standards. Each successful brokered trade creates a direct relationship between producer and buyer who can subsequently trade bilaterally. The trading company's informational advantage lies in knowing which supplier-buyer combinations work across many markets. The transition from pure intermediation to taking principal positions (buying commodities from producers and reselling to buyers, bearing inventory and price risk) is the canonical resource capture trajectory. Some trading companies evolve further into vertically integrated conglomerates.

## Part I. Base Model

The model is a discrete-time agent-based simulation of a matching market with two participant types: *agents* and a *broker*. Agents seeking pairwise matches either search their own network or outsource the search to the broker. Each period represents one calendar quarter. All economic quantities (match output, fees, surplus) are in the same monetary units.

A single broker serves the market. This is a simplification: with multiple brokers, the data pool fragments, there is competition for informational rents, and no single broker consolidates as large an informational advantage. The model can be interpreted as a monopolistic broker or as a single broker's segment within a competitive market. Analysis of broker competition is deferred to future work.

All agents use heuristic decision rules. No agent holds beliefs about other agents' strategies, in line with the tradition of ABM agents using simple, bounded-rationality rules grounded in empirically observable behavior (Brenner, 2006).

The base model specifies agents (§0), the matching problem (§1), how agents learn to predict match quality (§2), match economics (§3), network structure and agent turnover (§4), how agents and the broker find counterparties (§5), the outsourcing decision (§6), the broker's roster (§7), the match lifecycle (§8), and the complete step ordering (§9). There is no capture in the base model. Resource capture is specified in Part III (§12).

### 0. Agents

The model has $N$ agents (default 1000) and a single broker. Agents are nodes in an undirected network $G$ that structures repeated search opportunities and structural position: direct ties in $G$ define the known-neighbor pool in self-search, while self-search can also sample a small set of non-neighbor strangers from the wider population (§5). The network is initialized as a small-world graph with random node ordering (no built-in type assortativity). It evolves over time as matches create new edges between matched agents (§4).

Each agent $i$ is characterized by:

- **Type** $\mathbf{x}_i \in \mathbb{R}^d$: a fixed vector of observable characteristics assigned at initialization. Types determine general quality and productive compatibility with other agents through the matching function (§1). The dimensionality $d = 8$ is fixed.
- **Current-period matches** $M_i^t$: the list of active bilateral relationships involving $i$ that have already formed in period $t$. Each counterparty can appear at most once in $M_i^t$ during a period. The length $|M_i^t| \leq K$ (default $K = 5$), so $K$ is the maximum number of distinct active counterparties an agent can maintain in a period.
- **Available capacity**: $K - |M_i^t|$, the number of additional distinct counterparties the agent can enter relationships with in the current period.
- **Experience history** $\mathcal{H}_{i}^t = \{(\mathbf{x}_j, q_{ij})\}$: the set of (other party's type, realized match output) pairs from all matches $i$ has participated in, regardless of whether $i$ was the demander or the counterparty (§2a). Because the matching function is symmetric (§1a), both roles produce the same prediction target.
- **Satisfaction indices** $s_{i,c}^t$: one scalar per search channel $c \in \{\text{self}, \text{broker}\}$, tracking realized match value via an EWMA (§6a). Drives the outsourcing decision (§6).

Agents exit independently each period with probability $\eta$ (default 0.02) and are replaced by new entrants with fresh types, empty histories, self-satisfaction initialized from neighbors' opinions, and broker-satisfaction set to the current broker reputation (§6a).

#### Agent types

Agents are described by type vectors in $\mathbb{R}^d$ ($d = 8$). These types are the observable characteristics that determine productive compatibility through the matching function (§1).

Agent types lie near a smooth one-dimensional curve on the surface of the unit sphere in $\mathbb{R}^d$. The curve is parameterized by a position $t \in [0, 1]$:

$$\mathbf{x}(t) = \frac{\tilde{\mathbf{x}}(t)}{\|\tilde{\mathbf{x}}(t)\|}, \qquad \tilde{x}_k(t) = \begin{cases} \sin(2\pi f_k t + \theta_k) & k = 1, \ldots, s \\ 0 & k = s+1, \ldots, d \end{cases}$$

where $f_k \sim U\{1, 2, 3, 4, 5\}$ are random integer frequencies and $\theta_k \sim U[0, 2\pi)$ are random phases, both drawn once per simulation, and $s \leq d$ is the number of **active dimensions** (the dimensions along which the curve has nonzero variation). The remaining $d - s$ dimensions receive only noise (see below).

Each agent is drawn at a random position $t_i \sim U[0,1]$ on the curve, then perturbed:

$$\mathbf{x}_i = \frac{\mathbf{x}(t_i) + \boldsymbol{\epsilon}_i}{\|\mathbf{x}(t_i) + \boldsymbol{\epsilon}_i\|}, \qquad \boldsymbol{\epsilon}_i \sim N\!\left(\mathbf{0}, \frac{\sigma_x^2}{d} \mathbf{I}_d\right)$$

The noise $\boldsymbol{\epsilon}_i$ is applied in all $d$ dimensions (including inactive ones), so that type vectors are not exactly confined to the $s$-dimensional subspace of the curve. The per-dimension noise scale $\sigma_x / \sqrt{d}$ is chosen so that the expected Euclidean distance from an agent to its curve position is approximately $\sigma_x$ regardless of $d$. The result is then re-projected to the unit sphere.

The parameter $s$ controls the complexity of the matching problem. When $s = d$, the curve spans all $d$ dimensions: agents nearby on the curve have similar types, while agents far apart point in genuinely different directions across all of $\mathbb{R}^d$. When $s < d$, the curve is confined to a lower-dimensional subspace.

#### Broker

A single broker serves the market. The broker is a permanent node in $G$, connected to all agents on its standing roster, all current-period broker clients, and, within a period, agents currently engaged in broker-channel matches. The broker is characterized by:

- **Experience history** $\mathcal{H}_b^t = \{(\mathbf{x}_i, \mathbf{x}_j, q_{ij})\}$: the set of (demander type, counterparty type, realized match output) triples from all matches the broker has mediated (§2c).
- **Roster** $\text{Roster}^t$: the set of agents the broker maintains as a standing access base. The roster is kept near a fixed target size through low exogenous churn and uniform replenishment (§7).
- **Current clients** $D^t$: the agents who outsource to the broker in period $t$. These current clients augment the broker's accessible counterparties for that period but do not persist as a lagged state variable (§5b, §7).
- **Reputation** $\text{rep}^t$: the average satisfaction of current client agents (§6).

### 1. The Matching Problem

The model's central dynamics depend on a matching problem: how valuable will the match between agents $i$ and $j$ be? No agent knows the answer in advance; all must learn it from experience.

The structure of the matching problem and how the broker and agents try to solve it determines whether and when the broker develops an informational advantage over the agents it serves.

**Match quality** (§1a) is symmetric: $f(\mathbf{x}_i, \mathbf{x}_j) = f(\mathbf{x}_j, \mathbf{x}_i)$. Match quality is a property of the pairing, not of which party initiated the match. It decomposes into two components:

- **General quality** (§1b): each party's baseline contribution to any match, independent of who the other party is.
- **Match-specific interaction** (§1c): how well this particular pairing works, depending on both parties' types.

The broker, which mediates matches across many agents, observes the same agent types producing different outcomes with different counterparties; whereas each agent only sees its own matching history.

#### 1a. Match quality

Let $q_{ij}$ represent the **per-period value of the match between agents $i$ and $j$**. It is a function of both agents' types, it is measured in monetary units, and it represents the economic value the match generates:

$$q_{ij} = Q + f(\mathbf{x}_i, \mathbf{x}_j) + \varepsilon_{ij}, \qquad
\varepsilon_{ij} \sim N(0, \sigma_\varepsilon^2)$$

where $Q = 1.0$ is a constant offset that shifts $q$ positive for downstream economic computations (surplus, fees, satisfaction), and $\sigma_\varepsilon = 0.10$. The noise term $\varepsilon_{ij}$ represents idiosyncratic match-specific variation (unobserved characteristics, timing, context) that is irreducible even with perfect knowledge of $f$.

The matching function $f: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ is unknown to all agents and fixed for the duration of the simulation. $f$ represents the pure signal structure of the data-generating process.

The deterministic matching function has two components, the first relating to each party's general quality and the second to their pairing complementarity:

$$f(\mathbf{x}_i, \mathbf{x}_j) = \rho \cdot \frac{1}{2}\!\left[\mathbf{x}_i^\top \mathbf{c} + \mathbf{x}_j^\top \mathbf{c}\right] + (1 - \rho) \cdot g(\mathbf{x}_i, \mathbf{x}_j) \cdot \mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j$$

where $\mathbf{c} \in \mathbb{R}^d$ is an ideal type vector (§1b), $\mathbf{A} \in \mathbb{R}^{d \times d}$ is a symmetric random interaction matrix (§1c), and $g(\mathbf{x}_i, \mathbf{x}_j)$ is a **regime-dependent gain** that modulates the interaction strength (§1c). The gain $g$ depends on a second symmetric operator $\mathbf{B}$ that determines whether a pairing is in a high-gain or low-gain regime. Because $\mathbf{A}$ and $\mathbf{B}$ are symmetric, $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j = \mathbf{x}_j^\top \mathbf{A} \mathbf{x}_i$ and $g(\mathbf{x}_i, \mathbf{x}_j) = g(\mathbf{x}_j, \mathbf{x}_i)$, so $f(\mathbf{x}_i, \mathbf{x}_j) = f(\mathbf{x}_j, \mathbf{x}_i)$.

The mixing weight $\rho$ (§1d) controls how much the general quality component contributes to total match output compared to the interaction component.

#### 1b. Agent general quality

General quality captures the portable value each party brings to any match, independent of who the counterparty is. Both parties contribute quality through their dot product with an **ideal type vector** $\mathbf{c} \in \mathbb{R}^d$. Agents whose types are aligned with $\mathbf{c}$ are high-quality counterparties in any match.

The vector $\mathbf{c}$ is drawn at initialization as a perturbation of a random point on the agent type curve with the same $\sigma_x / \sqrt{d}$ per-dimension noise used for regular agents.

A match between two high-quality agents produces a high quality component; a match involving a low-quality agent is penalized regardless of the other party's quality. 

#### 1c. Match-specific interaction

The match-specific interaction combines a base interaction with a regime-dependent gain: $g(\mathbf{x}_i, \mathbf{x}_j) \cdot \mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j$.

**Base interaction.** The bilinear form $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j$ measures the complementarity of the pairing. The interaction matrix $\mathbf{A} \in \mathbb{R}^{d \times d}$ is symmetric positive definite (SPD), constructed as $\mathbf{A} = \mathbf{M}_A^\top \mathbf{M}_A \cdot (d / \text{tr}(\mathbf{M}_A^\top \mathbf{M}_A))$ where $\mathbf{M}_A$ has iid $N(0,1)$ entries. The trace normalization ensures $\text{tr}(\mathbf{A}) = d$, so for a random unit vector $\mathbf{x}$ drawn isotropically on the sphere, $E[\mathbf{x}^\top \mathbf{A} \mathbf{x}] = \text{tr}(\mathbf{A})/d = 1$. This fixes the average quadratic scale of the interaction operator. $\mathbf{A}$ is fixed for the duration of the simulation.

Because $\mathbf{A}$ is symmetric, $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j = \mathbf{x}_j^\top \mathbf{A} \mathbf{x}_i$, so the base interaction is symmetric without explicit symmetrization. Because $\mathbf{A}$ is positive definite, all of its eigenvalues are strictly positive, hence $\mathbf{A}$ is full rank and defines a nondegenerate quadratic form on $\mathbb{R}^d$. The trace normalization fixes only the average eigenvalue at 1, it does not impose any particular condition number. Writing the bilinear form in coordinates,

$$
\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j = \sum_{k=1}^d \sum_{l=1}^d A_{kl} x_{i,k} x_{j,l},
$$

shows that a symmetric $\mathbf{A}$ contributes $d(d+1)/2$ free coefficients.

**Regime-dependent gain.** A second symmetric matrix $\mathbf{B} \in \mathbb{R}^{d \times d}$ determines a gain that amplifies or attenuates the base interaction:

$$g(\mathbf{x}_i, \mathbf{x}_j) = 1 + \delta \cdot \text{sign}(\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j)$$

where $\delta \in (0, 1)$ (default 0.5) controls the gain strength. Because $\mathbf{B}$ is symmetric, $g(\mathbf{x}_i, \mathbf{x}_j) = g(\mathbf{x}_j, \mathbf{x}_i)$. Pairings divide into two regimes: when $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j > 0$, the gain is $(1 + \delta)$ (high-gain regime); when $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j < 0$, the gain is $(1 - \delta)$ (low-gain regime). At $\delta = 0.5$, the high-gain interaction is three times the low-gain interaction.

The implementation uses the approved `cov_full` construction. Let

$$\mathbf{S}_x = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i \mathbf{x}_i^\top$$

be the empirical second-moment matrix of realized agent types. First draw a symmetric Gaussian matrix $\mathbf{H}$ and recenter it to zero trace, then remove its weighted projection onto $\mathbf{A}$ under the inner product

$$\langle \mathbf{M}, \mathbf{N} \rangle_{\mathbf{S}_x} = \operatorname{tr}(\mathbf{S}_x \mathbf{M} \mathbf{S}_x \mathbf{N}).$$

Specifically,

$$\mathbf{B}_{\text{raw}} = \mathbf{H} - \frac{\operatorname{tr}(\mathbf{S}_x \mathbf{H} \mathbf{S}_x \mathbf{A})}{\operatorname{tr}(\mathbf{S}_x \mathbf{A} \mathbf{S}_x \mathbf{A})} \mathbf{A}, \qquad \mathbf{B} = \frac{\mathbf{B}_{\text{raw}}}{\lVert \mathbf{B}_{\text{raw}} \rVert_F}.$$

This construction makes $\mathbf{B}$ weighted-orthogonal to $\mathbf{A}$ under the bilinear form defined above, $\operatorname{tr}(\mathbf{S}_x \mathbf{B} \mathbf{S}_x \mathbf{A}) = 0$, while preserving symmetry. $\mathbf{B}$ is generally indefinite, not SPD. This is intentional: only the sign of $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j$ matters for regime assignment, so the orientation of the separating operator matters, not positive definiteness.

The gain modulates the *strength* of the base interaction without changing its sign. Among pairings with similar base interactions $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j$, those in the high-gain regime are worth substantially more than those in the low-gain regime. This difference is the source of the broker's informational advantage (§1e).

#### 1d. What controls the nature of the matching problem

- **$s$ (active dimensions).** When $s = d$, the type curve spans all $d$ dimensions, creating maximum diversity in the type space and the interaction effects that depend on it. When $s < d$, the curve is confined to a lower-dimensional subspace.

- **$\rho$ (mixing weight).** At high $\rho$, general quality dominates. At low $\rho$, the gain-modulated interaction dominates.

- **$\delta$ (gain strength).** Controls the magnitude of the regime effect. At $\delta = 0$, the gain is 1 for all pairings and the DGP reduces to a simple interaction without regimes. At $\delta > 0$, the true interaction results from a mixture of two regimes. Larger $\delta$ produces a larger gap between high-gain and low-gain pairings, making the regime more consequential for match rankings.

- **$\mathbf{A}$ and $\mathbf{B}$ (interaction and regime operators).** $\mathbf{A}$ determines the base interaction structure; $\mathbf{B}$ determines the regime boundary. $\mathbf{A}$ is SPD. $\mathbf{B}$ is symmetric and constructed to be weighted-orthogonal to $\mathbf{A}$ under the realized type second moment $\mathbf{S}_x$. For a fixed agent $i$, the base interaction $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j = \mathbf{a}_i^\top \mathbf{x}_j$ (where $\mathbf{a}_i = \mathbf{A} \mathbf{x}_i$) is linear in $\mathbf{x}_j$. The regime boundary ($\mathbf{b}_i^\top \mathbf{x}_j = 0$, where $\mathbf{b}_i = \mathbf{B} \mathbf{x}_i$) is therefore orthogonalized away from the payoff operator in the weighted geometry induced by realized types, reducing systematic alignment between payoff ranking and regime assignment.

- **$\sigma_\varepsilon$ (noise scale).** The match-level noise $\sigma_\varepsilon = 0.10$ should be interpreted relative to the actual variance of $f$, which depends on the parameter configuration. The typical magnitude of dot products on the unit sphere in $\mathbb{R}^d$ is $O(1/\sqrt{d})$. The effective signal-to-noise ratio should be measured empirically at initialization.

#### 1e. The information gap between single- and cross-agent data

The regime-dependent gain (§1c) creates an informational gap between single-agent and cross-agent data. This gap has three important characteristics:

1. The gap is inherent to the DGP, not purely model-related.

2. The gap is fundamental, not merely statistical.

    - A purely statistical advantage depends asymptotically on data volume only and erodes as agents accumulate data. A fundamental gap involve an identification problem that single-agent data cannot solve regardless of sample size.

3. The gap affects match selection.

    - Agents use predictions to select a best counterparty ($\arg\max$).
    - The gap causes single- and cross-agent data to produce *different rankings* among top candidates, not just more accurate point estimates or better predictions for candidates that would never selected.

These characteristics correspond to assumptions being made through model design.

**Why the regime creates a fundamental information gap.** For a fixed agent $i$, the gain-modulated interaction produces outcomes from a *mixture* of two linear functions of $\mathbf{x}_j$. Some partners are in the high-gain regime ($g = 1 + \delta$) and others in the low-gain regime ($g = 1 - \delta$), but it is hard for agent $i$ to determine which regime each match fell into. The regime boundary (where $\mathbf{b}_i^\top \mathbf{x}_j = 0$, with $\mathbf{b}_i = \mathbf{B}\mathbf{x}_i$ because $\mathbf{B}$ is symmetric) is along a direction in $\mathbf{x}_j$ space that the agent does not know. 

The agent data is generated by the mixture:

$$q_{ij} \approx \begin{cases} (1 + \delta) \cdot \mathbf{a}_i^\top \mathbf{x}_j & \text{if } \mathbf{b}_i^\top \mathbf{x}_j > 0 \\ (1 - \delta) \cdot \mathbf{a}_i^\top \mathbf{x}_j & \text{if } \mathbf{b}_i^\top \mathbf{x}_j < 0 \end{cases}$$

(omitting the quality component and noise for clarity). 

A linear model $\boldsymbol{\beta}^\top \mathbf{x}_j$ fitted on this mixture learns an *average slope* that is systematically wrong for both regimes. To separate the two regimes, the agent needs to detect that the slope of the relationship between $\mathbf{x}_j$ and outcomes changes along the direction $\mathbf{b}_i$. This is an unsupervised change-point detection problem in $d$ dimensions that requires qualitatively more sophisticated analysis than simple linear regression, regardless of sample size.

**Why the broker can resolve the regime.** The broker observes $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ triples across many different pairings. The regime depends on $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j$, which is a *bilinear* function of both types. The broker's network, taking $[\mathbf{x}_i; \mathbf{x}_j]$ as input (§2c), can learn that certain combinations of $(\mathbf{x}_i, \mathbf{x}_j)$ systematically produce higher or lower match quality, capturing the regime effect through the nonlinear interactions learned by its hidden layer. The regime boundary that is hidden from the individual agent (because it requires conditioning on $\mathbf{x}_i$, which is fixed in single-agent data) is *visible* in the broker's input (because $\mathbf{x}_i$ varies across observations in cross-agent data).

**Why the gap affects match selection.** The gain operates multiplicatively on the base interaction. Among an agent's candidates with a high true interaction component, some are in the high-gain regime and others in the low-gain regime. The agent's linear model, fitting the average slope, ranks these candidates similarly. But their true match qualities differ by a factor of $(1 + \delta) / (1 - \delta)$ (3:1 at $\delta = 0.5$). The broker, knowing the regime, can identify which top candidates are high-gain and rank them above the low-gain candidates. This produces *different selections* from the same candidate pool.

### 2. Learning

Each period, agents and the broker **fit** prediction models on their accumulated histories to learn from past outcomes, then **use** those models to rank candidates and make decisions. Models are fitted once per period (§9, Step 2.1), then used repeatedly during candidate evaluation and match formation (§9, Steps 2.2–3).

#### 2a. Architecture and fitting

Both agents and the broker use the same architecture: a fully-connected network with one hidden layer, ReLU activations, and a single linear output:

$$\hat{q}(\mathbf{z}) = \mathbf{w}_2^\top \text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) + b_2$$

where $\mathbf{z}$ is the input feature vector, $\mathbf{W}_1$ is the hidden-layer weight matrix, $\mathbf{b}_1$ is the hidden bias vector, $\mathbf{w}_2$ is the output-layer weight vector, and $b_2$ is the output bias. Neither agents nor the broker use hand-crafted features; both receive raw type vectors as input and learn the relevant structure from data.

**Fitting.** Each period, the network weights are updated by minimizing MSE using vanilla gradient descent on the full batch with a fixed learning rate $\eta_{lr}$ (default 0.03). No explicit regularization is applied: at the data scales the broker and agents accumulate, adding weight decay has no measurable effect on held-out fit, so it is omitted for simplicity.

**Initialization.** At $t = 0$ each network is trained from random weights for $E_{\text{init}}$ gradient steps (default 200) on its seed history. The output bias $b_2$ is initialized to $Q$ (the DGP offset, §1a) rather than zero so that an untrained network predicts the population-mean match quality. This avoids a large negative-bias artifact for fresh entrants whose network has not yet been trained, and is irrelevant for mature networks (the first training steps on any real data move $b_2$ to its fitted value). All other weights follow He initialization.

**Adaptive schedule.** The network is updated each period with warm start from the previous period's weights. The number of gradient steps adapts to the ratio of new observations to total history:

$$E_t = \max\!\left(50, \; \left\lceil E_{\text{init}} \cdot \frac{n_{\text{new}}}{n_{\text{total}}} \right\rceil\right)$$

where $n_{\text{new}}$ is the number of observations added this period and $n_{\text{total}} = |\mathcal{H}^t|$ is the current history size. The floor of 50 ensures that mature networks continue to receive enough gradient updates per period to converge close to the DGP's best-achievable fit, rather than stagnating far below it.

**Training window.** To avoid diluting new observations in a large full-batch gradient, each training period uses at most the $W = 500$ most recent observations from the agent's or broker's history. The warm start preserves what was learned from older data. This sliding window ensures that the gradient reflects recent experience while being large enough, after symmetry augmentation for the broker, to contain a representative cross-section of match types.

**Prediction.** Given a fitted network, the prediction for a candidate match is a single forward pass. An agent evaluates $\hat{q}_i(\mathbf{x}_j)$ for each candidate partner $\mathbf{x}_j$ and selects the candidate with the highest predicted quality ($\arg\max$). Because $f$ is symmetric, the same model serves both roles: evaluating potential counterparties (as demander) and evaluating incoming proposals (as counterparty). The broker evaluates $\hat{q}_b([\mathbf{x}_i; \mathbf{x}_j])$ for all (demander, available roster member) pairs and allocates greedily (§5b).

#### 2b. Agent $i$'s model

**History.** $\mathcal{H}_i^t = \{(\mathbf{x}_j, q_{ij})\}_{m=1}^{n_i}$ records the other party's type and the realized match output from every match $i$ has participated in, regardless of role. Because $f$ is symmetric (§1a), observations from both roles pool into a single history.

**Input and capacity.** The agent's network takes the partner's type as input: $\mathbf{z} = \mathbf{x}_j$ ($d = 8$ inputs). The hidden layer has $h_a = 16$ units. Total parameters: $h_a \cdot (d + 1) + (h_a + 1) = 161$.

**Why this architecture.** The regime-dependent gain (§1c) makes the agent's local prediction problem *piecewise linear*: for a fixed agent $i$, the target function is $f_i(\mathbf{x}_j) \approx (1 \pm \delta) \cdot \mathbf{a}_i^\top \mathbf{x}_j + \text{quality}$, with two different slopes on either side of a hyperplane boundary $\mathbf{b}_i^\top \mathbf{x}_j = 0$ that the agent does not know. A one-hidden-layer ReLU network is the natural function approximator for this structure: each ReLU unit computes a hinge function $\max(\mathbf{w}^\top \mathbf{x}_j + b, 0)$, and a small number of such units can represent piecewise linear functions with learned breakpoints.

With sufficient data, the network can in principle learn the piecewise linear structure. The ReLU units can discover the regime boundary as one of their activation thresholds. In practice, with sparse data, the network produces a smooth approximation that averages across regimes. As the agent accumulates observations, the approximation improves, but the identification problem persists: detecting a change in slope along an unknown direction in $\mathbb{R}^d$ from noisy data requires substantially more observations than fitting a single linear relationship.

#### 2c. Broker's model

**History.** $\mathcal{H}_b^t = \{(\mathbf{x}_i, \mathbf{x}_j, q_{ij})\}_{m=1}^{n_b}$ records both parties' types and the realized match output from every match the broker has mediated. Seeded at initialization from random roster member pairs (§11c).

**Input and capacity.** The broker's network takes both parties' types as input: $\mathbf{z} = [\mathbf{x}_i; \mathbf{x}_j]$ ($2d = 16$ inputs). The hidden layer has $h_b = 32$ units. Total parameters: $h_b \cdot (2d + 1) + (h_b + 1) = 577$. No hand-crafted features (such as outer products) are provided.

**Fitting.** The network must discover the bilinear interaction structure $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j$ and the regime boundary $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j$ from the raw concatenated inputs. With 32 hidden ReLU units, the network has sufficient capacity to approximate these bilinear forms. Each unit computes a piecewise linear function of $[\mathbf{x}_i; \mathbf{x}_j]$ that can represent products of input components through interactions between units. 

To exploit the symmetry of $f$, the broker augments its training data by including both orderings of each observation: for each $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ in $\mathcal{H}_b$, the broker trains on both $[\mathbf{x}_i; \mathbf{x}_j]$ and $[\mathbf{x}_j; \mathbf{x}_i]$ with the same target $q_{ij}$. This doubles the effective training set and ensures the network learns that the two input slots are interchangeable.

**Data scope.** The broker learns only from matches it mediates. It does not observe outcomes of self-search matches. After a brokered match forms, the realized output $q_{ij}$ is observed by all parties involved (the two agents and the broker), and the broker adds $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$.

#### 2d. The asymmetry between agents and the broker

The broker's advantage has two components:

1. **More data.** The broker accumulates observations across all client agents, giving it far more training examples than any individual agent.

2. **Regime identification from cross-agent variation.** The regime-dependent gain (§1c) creates a mixture in each agent's data: some observations come from high-gain pairings and others from low-gain pairings. The broker, by observing the *same* partner types producing *different* outcomes with *different* agents, can detect the regime structure. The regime depends on the interaction between both parties' types ($\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j$), which is visible only when $\mathbf{x}_i$ varies across observations.

An agent learns "what kind of partner works well for me" from a small, agent-specific sample, but its model averages across regimes. The broker learns "what kind of pairings work well" from a large, cross-market sample where both parties' types vary. The broker's advantage is both informational (it can identify regimes that agents cannot) and statistical (it has more data).

### 3. Match Economics

Matches form when both parties expect positive gains from trade, following a heuristic version of the standard search-and-matching framework (Rogerson, Shimer & Wright, 2005).

#### 3a. Outside options

All agents share a common outside option $r$: the minimum per-period match value an agent requires to participate. Below this threshold, the agent prefers to remain unmatched. The outside option is calibrated at initialization:

$$r = 0.60 \cdot \bar{q}_{\text{cal}}$$

where $\bar{q}_{\text{cal}}$ is the mean match output computed from a Monte Carlo sample (§11c). The 0.60 calibration sets the outside option at 60% of average match value, producing a market where approximately 40% of match output is surplus available for gains from trade. A constant $r$ means the profitability comparison is the same for every counterparty.

#### 3b. Participation constraints

A match between demander $i$ and counterparty $j$ forms only if both parties predict positive gains:

- **Demander**: $\hat{q}_{i}(\mathbf{x}_j) > r$
- **Counterparty**: $\hat{q}_{j}(\mathbf{x}_i) > r$

When the broker proposes a match, it applies the constraint using its own prediction: $\hat{q}_b([\mathbf{x}_i; \mathbf{x}_j]) > r$. The counterparty still evaluates independently using its own model.

#### 3c. Search costs

The model uses a single calibrated friction level for both channels:

$$
\phi = c_s = \lambda_c \cdot (\bar{q}_{\text{cal}} - r).
$$

At the baseline $\lambda_c = 0.15$, the common friction level is $0.15\cdot(\bar{q}_{\text{cal}} - r)$. The distinction between $\phi$ and $c_s$ is therefore not their magnitude, but how that same friction enters realized payoffs:

- **Self-search** labels the common friction as $c_s$ and charges it on each demanded relationship position routed through self-search, whether or not that position is filled.
- **Standard brokerage** labels the same friction as $\phi$ and charges it only on successful **standard** brokered placements.

The common friction is independent of realized match quality. Under principal mode (§12), no $\phi$ is charged to the demander because the broker is no longer acting as a pure intermediary.

An economically important asymmetry in the illustrative markets is **search-risk transfer**. Self-search typically requires the agent to incur time, attention, or internal business-development costs for each sought relationship position whether or not the search succeeds: calling dealers, screening counterparties, traveling to trade events, preparing offers, or canvassing foreign buyers. By contrast, broker compensation is often at least partly contingent on success: a broker or intermediary is usually paid when a relationship clears, not merely for having searched. In that sense, outsourcing shifts part of the risk of failed search from the agent to the intermediary. This creates a motive for brokerage that is distinct from pure informational superiority. Even when the broker and the agent faced the same cost level $\lambda_c$, the broker could still be valuable by absorbing failed-search risk.

### 4. Network Structure and Turnover

Agents interact through a single undirected network $G$ that determines each agent's search opportunities and structural position.

#### 4a. Network initialization

$G$ is initialized as a small-world graph (Watts & Strogatz, 1998). Agents are arranged on a ring in random order, each connected to its $k = 6$ nearest neighbors on the ring, and each edge is rewired with probability $p_{\text{rewire}} = 0.1$. This produces the high clustering and short path lengths characteristic of small-world graphs. Agents are placed on the ring in random order (rather than, e.g., sorted by type) so that the initial network is not type-assortative: neighbors at $t = 0$ are representative of the broader population, which avoids inflating baseline match quality through an artificially favorable neighborhood structure. An optional PC1-sorted variant is retained for robustness checks.

The broker is a permanent node in $G$, connected to all standing roster members, all current-period broker clients, and agents currently engaged in broker-channel matches (§7). The broker node has no type vector and is excluded from matching candidate pools, but is included in network measure computations (§10).

#### 4b. Match tie formation

Each realized match (whether through self-search or brokered) adds an undirected edge between the demander and counterparty in $G$, if one does not already exist. Ties persist unless one of the nodes exits, as former counterparties remain connected after their match dissolves. This is the only mechanism of network densification.

#### 4c. Agent turnover

Agents exit independently each period with probability $\eta$ (default 0.02), yielding an expected agent lifetime of 50 quarters (12.5 years).

Exiting agents are replaced by entrants with fresh types sampled from the curve at a random position $t \sim U[0,1]$ plus noise (same procedure as initialization), empty experience histories, self-satisfaction initialized from new neighbors' self-satisfaction (word-of-mouth), and broker-satisfaction set to the current broker reputation. The exiting agent's node in $G$ is removed (along with all its edges). 

The entrant is added to $G$ with $\lfloor k/2 \rfloor$ edges to agents sampled from the type neighborhood (probability $\propto \exp(-\|\mathbf{x}_{i'} - \mathbf{x}_j\|^2)$). Entrants join with fewer connections than the initial network degree $k$ to reflect the disadvantage of being new to a market: established agents have accumulated connections through prior matches, while entrants start with only a few type-similar contacts. New entrants with sparse networks are more likely to need the broker's matching service.

### 5. Search

At the start of each period, all $K$ relationship positions are open. Each position independently generates demand with probability $p_{\text{demand}}$ (default 0.50), so agent $i$ draws desired relationship demand $d_i \sim \text{Binomial}(K,\; p_{\text{demand}})$. If $d_i > 0$, the agent chooses **one channel for the batch** of current-period demand (§6): self-search or broker. Conditional on that batch decision, the chosen channel attempts to fill the batch through a finite sequence of within-period rounds rather than through a single pooled proposal pass.

Let $u_i^0 = d_i$ denote agent $i$'s remaining unfilled demand at the start of within-period matching. Round $\ell$ gives every still-active demander with $u_i^{\ell-1} > 0$ one opportunity to fill **one** additional relationship position through its chosen channel. If agent $i$ secures a match in round $\ell$, then $u_i^\ell = u_i^{\ell-1} - 1$; otherwise $i$ either continues to its next feasible candidate within the same round or, if it exhausts that list, exits matching for the rest of the period with remaining demand unfilled.

#### 5a. Self-search

In each round, agent $i$'s self-search candidate pool has two components, evaluated using current capacities after all previously accepted rounds have been finalized:

**Known neighbors.** Direct network neighbors in $G$ with available capacity, no active current-period relationship with $i$, and at least one previously observed match with $i$ (equivalently, a stored partner mean). For each such neighbor $j$, the agent evaluates quality using the **average of realized outcomes** from prior matches with $j$: $\bar{q}_{ij} = \frac{1}{n_{ij}} \sum q_{ij}^{(m)}$, where $n_{ij}$ is the number of times $i$ and $j$ have matched. This is a direct empirical estimate, not a model prediction. Not every graph neighbor is known in this sense: the initial network contains edges created by the network initialization, but each agent's seed history records only a subset of neighbor pairings (§11c). Neighbors with no stored partner mean are omitted from the known-neighbor component rather than being reclassified as strangers.

**Strangers.** $\min(n_s, |\text{eligible}|)$ agents sampled uniformly from the population (excluding current neighbors, current-period counterparties, and the broker node), where $n_s = 5$ (default) and eligible agents are those with available capacity. The agent has no prior history with these candidates and evaluates them using its **prediction model**: $\hat{q}_i(\mathbf{x}_j)$ (§2b). Strangers represent cold outreach: attending trade events, browsing listings, or following up on indirect referrals.

Within a round, agent $i$ orders all feasible self-search candidates by this demander-side evaluation, dropping any candidate whose evaluation fails the demand-side participation constraint $\hat{q}_i(\mathbf{x}_j) \le r$ (§3b). If the agent is rejected by the highest-ranked candidate, it immediately tries the next-best candidate in the same round, and so on until it either secures a tentative hold or exhausts its round-specific candidate list.

#### 5b. Broker-mediated search

When agent $i$ outsources to the broker, the broker includes agent $i$ in its client list for the current period. Outsourcing does not alter standing roster membership (§7), but current broker clients are added to the broker's one-period access overlay. The broker therefore allocates over a hybrid access set rather than over the standing roster alone.

At the end of Step 1 (after all outsourcing decisions), the broker observes its full client list $D^t$ (the set of demanders who outsourced this period) and forms its accessible counterparty set

$$A^t = \text{Roster}^t \cup D^t.$$

In each round, the broker considers the currently available accessible counterparties

$$A^t \cap \{\text{agents with available capacity after earlier accepted rounds and no active current-period relationship with } i\}.$$

For every active broker-client demander $i$ in the round and every accessible counterparty $j \in A^t$ with current capacity and no active current-period relationship with $i$, the broker computes predicted match quality $\hat{q}_b([\mathbf{x}_i; \mathbf{x}_j])$. It then constructs an ordered list of feasible candidates for each outsourced demander, ranked by the broker's prediction, dropping any candidate with non-positive predicted surplus $\hat{q}_b([\mathbf{x}_i; \mathbf{x}_j]) \le r$. If the broker's top candidate for demander $i$ is rejected within the round, the broker immediately tries the next-best candidate for the same demander in that round, and so on until demander $i$ either secures a tentative hold or exhausts its feasible list.

Agents already on the standing roster remain on it whether or not they outsource in the current period; current clients expand access only for the current period and do not become lagged standing members for that reason.

**Implementation note (exact-preserving).** The code may realize these same rules with performance-oriented scratch buffers and caches, provided the stochastic object is unchanged: self-search strangers are still sampled uniformly without replacement from the current eligible set, broker-side candidate rankings still reflect the same round-specific ordering implied by $\hat{q}_b$, current-period pair exclusion may be implemented with a period-local exact index of $M_i^t$ rather than repeated scans of match lists, and neural-network training still uses the same data windows and gradient steps. These implementation details are not separate model assumptions.

#### 5c. Within-round proposal and acceptance

Within a round, all still-active demanders attempt to fill one relationship position through a decentralized deferred-acceptance protocol with capacity.

1. Each active demander proposes to its highest-ranked not-yet-rejected feasible candidate under its chosen channel.
2. Each counterparty $j$ evaluates the offers it receives that round:
   - $j$ uses its own evaluation rule, $\bar{q}_{ji}$ for known partners and $\hat{q}_j(\mathbf{x}_i)$ for strangers, and rejects any offer with evaluation $\le r$.
3. Counterparty $j$ tentatively holds up to its remaining capacity's worth of incoming offers, ranked by its counterparty-side evaluation. Lower-ranked incoming offers are rejected.
4. Any rejected demander immediately proposes to its next-best feasible candidate in the same round.
5. Steps 2-4 repeat until no rejected demander has any feasible candidate left to try.

Three capacity rules matter. First, a counterparty can hold only up to its current spare capacity. Second, an agent can be both a demander and a counterparty in the same round, but both roles use the same total capacity $K$. Accordingly, if agent $i$ secures a tentative demander-side match while already tentatively holding incoming counterparty offers up to capacity, the lowest-ranked incoming counterparty hold is released so that total tentative commitments for $i$ do not exceed its current spare capacity. Third, a pair can hold at most one current-period relationship: repeated and reciprocal same-pair proposals are rejected or discarded before realization.

A demander **fails in the round** if it exhausts its feasible candidate list without securing a tentative hold. Failure is terminal for the rest of the period: because previously accepted rounds only reduce capacities, no later round can create a newly feasible opportunity that was absent when the demander exhausted its current round-specific list.

At the end of the round, all tentative holds are finalized as accepted matches. Realized outputs are drawn, histories and partner means are updated immediately, and any standard match adds an edge in $G$. These updates feed into later rounds in the same period.

The within-period process stops when all demand is filled, when a round produces no accepted matches, or when no still-unfilled demander has any feasible candidate left. Because a demander can fill at most one relationship position per round, the maximum number of rounds in a period is

$$
R_{\max} = \max_i d_i \le K.
$$

Under the baseline $K = 5$, a period therefore contains at most 5 within-period matching rounds.

### 6. The Outsourcing Decision

A **calibration reference** $\bar{q}_{\text{cal}} = E[q]$ is computed once at initialization from a Monte Carlo sample of random agent pairs (§11c). This is the unconditional mean match output, used to scale the reservation value $r$, broker fee $\phi$, and self-search cost $c_s$ (§11b). It is not used to initialize satisfaction indices or broker reputation; those are initialized from actual seed data (see below).

#### 6a. Satisfaction tracking

Each agent $i$ maintains a satisfaction index $s_{i,c}^t$ for each search channel $c \in \{\text{self}, \text{broker}\}$. These scores summarize past matching outcomes and drive the outsourcing decision.

The index is an exponentially weighted moving average (recency weight $\omega = 0.2$) of realized match value, net of search costs:

$$s_{i,c}^{t+1} = (1 - \omega)\,s_{i,c}^t + \omega \cdot \tilde{q}$$

where $\tilde{q}$ is the satisfaction input for the period. The averaging unit is the agent's **requested relationship demand** $d_i$: realized outcomes from accepted matches are summed, unfilled requested positions contribute zero output, and the total is divided by $d_i$. This makes partial fill mechanically lower satisfaction relative to full fill.

| Channel | Satisfaction input $\tilde{q}$ |
|---------|-------------------------------|
| Self-search | $\dfrac{\sum q_{ij} - c_s \cdot d_i}{d_i} = \dfrac{\sum q_{ij}}{d_i} - c_s$, summing over accepted self-search relationships |
| Standard brokered (base model) | $\dfrac{\sum (q_{ij} - \phi)}{d_i}$, summing over accepted brokered relationships |
| Broker channel under principal mode (M1, §12) | $\dfrac{\sum_{\text{standard}} (q_{ij} - \phi) + \sum_{\text{principal}} q_{ij}}{d_i}$ |

This implies an intentional asymmetry in total-failure episodes. If a brokered batch fails completely, then $\tilde{q}=0$ and broker satisfaction decays toward zero. If a self-search batch fails completely, then $\tilde{q}=-c_s$ because the per-position search effort was paid despite filling no position. Satisfaction indices are not floored: they can go negative. The EWMA's recency weighting ensures recovery from negative values within a few good observations.

Under the approved simplification, $s_{i,\text{self}}^t$ is interpreted as the reduced-form value of the entire internal-search channel. It summarizes realized self-search outcomes, including cases where the agent reused known partners directly, rather than separating out a distinct contemporaneous "known partners" score at decision time.

**Initialization from seed data.** At initialization, each agent's self-satisfaction is set to the mean of its seed match outcomes (§11c, step I.10), not to an arbitrary constant. Each agent's broker-satisfaction is set to the broker's seed-data reputation (§6c). This grounds the initial outsourcing decision in actual data: agents with good neighbors start with high self-satisfaction and are harder for the broker to recruit, while agents with poor neighbors are more open to outsourcing.

**Fresh entrants.** New agents entering via turnover (§4) initialize self-satisfaction as the mean of their new neighbors' self-satisfaction (word-of-mouth: the entrant inherits the local opinion about self-search quality). Broker-satisfaction is set to the current broker reputation (the market's current opinion). The `tried_broker` flag is false, so the entrant uses broker reputation for its first outsourcing decision.

**`tried_broker` flag semantics.** The flag flips from false to true the first time the agent chooses the broker channel for any demand in a period, regardless of whether the broker's proposal led to a successful placement. Once true, the agent uses its own $s_{i,\text{broker}}^t$ rather than the broker's reputation for subsequent decisions (§6b). The rationale is that after selecting the broker once, the agent's personal EWMA has started to absorb information about that channel, including failed broker episodes that update satisfaction through a zero realized input, so reputation stops being the better signal.

#### 6b. Decision rule

Each period, an agent with $d_i$ requested relationship positions compares two scores:

- **$\text{score}_{\text{self}}$** $= s_{i,\text{self}}^t$: the EWMA satisfaction from past self-search outcomes. This is a reduced-form internal-search score and is interpreted as already incorporating the value of exploiting known partners under the self-search channel.
- **$\text{score}_{\text{broker}}$** $= s_{i,\text{broker}}^t$ if the agent has tried the broker, otherwise the broker's reputation $\text{rep}_b^t$.

The agent outsources if $\text{score}_{\text{broker}} > \text{score}_{\text{self}}$; it self-searches if $\text{score}_{\text{broker}} < \text{score}_{\text{self}}$. At the boundary $\text{score}_{\text{broker}} = \text{score}_{\text{self}}$, the channel is chosen by a uniform coin flip between self-search and broker.

This simplification treats the self-search channel as a single reduced-form outside option. Agents do not separately compute a contemporaneous "best known partners" score at decision time; instead, the value of having discovered good partners is assumed to be reflected over time in realized self-search outcomes and therefore in $s_{i,\text{self}}^t$.

The search-risk-transfer asymmetry sharpens this comparison. Self-search exposes the agent to the risk of paying for effort on requested relationship positions that yield no placement, whereas standard brokerage shifts more of that downside onto the intermediary because compensation is tied more closely to successful matching. As a result, outsourcing can be attractive not only because the broker has better information or broader access, but also because it converts some search cost from a non-contingent expenditure into a contingent payment. This mechanism is especially relevant for agents facing uncertain fill rates, sparse networks, or highly lumpy demand.

**Initial conditions.** Self-satisfaction is initialized from each agent's seed match outcomes (mean of 5 neighbor pairings). Broker-satisfaction is initialized to the broker's seed-data reputation (mean of 100 seed broker match outcomes). Both values are grounded in actual data, not an arbitrary constant. Since the broker's seed reputation and the typical agent's seed self-satisfaction are close but not identical, the first period's outsourcing decisions reflect genuine (if noisy) differences in local match quality rather than a symmetric coin flip. Agents with above-average self-satisfaction prefer self-search; those with below-average self-satisfaction are more open to outsourcing. The broker's early client base is thus self-selected rather than random.

#### 6c. Broker reputation

$$\text{rep}_b^{t+1} = \begin{cases} \frac{1}{|D_b^t|} \sum_{i \in D_b^t} s_{i,b}^{t+1} & \text{if } D_b^t \neq \emptyset \\[4pt] \text{rep}_b^{t} & \text{otherwise} \end{cases}$$

where $D_b^t$ is the set of agents who outsourced to the broker this period. When the broker has current clients, reputation is updated to the mean of their (post-update) broker satisfaction. When it has no clients, the value is held from the previous period. Reputation is initialized from the mean of the broker's seed match outcomes (§11c, step I.9).

### 7. Broker Roster

The broker maintains a **roster** of agents it knows and can propose as counterparties when mediating matches.

**Initialization.** The roster is seeded with a fixed target size

$$R^* = \lceil \alpha_R N \rceil, \qquad \alpha_R = 0.20,$$

by drawing $R^*$ agents uniformly at random from the population (default 200 at $N = 1000$). This ensures the broker can serve early outsourcers without frequent no-match failures that would drive broker satisfaction down before the broker has a chance to demonstrate value. The broker's history is seeded with observations from random roster member pairs in $G$ (§11c).

**Standing roster with replenishment.** The broker maintains this roster as a standing access base. At the start of each period, after prior-period active matches are cleared and before current-period demand is realized (§9, Step 0.2), each current roster member independently exits the roster with exogenous probability $p_{\text{roster}}$ (default $0.02$). The broker then replenishes uniformly at random from agents not currently on the roster until the target size $R^*$ is restored. Formally, if $\widetilde{\text{Roster}}^t$ is the post-churn roster,

$$\widetilde{\text{Roster}}^t = \{i \in \text{Roster}^{t-1} : u_i^t > p_{\text{roster}}\}, \qquad u_i^t \overset{iid}{\sim} U[0,1],$$

then the broker samples without replacement from $\{1,\ldots,N\}\setminus \widetilde{\text{Roster}}^t$ until $|\text{Roster}^t| = R^*$, or until the population is exhausted. Standing-roster membership is therefore independent of current outsourcing decisions: outsourcing does not place an agent onto the standing roster, and being matched through the broker does not remove the agent from it.

**Current-client overlay.** In each period, the broker also maintains the one-period client set $D^t$ of agents who outsourced in that period. The broker's effective counterparty access set for period $t$ is therefore $A^t = \text{Roster}^t \cup D^t$. This restores an endogenous access channel, because current outsourcing expands the set of agents the broker can use as counterparties in that period without requiring a lagged client-memory mechanism.

**Broker edges in $G$.** Broker-node edges are synchronized to the standing roster, the current client set, and agents currently engaged in broker-channel matches. This means the broker is always adjacent in $G$ to its maintained access base and its current broker clients, while current broker-mediated relationships are also represented in the period graph even when the matched agents were not already on the standing roster. Because turnover removes exiting agents immediately but replenishment occurs at the next period start, the internal standing roster can temporarily fall below $R^*$ between the exit step and the next refresh.

**Availability.** A roster member is available as a counterparty in a given period if it has spare capacity ($|M_j^t| < K$) and is not already matched with the focal demander in that period. An agent may act as both a demander (seeking matches for its own relationship positions) and a counterparty (being matched with other demanders) in the same period, provided it still has open positions. Self-matches are excluded: the broker never matches an agent with itself.

### 8. Match Lifecycle

Matches are relationship positions within a period. Once a match is finalized in a within-period round, it occupies one distinct counterparty position for each side for the remainder of that period, both parties observe the realized match output immediately, and all positions reopen before the next period begins. The same unordered pair cannot form a second active relationship in the same period.

**At round finalization, for each accepted match:**
1. Realized output is drawn: $q_{ij} = Q + f(\mathbf{x}_i, \mathbf{x}_j) + \varepsilon_{ij}$.
2. Both parties add the observation to their histories: the demander adds $(\mathbf{x}_j, q_{ij})$ to $\mathcal{H}_{i}$; the counterparty adds $(\mathbf{x}_i, q_{ij})$ to $\mathcal{H}_{j}$.
3. If brokered, the broker adds $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$.
4. An edge is added between $i$ and $j$ in $G$ (if not already present).

These updates take effect immediately and therefore influence feasible capacities, known-neighbor sets, and learned partner means in later rounds of the same period.

**Before the next period begins:** clear the current-period match lists $M_i^t$ and $M_j^t$. Both sides regain the relationship position, so all $K$ positions are open again at the start of the next period.

### 9. Base Model Pseudocode

At the start of the simulation, the state of the world must be initialized.

> **INITIALIZE**
>
> *Agent types and matching function.*
> I.1. &emsp;Generate random frequencies $f_k$ and phases $\theta_k$ for the sinusoidal curve (§0).
> I.2. &emsp;Draw $N$ agent types: each at a random position $t_i \sim U[0,1]$ on the curve, perturbed by noise, and projected to the unit sphere.
> I.3. &emsp;Draw ideal type $\mathbf{c}$ (perturbation of a random curve position).
> I.4. &emsp;Draw SPD interaction matrix $\mathbf{A} = \mathbf{M}_A^\top \mathbf{M}_A \cdot (d / \text{tr}(\mathbf{M}_A^\top \mathbf{M}_A))$, where $\mathbf{M}_A \in \mathbb{R}^{d \times d}$ has iid $N(0,1)$ entries. Compute the empirical type second moment $\mathbf{S}_x = N^{-1} \sum_i \mathbf{x}_i \mathbf{x}_i^\top$. Then draw a symmetric Gaussian matrix $\mathbf{H}$, recenter it to zero trace, remove its weighted projection onto $\mathbf{A}$ under $\langle \mathbf{M}, \mathbf{N} \rangle_{\mathbf{S}_x} = \operatorname{tr}(\mathbf{S}_x \mathbf{M} \mathbf{S}_x \mathbf{N})$, and normalize the result to unit Frobenius norm: $\mathbf{B}_{\text{raw}} = \mathbf{H} - \frac{\operatorname{tr}(\mathbf{S}_x \mathbf{H} \mathbf{S}_x \mathbf{A})}{\operatorname{tr}(\mathbf{S}_x \mathbf{A} \mathbf{S}_x \mathbf{A})} \mathbf{A}$, $\mathbf{B} = \mathbf{B}_{\text{raw}} / \lVert \mathbf{B}_{\text{raw}} \rVert_F$. This makes the regime operator symmetric and weighted-orthogonal to $\mathbf{A}$ under the realized type distribution.
>
> *Calibration.*
> I.5. &emsp;Compute $\bar{q}_{\text{cal}} = E[q]$ from 10,000 random agent pairs $(i, j)$ with $i, j$ drawn independently and uniformly from $\{1, \ldots, N\}$ (self-pairs $i = j$ are not filtered; at $N = 1000$ the resulting bias is $O(1/N)$ and negligible). Set $r \leftarrow 0.60 \cdot \bar{q}_{\text{cal}}$.
> I.6. &emsp;Set channel costs from the shared search-cost rate $\lambda_c$: $\phi \leftarrow \lambda_c \cdot (\bar{q}_{\text{cal}} - r)$ and $c_s \leftarrow \lambda_c \cdot (\bar{q}_{\text{cal}} - r)$ (§11b).
>
> *Network.*
> I.7. &emsp;Build $G$: Watts–Strogatz with $N$ nodes, degree $k$, rewiring $p_{\text{rewire}}$. Node order is random (non-assortative initial network).
>
> *Broker.*
> I.8. &emsp;Seed broker roster with $R^* = \lceil 0.20 \cdot N \rceil$ randomly chosen agents (§7). Add broker-agent edges to $G$ for each roster member.
> I.9. &emsp;Seed broker history $\mathcal{H}_b$ with 100 observations drawn from random pairs of distinct roster members (sampling from the roster directly, not from pre-existing edges in $G$). For each sampled pair $(i, j)$, realize $q_{ij}$, append $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$, and add the edge $(i, j)$ to $G$ (the broker's seed placement creates the tie, mirroring the regular match flow in §4b).
>
> *State variables.*
> I.10. &emsp;For each agent $i$: seed $\mathcal{H}_{i}$ with 5 pairings sampled from $i$'s neighbors in $G$. For each sampled neighbor $j$, realize $q_{ij}$ and record $(\mathbf{x}_j, q_{ij})$ in $\mathcal{H}_i$ along with the corresponding `partner_mean` update for $j$. Seed observations are recorded only in the sampling agent's history (so the counterparty $j$ is not credited with this draw in $\mathcal{H}_j$); agents independently seed their own histories from their own neighborhoods. $M_i^0 \leftarrow \emptyset$.
> I.11. &emsp;Broker reputation from seed data: $\text{rep}^0 \leftarrow \text{mean}(\mathcal{H}_b)$. Agent satisfaction from seed data: $s_{i,\text{self}}^0 \leftarrow \text{mean}(\mathcal{H}_i)$; $s_{i,\text{broker}}^0 \leftarrow \text{rep}^0$.
>
> *Initial model training.*
> I.12. &emsp;For each agent $i$: train neural network on $\mathcal{H}_i$ for $E_{\text{init}}$ GD steps from random weights (§2a).
> I.13. &emsp;Train broker's neural network on $\mathcal{H}_b$ (symmetry-augmented) for $E_{\text{init}}$ GD steps from random weights (§2c). In the implementation, the symmetry-augmented examples are written into a preallocated broker-side buffer and training uses the active prefix of that buffer directly.

Each period proceeds through six steps (plus recording).

> **PERIOD $t$:**
>
> **0. CURRENT-PERIOD MATCH RESET**
> 0.1. &emsp;For each agent $i$: set $M_i^t \leftarrow \emptyset$, so all $K$ relationship positions are open at the start of period $t$.
> 0.2. &emsp;Clear the prior period's client overlay $D^{t-1}$. Refresh standing broker roster (§7): each current roster member exits independently with probability $p_{\text{roster}}$; then replenish uniformly without replacement from non-roster agents until the target size $R^* = \lceil 0.20 \cdot N \rceil$ is restored. Synchronize broker-agent edges in $G$ to the refreshed standing roster.
>
> **1. DEMAND GENERATION AND OUTSOURCING DECISIONS**
> 1.1. &emsp;For each agent $i$: draw demand count $d_i \sim \text{Binomial}(K,\; p_{\text{demand}})$.
> 1.2. &emsp;For each agent $i$ with $d_i > 0$:
> &emsp;&emsp;Compute $\text{score}_{\text{self}}, \text{score}_{\text{broker}}$ as in §6b.
> &emsp;&emsp;$\text{decision}_i \leftarrow \text{broker}$ if $\text{score}_{\text{broker}} > \text{score}_{\text{self}}$; else $\text{self}$. Ties broken uniformly at random. (Channel choice applies to all $d_i$ requested positions.)
> 1.3. &emsp;Form the current broker client set $D^t = \{i : \text{decision}_i = \text{broker}\}$. Synchronize broker-agent edges in $G$ so the broker is connected to the standing roster and all current clients. Output: for each demander, channel choice and demand count $d_i$. Broker client list $D^t$ with per-agent demand counts. Current standing roster $\text{Roster}^t$.
>
> **2. CANDIDATE EVALUATION**
>
> &emsp;**2.1. Fit prediction models:**
> 2.1.1. &emsp;For each agent $i$ whose parity matches the current period ($i \bmod 2 = t \bmod 2$): update neural network on $\mathcal{H}_{i}^t$ (§2b). Warm start; $E_t = \max(50, \lceil E_{\text{init}} \cdot n_{\text{new}} / n_i \rceil)$ GD steps on the sliding window of the most recent $W = 500$ observations. No regularization. Agents not selected in period $t$ keep accumulating $n_{\text{new}}$ observations and retrain the next period.
> 2.1.2. &emsp;Update broker's neural network on $\mathcal{H}_b^t$ with symmetry-augmented data (§2c). Same adaptive schedule and window. No regularization. In the implementation, the broker reuses a preallocated symmetry-augmented training buffer and trains on its active prefix directly.
>
> &emsp;**2.2. Round initialization:**
> 2.2.1. &emsp;Set remaining demander loads $u_i^0 \leftarrow d_i$ for all agents with demand.
> 2.2.2. &emsp;Set round index $\ell \leftarrow 1$.
>
> **3. WITHIN-PERIOD MATCHING ROUNDS**
>
> 3.1. &emsp;Form the current active demander set $U^\ell = \{i : u_i^{\ell-1} > 0 \text{ and } i \text{ still has spare capacity}\}$.
> 3.2. &emsp;If $U^\ell = \emptyset$: terminate within-period matching.
>
> &emsp;**3.3. Build round-specific candidate rankings**
>
> 3.3.1. &emsp;For each $i \in U^\ell$ with $\text{decision}_i = \text{self}$:
> &emsp;&emsp;Construct the current self-search candidate pool from:
> &emsp;&emsp;&emsp;known neighbors of $i$ in $G$ with spare capacity and stored partner means $\bar{q}_{ij}$, evaluated by those realized partner means;
> &emsp;&emsp;&emsp;up to $n_s$ sampled strangers with spare capacity, evaluated by $\hat{q}_i(\mathbf{x}_j)$.
> &emsp;&emsp;Drop any candidate whose demander-side evaluation is $\le r$, and order the remaining feasible candidates from best to worst.
>
> 3.3.2. &emsp;For each $i \in U^\ell$ with $\text{decision}_i = \text{broker}$:
> &emsp;&emsp;Form broker access set $A^t \leftarrow \text{Roster}^t \cup D^t$.
> &emsp;&emsp;Restrict to currently available accessible counterparties that are not already current-period counterparties of $i$.
> &emsp;&emsp;Compute broker predictions $\hat{q}_b([\mathbf{x}_i; \mathbf{x}_j])$ for all feasible $j \in A^t$, $j \neq i$.
> &emsp;&emsp;Drop any candidate with $\hat{q}_b([\mathbf{x}_i; \mathbf{x}_j]) \le r$, and order the remaining feasible candidates from best to worst.
>
> &emsp;**3.4. Within-round deferred acceptance**
>
> 3.4.1. &emsp;Each active demander proposes to its highest-ranked not-yet-rejected candidate.
> 3.4.2. &emsp;Each counterparty $j$ tentatively holds up to its current spare capacity's worth of incoming offers:
> &emsp;&emsp;Rank by the counterparty's own evaluation, $\bar{q}_{ji}$ for known partners and $\hat{q}_j(\mathbf{x}_i)$ for strangers; reject any offer with evaluation $\le r$.
> 3.4.3. &emsp;If a demander is rejected, it immediately proposes to its next-best feasible candidate in the same round.
> 3.4.4. &emsp;If an agent secures a tentative demander-side match while also holding incoming counterparty offers, release the lowest-ranked incoming hold as needed so that total tentative commitments do not exceed that agent's current spare capacity.
> 3.4.5. &emsp;Repeat 3.4.2-3.4.4 until no rejected demander has any feasible candidate left to try.
> 3.4.6. &emsp;Any demander that exhausts its ranked list without being held fails for the remainder of the period: it makes no further proposals in later rounds.
>
> &emsp;**3.5. Finalize accepted round matches**
>
> 3.5.1. &emsp;Finalize all tentatively held offers as accepted matches.
> 3.5.2. &emsp;For each accepted match $(i, j)$:
> &emsp;&emsp;Realize output: $q_{ij} = Q + f(\mathbf{x}_i, \mathbf{x}_j) + \varepsilon_{ij}$.
> &emsp;&emsp;Update histories immediately: add $(\mathbf{x}_j, q_{ij})$ to $\mathcal{H}_{i}$; add $(\mathbf{x}_i, q_{ij})$ to $\mathcal{H}_{j}$; if brokered, add $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$.
> &emsp;&emsp;Add match to current-period match lists: append $j$ to $M_i^t$; append $i$ to $M_j^t$.
> &emsp;&emsp;If standard (non-principal): add edge $(i, j)$ to $G$ if not already present.
> &emsp;&emsp;Record channel, realized output, predictions used, and whether $j$ was already a direct neighbor of $i$ before the round finalized.
> &emsp;&emsp;Set $u_i^\ell \leftarrow u_i^{\ell-1} - 1$.
> 3.5.3. &emsp;Synchronize broker-agent edges in $G$ so the broker remains connected to the standing roster, the current client set, and any agents currently engaged in broker-channel matches (§7).
> 3.5.4. &emsp;If the round produced zero accepted matches: terminate within-period matching.
> 3.5.5. &emsp;Else increment round index $\ell \leftarrow \ell + 1$ and return to 3.1.
>
> **4. LEARNING AND STATE UPDATES**
> 4.1. &emsp;Histories have already been updated during round finalization (Step 3.5.2), so no additional learning pass is needed here.
>
> 4.2. &emsp;Update satisfaction indices (§6a):
> &emsp;&emsp;For each agent $i$ with $d_i > 0$, let $c$ be $i$'s chosen channel:
> &emsp;&emsp;&emsp;Sum realized outputs over $i$'s accepted matches through $c$; unfilled requested positions contribute zero.
> &emsp;&emsp;&emsp;If $c = \text{self}$: compute $\tilde{q} = (\sum q_{ij} - c_s \cdot d_i) / d_i = \sum q_{ij}/d_i - c_s$.
> &emsp;&emsp;&emsp;If $c = \text{broker}$ in the base model: compute $\tilde{q} = \sum (q_{ij} - \phi) / d_i$ over accepted brokered relationships.
> &emsp;&emsp;&emsp;Update: $s_{i,c}^{t+1} = (1 - \omega)\, s_{i,c}^t + \omega \cdot \tilde{q}$.
>
> 4.3. &emsp;Update broker reputation (§6c):
> &emsp;&emsp;If $|D^t| > 0$: $\text{rep}^{t+1} \leftarrow \text{mean of } s_{i,\text{broker}}^{t+1} \text{ over } i \in D^t$.
> &emsp;&emsp;Else: $\text{rep}^{t+1} \leftarrow \text{rep}^{t}$ (hold previous value).
>
> **5. ENTRY AND EXIT**
> 5.1. &emsp;For each agent $i$:
> &emsp;&emsp;With probability $\eta$: agent exits.
> &emsp;&emsp;&emsp;Remove $i$ from $G$ (node and all edges).
> &emsp;&emsp;&emsp;Terminate all active matches involving $i$; counterparties regain capacity.
> &emsp;&emsp;&emsp;Remove $i$ from the broker's standing roster and from the current client set (if present). Broker-agent edges are then synchronized so the broker remains connected to the surviving standing roster, surviving current clients, and any surviving active broker-channel participants; replenishment of vacated standing-roster spots occurs at the next period start.
> &emsp;&emsp;&emsp;Replace with entrant $i'$: fresh type from curve + noise; empty histories; added to $G$ with $\lfloor k/2 \rfloor$ edges to type-similar agents ($\propto \exp(-\|\mathbf{x}_{i'} - \mathbf{x}_j\|^2)$). Self-satisfaction $\leftarrow$ mean of new neighbors' self-satisfaction; broker-satisfaction $\leftarrow$ current broker reputation (§6a).
>
> **6. RECORDING AND MEASUREMENT**
> 6.1. &emsp;Record period aggregates: match quality by channel; outsourcing rate (outsourced requested positions / total requested positions); mean self- and broker-satisfaction across agents; available-agent count.
> 6.2. &emsp;Record broker state: reputation $\text{rep}^t$; standing roster size; broker access size; $|\mathcal{H}_b^t|$.
> 6.3. &emsp;Compute per-agent averaged holdout prediction quality ($R^2$, bias, rank correlation) for broker and agents (§10), excluding fresh entrants with no match history. This runs every period because the cost is small (≈4,000 NN forward passes) and finer time resolution benefits the headline figures that track the informational gap over time.
> 6.4. &emsp;Every $M$ periods (default $M = 20$): compute network measures on $G$ (§10): betweenness centrality $C_B(b)$; Burt's constraint (broker's ego network); effective size (broker's ego network). The $M$-period cadence reflects the cost of Brandes BFS on the full graph, not a conceptual alignment with holdout measurement.

#### Parallelism summary

Steps 0 and 1 are embarrassingly parallel across agents. Step 2.1 (model fitting) remains the dominant parallel workload. Within Step 3, the self-search ranking stage is parallel across active demanders and the broker's round-level quality matrix is batch-computed across current broker clients, but the within-round deferred-acceptance pass is a conflict-resolution stage that is processed sequentially because tentative holds and rejections depend on shared capacities. Step 4 is lightweight because histories were already updated during round finalization; Step 5 still involves shared-state writes but on non-overlapping agent records. Network measures remain the most expensive standalone diagnostic computation; they read the full state but write nothing and can be offloaded to a separate thread or deferred to a coarser schedule.

### 10. Performance Measures

Computed on $G$ (which includes the broker as a permanent node; §4a) each measurement period. No agent uses these measures in its decisions; they are outputs for analysis.

#### Network measures

**Betweenness centrality.** Standard Freeman betweenness (Freeman, 1977) computed on $G$ using the Brandes (2001) algorithm adapted for single-node computation on undirected unweighted graphs. Neighbor iteration uses a compressed sparse row (CSR) adjacency structure built once per measurement call, with pre-allocated per-thread BFS workspaces for allocation-free parallel execution. The broker's betweenness is the fraction of all shortest paths that pass through the broker node, with the standard undirected normalization:

$$C_B(b) = \frac{1}{(n-1)(n-2)} \sum_{s \neq b} \sum_{t \neq s} \frac{\sigma_{st}(b)}{\sigma_{st}}$$

where $n = N+1$, $\sigma_{st}$ is the number of shortest paths from $s$ to $t$, and $\sigma_{st}(b)$ is the number passing through $b$. The double sum counts each undirected pair from both directions; dividing by $(n-1)(n-2)$ rather than $\binom{n}{2}$ corrects for this double-counting (Brandes, 2001, p. 9). As matches create direct ties, shortest paths increasingly bypass the broker, reducing betweenness: the structural erosion that the theory predicts.

**Burt's constraint.** Computed on the broker's ego network (Burt, 1992):

$$C_b = \sum_j \left(p_{bj} + \sum_{h \neq b,j}
p_{bh}\, p_{hj}\right)^2$$

where $p_{bj} = 1/d_b$ is the proportion of the broker's ties invested in node $j$ (for the unweighted network), and $p_{kj} = 1/d_k$ is the proportion of intermediary $k$'s ties invested in $j$ (Everett & Borgatti, 2020). Note that the indirect term uses the intermediary's degree, not the ego's. Low constraint = broker spans structural holes. High constraint = broker's contacts are interconnected.

**Effective size.** The number of non-redundant contacts in the broker's ego network. Using the Borgatti (1997) simplification for binary undirected networks: $\text{ES}_b = d_b - 2t_b / d_b$, where $d_b$ is the broker's degree and $t_b$ is the number of ties among the broker's neighbors (not counting ties to the broker). Equivalently: $\text{ES}_b = |N(b)| - \sum_j p_{bj} \sum_{h \neq b, h \in N(b)} m_{jh}$ where $p_{bj} = 1/d_b$ and $m_{jh} = 1$ if $j$ and $h$ are connected.

#### Prediction quality

**Winner's curse / selection bias.** Both agents and the broker select the counterparty with the highest *predicted* match quality from their candidate set ($\arg\max_j \hat{q}_{ij}$). When predictions are noisy, the selected counterparty's prediction $\hat{q}_{ij^*}$ is systematically inflated relative to the true match quality $f(\mathbf{x}_i, \mathbf{x}_{j^*})$, because the selection picks up positive noise realizations. This is the classic winner's curse.

**Holdout $R^2$ (model quality).** Each period, 100 agents are sampled at random (excluding fresh entrants with no match history). For each sampled agent $i$, 40 random partners $j$ are drawn, and both agent $i$'s neural network and the broker's neural network predict the noiseless true match quality $f(\mathbf{x}_i, \mathbf{x}_j)$ for each partner. Per-agent $R^2$, bias, and rank correlation are computed for each model, then averaged across the sampled agents. The implementation uses the standard $R^2 = 1 - \text{SSE}/\text{SST}$ definition, equivalently $1 - \text{MSE}/\operatorname{Var}_{\text{pop}}(q)$ with the population variance denominator. Because both models are evaluated on the same agent-partner sets, the resulting metrics are directly comparable: any gap reflects the models' relative quality, not differences in evaluation samples.

**Selected-sample metrics.** Three metrics are computed each period over all matches formed through each channel (self-search or brokered) that period:

- *Selected $R^2$* $= 1 - \text{SSE}/\text{SST} = 1 - \text{MSE}/\operatorname{Var}_{\text{pop}}(q)$. Because matched counterparties are those with the highest predictions, this sample is subject to the winner's curse: predictions are systematically inflated relative to outcomes, depressing $R^2$.

- *Bias* $= \frac{1}{n}\sum(\hat{q} - q)$. Tracks systematic over- or underprediction. Positive bias is expected in the selected sample due to the winner's curse.

- *Selected rank correlation* (Spearman's $\rho_S$). Measures whether the agent ranks matched counterparties correctly by realized output. The rank correlation is less affected by the winner's curse than $R^2$ because it is invariant to monotone transformations.

**Minimum variance threshold.** When $\text{Var}(q) < \sigma_\varepsilon^2 / 6 \approx 0.01$, the realized output variance in the sample is too small relative to the noise floor for $R^2$ to be informative. Below this threshold, all three metrics ($R^2$, bias, rank correlation) return NaN so that the row is treated uniformly as "insufficient signal" in downstream aggregation.

**Summary of prediction quality metrics:**

| Metric | What it measures | Selection bias? | Primary use |
|--------|-----------------|-----------------|-------------|
| Holdout $R^2$ | Model quality (approximation of $f$) | None (random sample, noiseless truth) | Informational advantage |
| Selected rank correlation | Matching decision quality (correct ordering) | Mild (order is more robust than level) | Allocation effectiveness |
| Selected $R^2$ | Prediction accuracy on actual matches | Strong (winner's curse) | Economic outcomes |

The broker-agent gap in holdout $R^2$ is the purest measure of the informational advantage. The gap in selected rank correlation shows whether the advantage translates into better matching decisions.

#### Other measures

**Access vs. assessment decomposition.** For each brokered match, record whether counterparty $j$ was a direct neighbor of demander $i$ in $G$ at the time of the match. If not: access value (the demander could not have found this counterparty through its own network). If yes: assessment value (the demander could have found this counterparty but the broker predicted match quality better).

**Match quality by channel.** Average realized match output $\bar{q}_c^t$ per period, where $c \in \{\text{self}, \text{brokered}\}$.

**Mean channel satisfaction.** Cross-agent means of the two satisfaction states, $N^{-1}\sum_i s_{i,\text{self}}^t$ and $N^{-1}\sum_i s_{i,\text{broker}}^t$. These summarize how the market's recent realized experience with each channel evolves over time.

**Outsourcing rate.** Fraction of requested relationship positions that are outsourced to the broker: outsourced requested positions / total requested positions. A demander-level outsourcing share (fraction of demanders choosing the broker channel) is retained as a secondary diagnostic in the code, but the requested-position share is the primary quantity because the model's demand object is the relationship position.

**Standing roster size.** Number of agents currently on the broker's standing roster (§7). In the recorded period outputs, this is measured after the start-of-period refresh and before Step 5 entry/exit, so it typically equals the target $R^*$. Internally, the roster can dip below target immediately after exits and is replenished at the next period start.

**Broker access size.** Number of distinct agents in the broker's within-period access set, $|A^t| = |\text{Roster}^t \cup D^t|$, where $D^t$ is the set of current-period broker clients. In the recorded period outputs, this is measured after current outsourcing decisions have formed $D^t$ and before Step 5 entry/exit. This is the meaningful quantity for how many agents the broker can search over in period $t$. Because some current clients can already be on the standing roster, broker access size is generally smaller than standing roster size plus the number of current broker clients.

**Available agents.** Number of agents with spare capacity at the time metrics are recorded, equivalently those with $|M_i^t| < K$. This is a capacity-based availability count, not merely a count of fully idle agents.

## Part II. Parameters, Calibration, and Initialization

### 11. Parameters

#### 11a. Parameter table

Parameters are organized into four categories reflecting their role in the analysis.

**Structural constants.** Define the model's mechanisms. Values are set by design rationale and not varied.

| Symbol | Meaning | Value | Notes |
|--------|---------|-------|-------|
| $d$ | Type dimensionality | 8 | Fixed |
| $k$ | Network mean degree | 6 | Watts-Strogatz ring lattice degree |
| $p_{\text{rewire}}$ | Network rewiring probability | 0.1 | Watts-Strogatz rewiring |
| $\omega$ | Satisfaction recency weight (§6a) | 0.2 | EWMA weight |
| $p_{\text{demand}}$ | Per-position demand probability | 0.50 | All $K$ relationship positions are open at period start; $d_i \sim \text{Binomial}(K, p_{\text{demand}})$ |
| $n_s$ | Max strangers in self-search | 5 | Sampled uniformly from non-neighbors with capacity |
| $\sigma_x$ | Type noise scale | 0.5 | Expected distance from agent to curve position |
| $\alpha_R$ | Target roster share (§7) | 0.20 | Standing roster target size is $R^* = \lceil \alpha_R N \rceil$ |

**Calibration parameters.** Set during model development. Constant in production runs.

| Symbol | Meaning | Default | Notes |
|--------|---------|---------|-------|
| $r$ | Outside option | $0.60 \cdot \bar{q}_{\text{cal}}$ | Constant for all agents; calibrated at initialization |
| $\eta_{lr}$ | Learning rate | 0.03 | Vanilla gradient descent, full-batch, no weight decay |
| $E_{\text{init}}$ | Initial training steps | 200 | Full convergence at initialization; in production periods each agent retrains every other period on a deterministic parity schedule, with steps $\max(50, \lceil E_{\text{init}} \cdot n_{\text{new}} / n_{\text{total}} \rceil)$ |
| $W$ | Training window | 500 | Train on at most $W$ most recent observations (sliding window) |
| $h_a$ | Agent hidden width | 16 | One hidden layer, ReLU activations |
| $h_b$ | Broker hidden width | 32 | One hidden layer, ReLU activations |
| $b_2^{(0)}$ | Initial output bias | $Q$ | Untrained networks predict population-mean quality rather than zero |
| $\sigma_\varepsilon$ | Match output noise SD | 0.10 | |
| $\delta$ | Regime gain strength (§1c) | 0.5 | $\delta = 0$: no regime effect; $\delta = 1$: maximum gain contrast |
| $\lambda_c$ | Shared search-cost rate | 0.15 | $\phi = \lambda_c\cdot(\bar{q}_{\text{cal}} - r)$, $c_s = \lambda_c\cdot(\bar{q}_{\text{cal}} - r)$; $c_s$ is a self-search cost per demanded relationship position, $\phi$ a successful standard-placement fee; §11b |
| $p_{\text{roster}}$ | Standing-roster churn probability (§7) | 0.02 | Each roster member is dropped independently at the start of a period, before uniform replenishment back to $R^*$ |

**Phase diagram axes.** Primary parameters of interest.

| Symbol | Meaning | Default | Sweep |
|--------|---------|---------|-------|
| $s$ | Active dimensions | 8 | {2, 4, 6, 8} |
| $\rho$ | Quality-interaction mixing weight | 0.50 | {0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0} |

**Model 1 parameters.** Apply only under resource capture (§12).

| Symbol | Meaning | Default | Notes |
|--------|---------|---------|-------|
| `enable_principal` | Resource capture toggle | false | When true, the broker can pre-capture whole currently available counterparty blocks before the residual round matching stage (§12c) |

**OAT sensitivity parameters.** Varied one at a time while holding all others at defaults.

| Symbol | Meaning | Default | Sweep | Notes |
|--------|---------|---------|-------|-------|
| $K$ | Distinct-counterparty capacity | 5 | {1, 2, 5, 10, 20, 50} | Exclusive at $K = 1$; multiple concurrent counterparties at $K > 1$ |
| $p_{\text{demand}}$ | Per-position demand probability | 0.50 | {0.10, 0.25, 0.50, 0.75, 0.90} | Higher values produce a thicker, faster-moving market |
| $\eta$ | Agent entry/exit rate | 0.02 | {0.01, 0.02, 0.05, 0.10} | |
| $\delta$ | Regime gain strength | 0.5 | {0, 0.25, 0.50, 0.75} | $\delta = 0$: no regime effect (pure statistical advantage) |

The activity parameters $p_{\text{demand}}$ and $K$ jointly determine the market regime. Because demand is per relationship position, the expected demand volume scales with $K \cdot p_{\text{demand}}$: high-capacity agents in high-demand environments generate more opportunities per period, reflecting a thicker, faster-moving market. Different combinations map to the illustrative domains:

| Domain | $p_{\text{demand}}$ | $K$ | Rationale |
|--------|---------------------|-----|-----------|
| Interdealer brokerage | High | 10–50 | Frequent opportunities; many concurrent positions |
| Collector networks | Moderate | 2–5 | Episodic transactions; moderate concurrency |
| Import-export trading | Low to moderate | 2–5 | Slower opportunity flow; moderate concurrency |

**Implementation parameters.** Control simulation scale.

| Symbol | Meaning | Default | Scale check |
|--------|---------|---------|-------------|
| $N$ | Agent population | 1000 | {500, 1000, 2000} |
| $T$ | Simulation length (periods) | 200 | {100, 200, 400} |
| $T_{\text{burn}}$ | Burn-in periods (discarded) | 30 | — |
| $M$ | Network measure interval | 20 | — |

#### 11b. Search-cost calibration

The two channel costs are calibrated jointly from the average match surplus scale $(\bar{q}_{\text{cal}} - r)$ using a shared search-cost rate $\lambda_c$:

$$
\phi = \lambda_c \cdot (\bar{q}_{\text{cal}} - r), \qquad
c_s = \lambda_c \cdot (\bar{q}_{\text{cal}} - r).
$$

In the default $\lambda_c = 0.15$, both channels use the same friction level. The two quantities are computed once at initialization and held constant thereafter, but they enter realized payoffs asymmetrically: $c_s$ is charged on each requested self-search relationship position regardless of fill, whereas $\phi$ is charged only on successful standard brokered placements.

#### 11c. Initial conditions

The initialization procedure is specified in the pseudocode (§9, steps I.1–I.13). The key design choices are:

- Agent types are drawn at random positions on the sinusoidal curve with noise, then projected to the unit sphere (§0).
- The matching function parameters ($\mathbf{c}$, $\mathbf{A}$, $\mathbf{B}$) are drawn once and held fixed (§1).
- Calibration quantities ($\bar{q}_{\text{cal}}$, $r$, $\phi$, $c_s$) are computed from 10,000 random agent pairs (§11b).
- Each agent's history is seeded with 5 pairings from its neighbors in $G$, ensuring initial predictions reflect the local network.
- The broker's roster is seeded at the fixed target size $R^* = \lceil 0.20 \cdot N \rceil$, and its history is seeded from 100 random roster member pairs.
- All neural networks are trained from random weights for $E_{\text{init}}$ steps on their seed histories before the first period (§2a). These seed histories initialize predictive capability, but under Model 1 they do **not** initialize principal-mode confidence: principal mode is disabled until the broker has observed live brokered outcomes in the simulation (§12c).

#### 11d. Reproducibility

All randomness flows from a single integer seed. The seed determines: type draws, the realization of $G$, matching function parameters ($\mathbf{c}$, $\mathbf{A}$, $\mathbf{B}$), broker seed roster, standing-roster churn and replenishment draws, and all subsequent random events. Simulations are fully reproducible given (parameter dictionary, seed).

## Part III. Model Variant: Resource Capture

All base model mechanisms (§§0–10) operate unchanged. The difference: the broker can additionally act as a **principal**, acquiring a counterparty's position or resource and presenting itself as the counterparty to the demander. Rather than connecting two agents, the broker takes one side of the match. This implements the resource capture mode of Proposition 3a.

### 12. Resource Capture

#### 12a. Setup

Under resource capture, the broker transitions from intermediary to principal. Instead of connecting a demander with a counterparty, the broker acquires the counterparty's position (paying the counterparty for its resource or service) and then matches directly with the demander. The demander deals with the broker, not with the original counterparty. The broker earns the spread between what it charges the demander and what it pays the counterparty, bearing inventory risk if the match output falls short.

Throughout Model 1, "demanders" means agents currently expressing **demand for matching** in the period. Depending on the empirical domain, these can stand in for buyers, sellers, producers, importers, dealers seeking an offsetting trade, or any other side currently seeking a match. The label refers to the direction of current matching demand, not to a fixed market role.

**State additions.** Matches gain a flag: *standard* (brokerage as in the base model) or *principal* (broker takes one side). The matching mechanism adds no new persistent agent-level behavioral state. For diagnostics, however, the implementation maintains two cumulative agent-level counters used only for broker-dependency measurement in §12i: total matches participated in, and principal-mode acquisitions as counterparty. On the broker side, Model 1 tracks a scalar confidence state $\kappa_b^t$, defined from realized **broker-controlled exposure** errors. The implementation also maintains period-local broker-owned inventory for acquired relationship positions during the current period only.

#### 12b. Mechanism

Model 1 moves the capture decision **before the round loop**. After current demand and outsourcing choices have been realized, the broker evaluates whether any currently available accessible counterparty block is worth buying outright for the current period. The broker then enters the round-by-round matching flow already owning those acquired relationship positions.

For any currently available accessible counterparty $j \in A^t$:

1. Let $c_j^t$ be $j$'s currently available block size at the moment of acquisition planning, that is, the number of distinct relationship positions it could still supply in the current period. At the start of period matching this is the full current block, net only of earlier acquired blocks in the same pre-period planning pass.
2. Let $\bar{q}_j$ be $j$'s acquisition reservation, equal to the mean of all outputs in $j$'s history, or $\bar{q}_{\text{cal}}$ if $j$ has no history.
3. For each currently outsourced demander $i$ with residual current demand in the pre-period planning state and no active current-period relationship with $j$, the broker computes the predicted pairing value $\hat{q}_b([\mathbf{x}_i; \mathbf{x}_j])$ and the position-level margin relative to standard placement,

$$
m_{ij}^t = \hat{q}_b([\mathbf{x}_i; \mathbf{x}_j]) - \bar{q}_j - \phi.
$$

4. The broker evaluates the **whole block** of $c_j^t$ positions by assigning those positions to the best distinct current outsourced demanders for $j$ in the pre-period planning state. A demander can justify at most one position in a candidate counterparty block, even if it has multiple residual requested positions, because a pair can hold only one current-period relationship. Partial capture is not allowed: either all $c_j^t$ currently available positions are taken or none are.
5. If the block is captured, the broker acquires all currently available positions from $j$ immediately. Those positions are removed from the open market for the rest of the period and become broker-owned same-period inventory. There is no cross-period inventory carry.

After acquisition, round-by-round execution proceeds using that owned inventory:

1. **The broker acquires $j$'s position** at reservation $\bar{q}_j$. Agent $j$'s acquired positions are consumed for the current period as broker-owned inventory, so those positions are no longer available to self-searchers or to residual standard broker matching. Agent $j$'s history and satisfaction are otherwise unaffected (satisfaction is updated only for the demander role, §6a); $j$ does not observe who the end-use counterparty is.
2. **During each matching round,** the broker may place at most one owned position with each active outsourced demander, using the broker's own ranking over the currently owned inventory. A demander already matched with the acquired counterparty in the current period cannot receive another position from that same counterparty.
3. **If an owned position is placed with demander $i$,** the broker matches with demander $i$ as the counterparty, stepping into $j$'s role. Match output is realized as $q_{ij} = Q + f(\mathbf{x}_i, \mathbf{x}_j) + \varepsilon_{ij}$, determined by the underlying pairing $(i, j)$ even though $i$ deals only with the broker.
4. **Both the demander and the broker experience $q_{ij}$.** The capture surplus of the placed position is $\Delta q_{ij} = q_{ij} - \bar{q}_j$. Negative realizations are the broker's inventory risk.
5. **If an acquired position is not placed by period end,** it expires with realized value 0. Its realized capture surplus is therefore $-\bar{q}_j$.
6. **Neither party observes the other's type.** On a placed principal position, the demander observes $q_{ij}$ but not $\mathbf{x}_j$; the counterparty does not observe $\mathbf{x}_i$ or $q_{ij}$. Neither party can update its prediction history.
7. **No edge is added to $G$ between $i$ and $j$.** The structural hole between them remains open.
8. **The broker adds $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$ only for placed principal positions.** The broker is the only agent that learns from principal-mode matches.

Counterparty acceptance is automatic by construction: in principal mode the counterparty does not evaluate the end-use pairing, because it has sold the currently available position to the broker at reservation $\bar{q}_j$. The broker absorbs capture risk by taking inventory exposure on the acquired block before round-by-round deployment is known.

As the market evolves and counterparties accumulate better match histories, their reservations $\bar{q}_j$ rise, naturally compressing capture margins. The broker therefore sustains positive capture only for counterparties whose self-assessed value remains below the broker's best current cross-demander uses of that counterparty.

#### 12c. Broker's decision: standard vs. principal

For each candidate counterparty block $j$, the broker forms the ordered set of current position margins $m_{ij}^t$ across currently outsourced demanders in the residual pre-period planning state, counting each demander at most once for that candidate block. Let

$$
V_j^t = \sum_{h=1}^{c_j^t} m_{j,(h)}^t,
$$

where $m_{j,(1)}^t \ge m_{j,(2)}^t \ge \cdots$ are the current position margins for counterparty $j$ sorted from best to worst across distinct demanders, and $c_j^t$ is the counterparty's currently available block size in that planning state. Thus $V_j^t$ is the net expected advantage, relative to standard placement fees, of taking **the entire currently available block** from $j$ based on its best current outsourced uses.

Let $\delta_j^t$ be the number of **distinct demanders** represented among the selected top-$c_j^t$ uses with strictly positive position margins. Model 1 requires current depth: the block is eligible for capture only if $\delta_j^t \ge 2$.

The broker captures block $j$ only if all of the following hold:

1. **Live confidence:** $\kappa_b^t$ is available.
2. **Full planned use:** there are at least $c_j^t$ current outsourced uses of $j$ in the residual pre-period planning state, so the whole block is worth taking on the current market-wide information.
3. **Current depth:** $\delta_j^t \ge 2$.
4. **Whole-block profitability:** the block clears the confidence hurdle

$$
V_j^t > c_j^t \kappa_b^t.
$$

The broker then chooses the currently available block with the largest positive excess

$$
\Xi_j^t = V_j^t - c_j^t \kappa_b^t,
$$

captures that entire block, updates residual current demand and residual current capacities in the **pre-period planning state**, and repeats the same calculation until no block has positive excess. The resulting inventory is then executed round by round during the period.

The objects in the rule are:

- $m_{ij}^t = \hat{q}_b([\mathbf{x}_i; \mathbf{x}_j]) - \bar{q}_j - \phi$: the broker's position-level expected gain from capture relative to standard placement.
- $\bar{q}_j$: agent $j$'s acquisition reservation, equal to the mean of all outputs in $j$'s history, or $\bar{q}_{\text{cal}}$ if $j$ has no history.
- $\phi$: the standard brokered-placement friction (§3c). Under standard placement the demander incurs $\phi$ and the broker has no output-level stake. Under principal mode no $\phi$ is charged to the demander; instead the broker bears capture risk and records realized capture surplus $\Delta q_{ij}$.
- $c_j^t$: counterparty $j$'s currently available block size in the pre-period planning state.
- $V_j^t$: the whole-block expected advantage of capturing $j$ now rather than leaving those positions for residual standard matching.
- $\delta_j^t$: the number of distinct current outsourced demanders among the selected positive-margin uses of $j$.
- $\kappa_b^t$: the broker's current confidence MAE, an exponentially weighted moving average of realized broker-controlled exposure errors.

Principal mode is therefore unavailable in period 1 by construction and becomes available as soon as live confidence has been initialized from observed brokered outcomes.

Formally, let $E_t$ be the set of all broker-controlled exposures in period $t$:

- each realized **standard** brokered match contributes one exposure with realized value equal to the realized match output and forecast equal to the broker's ex ante match prediction;
- each acquired **principal** position contributes one exposure with forecast equal to the broker's acquisition-time forecast, and realized value equal to the realized match output if the position is placed or 0 if it expires unplaced at period end.

The current-period broker exposure MAE is then

$$
\tilde{\kappa}_b^t =
\begin{cases}
\dfrac{1}{|E_t|}\sum_{e \in E_t} |y_e - \hat{y}_e| & \text{if } |E_t| > 0 \\
\text{undefined} & \text{if } |E_t| = 0,
\end{cases}
$$

and the broker updates its confidence state as

$$
\kappa_b^{t+1} =
\begin{cases}
\tilde{\kappa}_b^t & \text{if } \kappa_b^t \text{ is not yet available and } |E_t| > 0 \\
(1-\omega)\kappa_b^t + \omega \tilde{\kappa}_b^t & \text{if } \kappa_b^t \text{ is available and } |E_t| > 0 \\
\kappa_b^t & \text{if } \kappa_b^t \text{ is available and } |E_t| = 0,
\end{cases}
$$

If $\kappa_b^t$ is not yet available and $|E_t| = 0$, it remains unavailable and principal mode stays disabled. The broker uses $\kappa_b^t$ during period $t$ acquisition decisions and updates it only after period-$t$ outcomes are realized. This state variable depends only on information the broker plausibly observes about its own activity: its own ex-ante forecasts and realized outcomes on broker-controlled positions.

**Design motivation.** The decision now occurs before rounds because the substantive object of resource capture is no longer a single pair, but a currently available **counterparty block** whose value depends on the broker's ability to rank uses of that block across multiple outsourced demanders. This keeps the information content genuinely cross-market: an individual agent sees only its own realized matches, whereas the broker values the same counterparty by comparing it across the current outsourced demand in the market. The all-or-nothing rule preserves simplicity and makes capture lumpy rather than diffuse. The current-depth condition excludes degenerate cases where one demander alone drives capture. Removing the extra readiness gate keeps the mechanism lean: once the broker has live confidence, capture depends only on whether a whole current block is worth taking. Because unplaced acquired positions now enter $\kappa_b^t$ as zero-realization exposures, mistaken literal acquisitions feed back directly into future caution without adding another state variable.

Early in the simulation, principal mode is unavailable at first because live confidence does not yet exist. Once $\kappa_b^t$ has been initialized, principal capture can begin immediately if the broker sees a whole current block with enough depth and value to clear the rule above. As the broker's exposure errors fall and multiple current outsourced demanders value the same counterparties highly, more whole blocks clear the rule. Conversely, unsuccessful literal acquisition, including unplaced end-of-period inventory, raises $\kappa_b^t$ and makes further capture harder.

The capture dynamic relies on informational dependency and supply scarcity rather than long-duration lock-in (see §12e for the full feedback mechanism).

#### 12d. Principal-mode outcomes for each party

In principal mode the broker acts as the counterparty; no placement friction $\phi$ is charged, since the broker is no longer intermediating.

**Demander's perspective.** The demander experiences $q_{ij}$ with no friction deduction (no $\phi$ in principal mode). Its satisfaction input is therefore, all else equal, higher than on an otherwise identical standard brokered match (where $\phi$ is deducted), and higher than on self-search once self-search effort costs are accounted for. This makes principal-mode exposure attractive on the realized-payoff margin and reinforces the capture dynamic. The demander cannot update its prediction model from that principal-mode match, however: it observes $q_{ij}$ but lacks the counterparty type $\mathbf{x}_j$ needed for a history entry. Each principal-mode match therefore blocks learning on that observation. The demander can still learn from any standard or self-search matches realized through other relationship positions or in later periods.

**Broker's perspective.** The broker experiences $q_{ij}$ as the counterparty, against the counterparty's acquisition reservation $\bar{q}_j$. The **capture surplus** of a placed position is $\Delta q_{ij} = q_{ij} - \bar{q}_j$. If an acquired position expires unplaced at period end, its realized value is 0 and its realized surplus is therefore $-\bar{q}_j$. When the broker's predictions are accurate it selects high-output pairings with $q_{ij} \gg \bar{q}_j$ and captures positive surplus; when predictions are poor, or when inventory cannot be deployed, realized surplus may be negative. The broker adds $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$ only for placed principal positions and continues to learn from realized placements.

**Counterparty's perspective.** The counterparty's acquired position is consumed for the period as soon as it is sold to the broker. Because the counterparty does not observe the end-use match outcome or the demander's type (§12b.6), its history $\mathcal{H}_j$ is not updated by that principal-mode position; because satisfaction is updated only for the demander role (§6a), its satisfaction tracking is unaffected. The counterparty can still accumulate information through any standard matches it realizes elsewhere. Its reservation $\bar{q}_j$ therefore rises over time as it accumulates outputs from observable standard matches, narrowing the pool of pairings where capture surplus is likely positive.

#### 12e. Lock-in dynamics

Resource capture produces a **triple lock-in mechanism** at the level of principal-mode exposure:

**Informational lock-in.** On a principal-mode match, neither agent receives the typed observation needed to append a new history entry. Those matches therefore contribute no new agent-side training data. Agents can still refit their networks and continue learning from non-principal matches, so principal exposure slows or truncates learning rather than mechanically halting it unless principal mode fully displaces other match types.

**Structural lock-in.** No direct tie forms between $i$ and $j$ in $G$ on a principal-mode match. Standard and self-search matches still add ties, but principal-mode matches do not contribute to network densification. Structural holes therefore remain open along the captured pairings, and the broker's betweenness centrality does not decline from those matches.

**Supply-side lock-in.** Agents whose positions are repeatedly acquired by the broker are effectively removed from the open market during those periods and on those relationship positions. Self-searchers face a thinner candidate pool, degrading the quality of self-search outcomes and pushing more agents toward the broker. This supply scarcity reinforces the informational and structural effects above.

**Positive feedback loop.** Principal-mode matching removes some agent learning opportunities, prevents tie formation on those pairings, and thins the open market → exposed agents learn more slowly and self-search candidate pools worsen → self-search satisfaction falls relative to broker-channel satisfaction → more agents outsource → the broker observes more broker-controlled exposures, and if those exposures are used well $\kappa_b^t$ falls, increasing the set of counterparties for which there are multiple strong current outsourced uses → more whole blocks clear the decision rule in §12c → principal exposure expands further. Because capture is block-level and all-or-nothing, the transition can be visibly lumpy rather than smooth. Literal acquisition also adds a countervailing discipline: failed deployment raises $\kappa_b^t$.

This feedback loop is self-reinforcing once initiated, producing the abrupt capture trajectory predicted by Proposition 3a. The self-liquidating dynamic of structural advantage is attenuated, and can be largely suspended if principal mode becomes dominant, because principal-mode matches create no direct ties between agents.

#### 12f. Illustrative domains

Under resource capture, the broker transitions from connecting agents to taking one side of the match: acquiring a counterparty's position or resource and reselling it, with the broker bearing inventory risk. The mapping below shows how each v0.2 quantity corresponds to real-world analogs in the three illustrative domains.

**Interdealer brokerage (IDBs).** The broker transitions from voice intermediation to principal trading. Instead of finding a counterparty for a dealer's trade, the broker takes the other side itself, buying a position from one dealer and selling it to another. The broker warehouses the position and bears price risk until the offsetting trade clears. Neither dealer knows who is on the other side; both deal with the broker.
- $q_{ij}$ (match output): the joint value realized by the two-sided trade once both legs are placed.
- $\bar{q}_j$ (acquisition reservation): the counterparty dealer's ask price — the price level at which that dealer would normally clear the trade with a conventional counterparty.
- $\Delta q_{ij}$ (capture surplus): the realized bid-ask spread, net of position risk taken on during warehousing.
- Capture risk ($\Delta q_{ij} < 0$): the warehoused position moves against the broker before the offsetting leg is placed.
- $\phi$ (placement threshold): the traditional voice-broker commission the broker would otherwise collect as a pure intermediary.

**Collector networks.** The dealer transitions from pure intermediation to holding inventory. Instead of connecting a seller with a buyer, the dealer buys the piece outright (acquiring the seller's holding) and later sells it to a buyer. The dealer bears the risk that the piece may not find a suitable buyer at a profitable price. This is the standard transition from consignment dealer to gallery or wine merchant.
- $q_{ij}$ (match output): the realized buyer-seller joint value once the piece finds its end buyer.
- $\bar{q}_j$ (acquisition reservation): the seller's expected consignment price based on its own sale history.
- $\Delta q_{ij}$ (capture surplus): the dealer's margin — retail price minus the consignment reservation.
- Capture risk ($\Delta q_{ij} < 0$): the piece sells for less than the consignment reservation, or sits unsold and is eventually marked down.
- $\phi$ (placement threshold): the consignment dealer fee under pure intermediation.

**Import-export trading companies.** The trading company transitions from pure intermediation to taking principal positions. Instead of connecting a producer with a buyer, the company buys goods from the producer (acquiring the supply position) and resells to the buyer. The company bears inventory and price risk between acquisition and resale. This is the canonical trajectory of trading houses that evolve from brokers to merchants to vertically integrated conglomerates.
- $q_{ij}$ (match output): the joint value realized by producer-and-buyer once the shipment reaches the buyer.
- $\bar{q}_j$ (acquisition reservation): the producer's selling price based on its past export history.
- $\Delta q_{ij}$ (capture surplus): the trading margin — resale value minus procurement price.
- Capture risk ($\Delta q_{ij} < 0$): demand-side price drops, currency moves, or spoilage/delay reduce realized value below procurement cost.
- $\phi$ (placement threshold): the trading company's commission under pure intermediation.

#### 12g. Channel comparison

| | Base (no capture) | Resource capture (principal) | Data capture (§13a) |
|---|---|---|---|
| Who matches? | Agent matches directly | Broker takes one side | Agent matches directly |
| Agent observes counterparty type? | Yes | No (deals with broker) | Yes |
| Agent's prediction model improves? | Yes | No | Yes |
| Whose predictions guide selection? | Agent's own | Broker's | Broker's (sold to agent) |
| Direct tie forms? | Yes | No | Yes |
| Structural erosion? | Continues | Suspended | Continues |
| Supply scarcity? | No | Yes (broker acquires positions) | No |
| Broker bears capture risk? | No | Yes ($\Delta q_{ij}$ can be negative) | No |
| Quantity associated with placement | Successful standard-placement fee $\phi$ on demander; broker has no output-level stake | Capture surplus $\Delta q_{ij} = q_{ij} - \bar{q}_j$ accrues to broker's ledger of realized outcomes; no $\phi$ on demander | Friction $\mu$ on subscriber; broker has no output-level stake |
| Broker learns from match? | Yes | Yes | No (agent matched directly) |
| Predicted trajectory | Self-liquidating | Abrupt capture (Prop 3a) | Gradual capture (Prop 3b) |

The two capture modes differ on every dimension. Under resource capture, the broker becomes a principal, acquiring positions, bearing capture risk, and preventing clients from learning or forming direct ties. Under data capture, the broker *licenses* its informational advantage, selling predictions while clients continue matching directly, learning, and forming ties. These are two ways of monetizing the same informational asset: by exploiting it privately or by licensing it.

A subtle asymmetry in learning dynamics: under resource capture, the broker keeps learning (it observes both types in every principal-mode match). Under data capture, the broker's learning *slows* because agents match directly and the broker does not observe those outcomes unless a reporting mechanism exists. This creates a natural ceiling on data capture that resource capture does not face, and may contribute to the gradual-vs-abrupt distinction.

#### 12h. Pseudocode modifications

Steps not listed are identical to the base model pseudocode (§9).

<small>

> **3. WITHIN-PERIOD MATCHING ROUNDS** (principal-mode branch added)
>
> &emsp;**3.3. Pre-period principal acquisition plan** (new):
> 3.3.1. &emsp;If $\kappa_b^t$ is unavailable: skip principal mode for the period.
> 3.3.2. &emsp;Otherwise, compute the broker quality matrix $\hat{q}_b([\mathbf{x}_i; \mathbf{x}_j])$ over current outsourced demanders and the current access set $A^t$.
> 3.3.3. &emsp;For each currently available accessible counterparty block $j$:
> &emsp;&emsp;Compute acquisition reservation $\bar{q}_j$.
> &emsp;&emsp;Compute position margins $m_{ij}^t = \hat{q}_b([\mathbf{x}_i; \mathbf{x}_j]) - \bar{q}_j - \phi$ for all current outsourced demanders, counting each demander at most once for this candidate counterparty block.
> &emsp;&emsp;Take the top $c_j^t$ distinct current uses of $j$, compute block value $V_j^t$, and count distinct positive-margin demanders $\delta_j^t$.
> &emsp;&emsp;If the block cannot be fully justified on the residual pre-period state, or $\delta_j^t < 2$, or $V_j^t \le c_j^t \kappa_b^t$: mark the block as not acquirable.
> 3.3.4. &emsp;If at least one block is acquirable, choose the one with the largest positive excess $\Xi_j^t = V_j^t - c_j^t \kappa_b^t$.
> 3.3.5. &emsp;Acquire that entire block immediately, remove those positions from open-market availability for the period, update the residual pre-period state, and repeat 3.3.3-3.3.5 until no block is acquirable.
>
> &emsp;**3.4. Round execution of owned inventory**
> 3.4.1. &emsp;At the start of each round, before residual standard matching, consider the broker-owned inventory still unplaced.
> 3.4.2. &emsp;Among active outsourced demanders, assign at most one owned position to each demander in the round, using the broker's ranking over the currently owned inventory and excluding demander-counterparty pairs that already have a current-period relationship.
> 3.4.3. &emsp;Each placed owned position becomes one realized principal-mode match; each unplaced owned position remains broker inventory for later rounds in the same period only.
>
> &emsp;**3.5. Residual broker rankings and deferred acceptance**
> 3.5.1. &emsp;After the principal-inventory pass, rebuild the standard broker preference lists over the remaining broker demanders and the remaining currently available accessible counterparties.
> 3.5.2. &emsp;Run the base within-round deferred-acceptance procedure (§9 Step 3.4) on the residual current state. This residual round contains only standard offers.

> **4. OUTCOME REALIZATION AND LEARNING** (principal-mode branch added)
>
> 4.1. &emsp;For each accepted match $(i, j)$ realized in the current period, whether placed from owned inventory during Step 3.4 or finalized through the residual standard round:
> &emsp;&emsp;Realize output: $q_{ij} = Q + f(\mathbf{x}_i, \mathbf{x}_j) + \varepsilon_{ij}$.
> &emsp;&emsp;Increment $n_{\text{matches},i}$ and $n_{\text{matches},j}$ (cumulative match counters, any role, any channel, used for broker dependency $D_j$, §12i).
> &emsp;&emsp;**If standard** (self-search or residual standard brokered):
> &emsp;&emsp;&emsp;Add $(\mathbf{x}_j, q_{ij})$ to $\mathcal{H}_i$.
> &emsp;&emsp;&emsp;Add $(\mathbf{x}_i, q_{ij})$ to $\mathcal{H}_j$.
> &emsp;&emsp;&emsp;If brokered: add $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$.
> &emsp;&emsp;&emsp;Add edge $(i, j)$ to $G$.
> &emsp;&emsp;**If principal mode:**
> &emsp;&emsp;&emsp;Agent histories $\mathcal{H}_i$ and $\mathcal{H}_j$ are not updated.
> &emsp;&emsp;&emsp;Add $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$.
> &emsp;&emsp;&emsp;Do not add edge $(i, j)$ to $G$.
> &emsp;&emsp;&emsp;Increment $n_{\text{principal-acquired},j}$.
>
> 4.2. &emsp;Update satisfaction indices exactly as in the implemented broker-channel payoff rule:
> &emsp;&emsp;for each demander $i$ with broker channel, average $\sum_{\text{standard}} (q_{ij} - \phi) + \sum_{\text{principal}} q_{ij}$ over requested broker positions.
>
> 4.3. &emsp;Capture surplus recording:
> &emsp;&emsp;for each acquired principal position, record its realized value (equal to $q_{ij}$ if placed or 0 if unplaced), its acquisition reservation $\bar{q}_j$, and its acquisition-time forecast.
>
> 4.4. &emsp;Broker confidence update:
> &emsp;&emsp;Collect all broker-controlled exposures in period $t$, including standard broker matches and acquired principal positions that end either placed or unplaced.
> &emsp;&emsp;If at least one exists: compute $\tilde{\kappa}_b^t = \frac{1}{|E_t|}\sum_{e \in E_t}|y_e - \hat{y}_e|$.
> &emsp;&emsp;Update $\kappa_b^{t+1}$ by the EWMA rule in §12c.

</small>

#### 12i. Model 1 performance measures

**Principal-mode share** $P^t$: the fraction of brokered matches in period $t$ that are principal-mode (versus standard placement). This is the primary capture metric. Proposition 3a predicts an abrupt tipping point: $P^t$ should remain near zero while the broker is still operating without live confidence, then jump sharply once whole counterparty blocks begin clearing the rule in §12c. Under the current timing, $P^1 = 0$ by construction.

**Broker confidence state.**

- **Broker selected-match MAE** $= \frac{1}{|B_t|}\sum_{m \in B_t}|q_m - \hat{q}_{b,m}|$ over all realized broker matches in period $t$ (standard and principal placements only). This remains a realized-match prediction diagnostic.
- **Broker confidence MAE** $\kappa_b^t$: the broker's state variable carried into period $t$ and used in principal-mode decisions. Before the first live broker period with broker-controlled exposures, this quantity is unavailable. A declining $\kappa_b^t$ indicates that the broker's recent acquisition and brokerage forecasts have become more reliable, lowering the effective caution term in §12c.

**Agent prediction quality by principal-mode exposure.** Average holdout $R^2$ stratified by agents' cumulative principal-mode match fraction. Agents with high exposure should show stagnating prediction quality (informational lock-in), while agents who primarily self-search or receive standard placements should continue improving.

**Capture outcome.** Over acquired principal positions in period $t$, whether placed or unplaced:

- **Mean capture surplus** $\overline{\Delta q}^t = \frac{1}{|C_t|} \sum_{e \in C_t} \Delta q_e$: the typical per-position capture surplus, where $\Delta q_e = y_e - \bar{q}_e$ and $y_e = 0$ for unplaced acquired inventory.
- **Capture loss rate** $= |\{e \in C_t : \Delta q_e < 0\}| / |C_t|$: the share of acquired principal positions whose realized value falls below the counterparty's acquisition reservation.
- **Capture loss magnitude** $= \mathrm{mean}(|\Delta q_e| \mid \Delta q_e < 0)$: the typical size of losses when they occur.

Early in the capture transition, when the broker's predictions are less accurate, losses should be more frequent and larger. As predictions improve, both loss rate and loss magnitude should decline. Persistent losses would indicate that the broker is going principal too early.

**Capture decision quality.** On the principal-mode subset in each period, we measure whether the broker's ex-ante basis $\hat{q}_b - \bar{q}_j$ correlates with the realized surplus $q_{ij} - \bar{q}_j$. If the subset has fewer than 5 matches, both metrics return NaN.

- **Capture decision rank correlation** $= $ Spearman $\rho$ between $\hat{q}_b - \bar{q}_j$ and $q_{ij} - \bar{q}_j$. A high value means the broker reliably ranks matches by realized surplus; a low value means it goes principal somewhat blindly.
- **Capture decision RMSE** $= \sqrt{\mathrm{mean}((\hat{q}_b - q_{ij})^2)}$ on principal-mode matches. Calibration of the broker's output prediction on captured pairings.

**Supply scarcity.** The per-period fraction of agents whose positions are acquired by the broker in principal mode, and the resulting impact on self-search candidate pool sizes. Under capture, self-searchers face a shrinking pool of available counterparties.

**Broker dependency.** For each agent $j$ with at least one match, the cumulative dependency ratio
$$D_j^t = \frac{n_{\text{principal-acquired},j}^t}{n_{\text{matches},j}^t},$$
where $n_{\text{principal-acquired},j}^t$ is the cumulative count of principal-mode matches in which $j$ was the acquired counterparty and $n_{\text{matches},j}^t$ is the cumulative count of all matches $j$ has participated in (any role, any channel). At $D_j = 0$, the agent participates only in direct or standard-brokered matches; at $D_j = 1$, every match it has ever been in has been a principal-mode acquisition from it. The distribution of $\{D_j^t\}$ across eligible agents characterizes the extent and concentration of monopolization. Summary statistics:

- **Mean $D_j$**: overall dependency level.
- **$D_j$ 90th percentile**: the heavy-use tail.
- **Fraction of agents with $D_j > 0.5$**: effectively monopolized agents.
- **Gini coefficient of $\{D_j\}$**: concentration of capture across agents.

---

## Part IV. Outstanding Design Choices

### 13. Deferred Design Choices

The following design choices are deferred for future work. They are described at a conceptual level to guide subsequent development.

#### 13a. Data Capture (Model 2)

Under data capture (Proposition 3b), the broker sells access to its prediction model as a per-period subscription service. Subscribing agents use the broker's model when evaluating strangers during self-search (§5a), while continuing to match directly, learn from outcomes, and form ties. For known neighbors, subscribers still use their own historical averages. The broker earns per-period subscription revenue $\mu$ rather than per-match fees.

Data capture produces the gradual trajectory of Proposition 3b: agents keep learning and forming ties, structural erosion continues, and the broker's advantage narrows as subscribers improve their own predictions. The channel comparison table in §12g summarizes the contrast with resource capture.

**Open design questions:**

**Does the broker observe outcomes of subscriber-directed matches?** If not, the broker's learning slows under data capture: subscribers use the broker's model to find better matches with strangers, but the broker doesn't see the outcomes. This creates a natural ceiling on the broker's model quality. If the broker does observe outcomes (e.g., through a reporting requirement in the subscription contract), the ceiling disappears and data capture dynamics change.

**Does subscription replace or supplement the agent's own model for stranger evaluation?** If the subscription replaces the agent's neural network entirely (the agent uses the broker's predictions for all strangers), the agent becomes dependent and its own model atrophies. If the subscription supplements (e.g., the agent uses the better of its own prediction and the broker's for each stranger), the agent's model continues to improve alongside the broker's. The replacement version is simpler and produces stronger capture dynamics; the supplement version is more realistic.

**Can subscribers also use the broker for standard placement simultaneously?** If yes, the broker can earn revenue from both subscription fees and placement fees, and subscribers benefit from both better predictions and access to the broker's roster. If no, subscription and brokerage are mutually exclusive channels.

#### 13b. Alternative Resource Capture Mechanisms

Two alternatives to the principal-mode mechanism (§12) are noted for future exploration.

**Exclusive contracts without information lock-in.**

Agents sign exclusive contracts with the broker for $L$ periods but still observe counterparty types and form direct ties. Lock-in is contractual, not informational: the agent's prediction model and network continue to improve, but it cannot switch to self-search during the contract.

This variant serves as a **comparison case**: if capture is weaker without information lock-in, it demonstrates that the informational channel (not the contractual restriction) drives the abrupt capture dynamics of Proposition 3a.

**Partial obfuscation.**

The broker reveals noisy or partial type information: $\tilde{\mathbf{x}}_j = \mathbf{x}_j + \boldsymbol{\zeta}$, where $\boldsymbol{\zeta} \sim N(0, \sigma_{\text{obf}}^2 \mathbf{I}_d)$. The agent can update its history with $(\tilde{\mathbf{x}}_j, q_{ij})$, but the noisy type degrades the quality of its learned model. At $\sigma_{\text{obf}} = 0$, this is standard brokerage; as $\sigma_{\text{obf}} \to \infty$, the agent's observation becomes progressively less informative, approaching the per-match informational opacity of principal mode.

This variant provides a continuous lock-in parameter that can be swept, serving as a **robustness check**: does the capture result survive when informational obstruction is weaker or noisier than in the principal-mode benchmark? The theoretical prediction is that capture requires sufficiently strong obfuscation but is not knife-edge: there should be a threshold $\sigma_{\text{obf}}^*$ above which capture dynamics emerge.

#### 13c. Prediction Confidence and Uncertainty

The current model does not track per-prediction posterior uncertainty. All match predictions are still point estimates. It does, however, track a reduced-form broker confidence state, $\kappa_b^t$, defined from live realized broker-controlled exposure errors and used as a scalar caution term in the principal-mode decision (§12c). This captures recent realized forecasting reliability without requiring a full predictive distribution.

**Bayesian last layer.** A natural extension of the current neural network architecture (§2a): the hidden layer remains a deterministic feature extractor trained by gradient descent, but the output layer is replaced with Bayesian linear regression. Given hidden features $\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1)$ from the training data, the posterior over output weights $\mathbf{w}_2$ is available in closed form (conjugate Gaussian). For a new input $\mathbf{z}^*$, the predictive distribution is $N(\boldsymbol{\mu}_{\text{post}}^\top \mathbf{h}^*, \; \sigma_\varepsilon^2 + \mathbf{h}^{*\top} \boldsymbol{\Sigma}_{\text{post}} \mathbf{h}^*)$, where the second variance term $\mathbf{h}^{*\top} \boldsymbol{\Sigma}_{\text{post}} \mathbf{h}^*$ is the *epistemic* uncertainty (large when the input is far from training data in feature space, small when it is well-covered). Implementation cost is minimal: one $h \times h$ matrix inversion per agent per period (at $h = 16$, this is trivial).

**Uses of per-prediction uncertainty:**
- *Match selection.* An upper confidence bound (UCB) rule (select the partner with the highest $\hat{q} + \kappa \cdot \hat{\sigma}$) would balance exploitation (high predicted quality) with exploration (high uncertainty), generating more informative data and accelerating learning.
- *Principal-mode decision.* The broker could replace the current global MAE-based caution term with pair-specific prediction uncertainty, avoiding principal positions where $q_{ij}$ is highly uncertain and inventory risk is greatest.
- *Outsourcing decision.* An agent whose average predictive uncertainty is high might rationally prefer the broker even when satisfaction scores are comparable.
- *Measuring the informational advantage.* The epistemic uncertainty gap between agent and broker (the broker's $\Sigma_{\text{post}}$ is smaller because it has more diverse training data) directly quantifies the informational advantage at the prediction level.

Deferred because the current reduced-form MAE-based confidence state is sufficient to demonstrate the core propositions while keeping the model parsimonious. A Bayesian last layer would enrich the dynamics by replacing the global caution term with pair-specific epistemic uncertainty, and could be added without changing the hidden-layer training procedure.

#### 13d. Pricing Alternatives

The base model uses a fixed successful-placement fee $\phi$ on standard brokered matches. Under Model 1 (principal mode), the broker's compensation is the spread $q_{ij} - \bar{q}_j$ with no additional fee. Two alternative pricing mechanisms are noted for future exploration.

**Surplus-proportional fee.** $\phi = \alpha \cdot \hat{q}_b([\mathbf{x}_i; \mathbf{x}_j])$. The broker charges a fraction of its predicted match quality. This creates a recognition gap: the broker's revenue depends on its own prediction, while the agent's satisfaction depends on realized quality. Better predictions increase broker revenue, strengthening the incentive to invest in prediction accuracy.

**Prediction-based fee.** $\phi = \alpha \cdot (\hat{q}_b - \hat{q}_i)$. The broker charges for the prediction improvement it provides over the agent's own model. This directly prices the informational advantage but requires the broker to know (or estimate) the agent's prediction quality.

Both alternatives create richer dynamics but add parameters and complicate the satisfaction comparison between channels. The fixed-fee design isolates the informational channel by removing price as a margin of competition.

#### 13e. Other Design Choices

**Cross-period inventory under before-pairing capture.**

The current Model 1 already uses a before-round acquisition decision, but it is restricted to **same-period-only** capture: the broker can carry acquired inventory across rounds within the current period, but any unplaced inventory expires at period end. A future variant could allow the broker to carry unassigned inventory across periods. That extension would require explicit holding-cost, depreciation, expiry, or liquidation rules for unplaced inventory and would materially strengthen the broker's warehousing role.

**Fixed acquisition price at outside option.** The current model sets the counterparty's ask price at $\bar{q}_j$ (average realized match quality from its history). A simpler alternative: the counterparty always accepts at the fixed outside option $r$, regardless of its match history. This makes acceptance truly automatic and the acquisition cost constant across counterparties, producing a clean regime shift in the broker's principal-mode decision. The tradeoff is that the broker acquires all positions at the same low price, which may make capture too easy and remove the natural margin compression from rising counterparty experience.

**Exclusivity under principal mode.** The base Model 1 uses relationship-position capacity accounting: each acquired or placed principal position consumes one of an agent's $K$ distinct-counterparty positions, and any remaining positions can still be used elsewhere in the period. An alternative is full exclusivity ($\xi = 1$): an agent with any principal-mode exposure cannot self-search at all during that period, routing all demand through the broker. This produces total information freeze (the agent gains no new observations from any source) and stronger lock-in. With relationship-position demand, an agent at $K = 5$ generates about 2.5 requested positions per period; under current position accounting, some of these could still go through self-search even if one position is affected by principal mode. Full exclusivity would block all self-search, significantly strengthening the lock-in. Comparing dynamics under relationship-position accounting and full exclusivity would test whether the full information freeze is necessary for the abrupt capture trajectory of Proposition 3a.

#### 13f. Counterparty-Capacity Interpretation of $K$

The current specification interprets $K$ as the maximum number of **concurrent counterparties** or active bilateral relationships an agent can maintain within a period. Each accepted match consumes one relationship position for the demander and one for the counterparty for the remainder of the period. Repeated matches with the same partner in the same period are not allowed.

Under this interpretation, an accepted match no longer represents a single transaction-level placement, but the formation of a period-level commercial relationship. Repeated transactions between the same two parties within the period are bundled into that relationship rather than modeled individually. This interpretation is closer to settings such as labor-market intermediation, where a match is naturally read as a filled bilateral position rather than a sequence of repeated trades.

This reinterpretation has several conceptual advantages. The network edge created by an accepted match aligns more naturally with the object being modeled, since one accepted match corresponds to one active relationship. The block-capture rule in principal mode is also easier to read, because acquiring one of an agent's $K$ positions means taking one of a limited number of concurrent relationship positions. Likewise, supply scarcity under resource capture becomes more relationship-based and easier to interpret.

The tradeoff is that the model has changed meaning, not just notation. Match output $q_{ij}$, search costs, broker fees, satisfaction updates, and capture surplus are now interpreted at the relationship-position level rather than the transaction level. Principal-mode acquisition is a stronger form of scarcity than a transaction-slot acquisition, because occupying one of an agent's $K$ counterparty positions blocks an entire relationship for the period. Repeated within-period trade volume between the same pair is no longer observed directly.

## Figures

**Fig. 1.** The informational mechanism.
- *Purpose:* Establishes the core mechanism: the broker learns faster than individual agents, the gap widens with matching complexity, and this drives increasing outsourcing (Propositions 1.1, 1.2, 1.3).
- *Content:* All panels at default parameters ($s = 8$, $\rho = 0.50$). Each panel includes a **base model** series (dashed grey) as a no-capture reference line, plus Model 1 series.
  - Panel A: time on the horizontal axis, prediction quality (holdout $R^2$) on the vertical axis. One line for the broker, one for the average agent. The broker-agent gap reflects the informational advantage and its dynamics over time. An inset shows the effect of varying $s$.
  - Panel B: time on the horizontal axis, outsourcing rate on the vertical axis. The base model establishes the reference trajectory. Model 1 diverges.
  - Panel C: time on the horizontal axis, average realized match output by channel (self-search, standard brokered, principal mode).

**Fig. 2.** Decoupling of structural position from informational advantage.
- *Purpose:* The central empirical implication. Shows that betweenness centrality declines while the broker's informational advantage grows, and that resource capture suspends the structural erosion (Propositions 2.1, 3a).
- *Content:* Time on the horizontal axis, dual vertical axes for broker betweenness centrality and broker prediction quality. Under Model 1, betweenness plateaus or recovers once principal mode dominates.

**Fig. 3.** Access vs. assessment decomposition over time.
- *Purpose:* Traces the shift from network access to information assessment as the dominant source of broker value (Propositions 1.3a, 1.3b).
- *Content:* Time on the horizontal axis, fraction of brokered matches on the vertical axis, decomposed into access value (counterparty was not in demander's network) and assessment value (counterparty was reachable but broker predicted better).

**Fig. 4.** Capture dynamics and the lock-in mechanism.
- *Purpose:* Shows that capture occurs and the lock-in mechanism explains why resource capture is abrupt and self-reinforcing (Proposition 3a).
- *Content:*
  - Panel A: time on the horizontal axis, principal-mode share $P^t$ on the vertical axis. Shows the abrupt tipping point as the broker shifts from standard placement to principal mode.
  - Panel B: time on the horizontal axis, average agent prediction quality on the vertical axis, stratified by principal-mode exposure (high vs. low). Broker-dependent agents stagnate; others continue improving. Panel A shows the outcome; Panel B shows the mechanism.

**Fig. 5.** Phase diagram.
- *Purpose:* Maps the conditions under which capture occurs, identifying regions of no capture, partial capture, and full capture as a function of matching complexity (Proposition 2.2).
- *Content:* Main axes TBD. Heatmap or contour plot showing the broker-agent prediction quality gap (or principal-mode share at steady state) across the parameter space.

#### SI figures

**Fig. S1.** Prediction quality decomposition.
- *Content:* Three sub-panels: $R^2$, bias, and rank correlation over time (broker and average agent). Under Model 1, agent lines stratified by principal-mode exposure.

**Fig. S2.** Attributional vs. relational channel (Proposition 1.2).
- *Content:* $\rho$ on horizontal axis; broker-agent gap in holdout $R^2$; outsourcing rate at steady state.

**Fig. S3.** OAT parameter sweeps.
- *Content:* Grid of panels varying $\eta$, $\delta$, $p_{\text{demand}}$, $K$ while holding others at defaults.

**Fig. S4.** Network visualization snapshots.
- *Content:* The network $G$ at early, middle, and late periods. Broker node positioned centrally. Under Model 1, late-period graph should show persistent structural holes between agents matched through the broker's principal mode.

**Fig. S5.** Broker risk profile.
- *Purpose:* Shows the frequency and magnitude of inventory losses the broker absorbs in principal mode.
- *Content:* Time on the horizontal axis, distribution of $q_{ij} - r$ for principal-mode matches. Early: wider distribution with more losses. Late: concentrated in positive territory as predictions improve.

**Fig. S6.** Network degree diagnostics in the base-model exploration.
- *Purpose:* Tracks how the overall connectivity of $G$ evolves as the market densifies, complementing the broker-centered structural-hole measures.
- *Content:* Four panels with time on the horizontal axis and agent-network degree statistics on the vertical axis: mean degree, median degree, minimum degree, and maximum degree (computed over agent nodes only, excluding the broker).

## References

Bethune, Z., Sultanum, B., & Trachter, N. (2024). An information-based theory of financial intermediation. *Review of Economic Studies*, *91*(3), 1424–1454.

Brandes, U. (2001). A faster algorithm for betweenness centrality. *Journal of Mathematical Sociology*, *25*(2), 163–177.

Borgatti, S. P. (1997). Structural holes: Unpacking Burt's redundancy measures. *Connections*, *20*(1), 35–38.

Brenner, T. (2006). Agent learning representation: Advice on modelling economic learning. In K. Judd & L. Tesfatsion (Eds.), *Handbook of computational economics* (Vol. 2, pp. 895–947). North-Holland.

Burt, R. S. (1992). *Structural holes: The social structure of competition*. Harvard University Press.

Burt, R. S. (2005). *Brokerage and closure: An introduction to social capital*. Oxford University Press.

Duffie, D., Gârleanu, N., & Pedersen, L. H. (2005). Over-the-counter markets. *Econometrica*, *73*(6), 1815–1847.

Everett, M. G., & Borgatti, S. P. (2020). Unpacking Burt's constraint measure. *Social Networks*, *62*, 50–57.

Freeman, L. C. (1977). A set of measures of centrality based on betweenness. *Sociometry*, *40*(1), 35–41.

Li, D. D. (1998). Middlemen and private information. *Journal of Monetary Economics*, *42*(1), 131–159.

Muscillo, A. (2021). A note on matricial ways to compute Burt's structural holes in networks. *arXiv preprint arXiv:2102.05114*.

Rogerson, R., Shimer, R., & Wright, R. (2005). Search-theoretic models of the labor market: A survey. *Journal of Economic Literature*, *43*(4), 959–988.

Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, *393*(6684), 440–442.
