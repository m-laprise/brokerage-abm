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

High matching complexity makes the principals' learning problem harder, widening and preserving the broker's advantage.

In markets with simple matching problems, principals learn fast and well enough that the broker does not accumulate a decisive informational advantage. Brokers persist as commodity intermediaries earning thin margins and may attempt capture but do not consolidate into dominant principals. This is the no-capture region of the parameter space.

Within the capture region of the parameter space, if the broker does not capture, its information advantage may start to erode over time.

### 3. Forms of capture

#### Proposition 3

**Capture can occur in two forms with qualitatively different dynamics.**

**3a. Under resource capture, the transition is abrupt, and the broker suddenly starts taking inventory risk and acting primarily as a principal.** Resource capture creates a triple lock-in: the client's information state freezes (it doesn't learn from new matches like it did when the broker acted as an intermediary), the client's network no longer grows (the broker is everyone's counterparty, so no direct ties form between principals), and the open market thins as the broker acquires client-origin positions that would otherwise have generated direct matches. The self-liquidating dynamic of structural advantage is suspended, because the broker no longer creates direct ties between clients. This produces a steep capture trajectory.

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
- **Current-period matches** $M_i^t$: the list of active bilateral relationships involving $i$ that have already formed in period $t$. Each counterparty can appear at most once in $M_i^t$ during a period. There is no per-period cap on the number of counterparties an agent can receive through incoming accepted offers.
- **Active demand bound** $K$: the upper bound on the number of active demands an agent can initiate in a period (default $K = 5$). It governs outgoing search intensity, not the number of counterparties the agent can have.
- **Experience history** $\mathcal{H}_{i}^t = \{(\mathbf{x}_j, q_{ij})\}$: the set of (other party's type, realized match output) pairs from all matches $i$ has participated in, regardless of whether $i$ was the demander or the counterparty (§2a). Because the matching function is symmetric (§1a), both roles produce the same prediction target.
- **Satisfaction indices** $s_{i,c}^t$: one scalar per search channel $c \in \{\text{self}, \text{broker}\}$, tracking realized match value via an EWMA (§6a). Drives the outsourcing decision (§6).

Agents exit independently each period with probability $\eta_{\mathrm{exit}}$ (default 0.02) and are replaced by new entrants with fresh types, empty histories, self-satisfaction initialized from neighbors' opinions, and broker-satisfaction set to the current broker reputation (§6a).

#### Agent types

Agents are described by type vectors in $\mathbb{R}^d$ ($d = 8$). These types are the observable characteristics that determine productive compatibility through the matching function (§1).

Agent types lie near a smooth one-dimensional curve on the surface of the unit sphere in $\mathbb{R}^d$. The curve is parameterized by a position $u \in [0, 1]$:

$$\mathbf{x}(u) = \frac{\tilde{\mathbf{x}}(u)}{\|\tilde{\mathbf{x}}(u)\|}, \qquad \tilde{x}_\ell(u) = \begin{cases} \sin(2\pi \nu_\ell u + \psi_\ell) & \ell = 1, \ldots, d_\gamma \\ 0 & \ell = d_\gamma+1, \ldots, d \end{cases}$$

where $\nu_\ell \sim \mathrm{Unif}\{1, 2, 3, 4, 5\}$ are random integer frequencies and $\psi_\ell \sim \mathrm{Unif}[0, 2\pi)$ are random phases, both drawn once per simulation, and $d_\gamma \leq d$ is the number of **active dimensions** (the dimensions along which the curve has nonzero variation). The remaining $d - d_\gamma$ dimensions receive only noise (see below).

Each agent is drawn at a random position $u_i \sim \mathrm{Unif}[0,1]$ on the curve, then perturbed:

$$\mathbf{x}_i = \frac{\mathbf{x}(u_i) + \boldsymbol{\epsilon}_i}{\|\mathbf{x}(u_i) + \boldsymbol{\epsilon}_i\|}, \qquad \boldsymbol{\epsilon}_i \sim \mathcal{N}\!\left(\mathbf{0}, \frac{\sigma_x^2}{d} \mathbf{I}_d\right)$$

The noise $\boldsymbol{\epsilon}_i$ is applied in all $d$ dimensions (including inactive ones), so that type vectors are not exactly confined to the $d_\gamma$-dimensional subspace of the curve. The per-dimension noise scale $\sigma_x / \sqrt{d}$ is chosen so that the expected Euclidean distance from an agent to its curve position is approximately $\sigma_x$ regardless of $d$. The result is then re-projected to the unit sphere.

The parameter $d_\gamma$ controls the complexity of the matching problem. When $d_\gamma = d$, the curve spans all $d$ dimensions: agents nearby on the curve have similar types, while agents far apart point in genuinely different directions across all of $\mathbb{R}^d$. When $d_\gamma < d$, the curve is confined to a lower-dimensional subspace.

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
\varepsilon_{ij} \sim \mathcal{N}(0, \sigma_\varepsilon^2)$$

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

**Base interaction.** The bilinear form $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j$ measures the complementarity of the pairing. The interaction matrix $\mathbf{A} \in \mathbb{R}^{d \times d}$ is symmetric positive definite (SPD), constructed as $\mathbf{A} = \mathbf{M}_A^\top \mathbf{M}_A \cdot (d / \text{tr}(\mathbf{M}_A^\top \mathbf{M}_A))$ where $\mathbf{M}_A$ has iid $\mathcal{N}(0,1)$ entries. The trace normalization ensures $\text{tr}(\mathbf{A}) = d$, so for a random unit vector $\mathbf{x}$ drawn isotropically on the sphere, $E[\mathbf{x}^\top \mathbf{A} \mathbf{x}] = \text{tr}(\mathbf{A})/d = 1$. This fixes the average quadratic scale of the interaction operator. $\mathbf{A}$ is fixed for the duration of the simulation.

Because $\mathbf{A}$ is symmetric, $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j = \mathbf{x}_j^\top \mathbf{A} \mathbf{x}_i$, so the base interaction is symmetric without explicit symmetrization. Because $\mathbf{A}$ is positive definite, all of its eigenvalues are strictly positive, hence $\mathbf{A}$ is full rank and defines a nondegenerate quadratic form on $\mathbb{R}^d$. The trace normalization fixes only the average eigenvalue at 1, it does not impose any particular condition number. Writing the bilinear form in coordinates,

$$
\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j = \sum_{k=1}^d \sum_{l=1}^d A_{kl} x_{i,k} x_{j,l},
$$

shows that a symmetric $\mathbf{A}$ contributes $d(d+1)/2$ free coefficients.

**Regime-dependent gain.** A second symmetric matrix $\mathbf{B} \in \mathbb{R}^{d \times d}$ determines a gain that amplifies or attenuates the base interaction:

$$g(\mathbf{x}_i, \mathbf{x}_j) = 1 + \delta \cdot \text{sign}(\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j)$$

where $\delta \in [0, 1]$ (default 0.5) controls the gain strength. Because $\mathbf{B}$ is symmetric, $g(\mathbf{x}_i, \mathbf{x}_j) = g(\mathbf{x}_j, \mathbf{x}_i)$. Pairings divide into two regimes: when $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j > 0$, the gain is $(1 + \delta)$ (high-gain regime); when $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j < 0$, the gain is $(1 - \delta)$ (low-gain regime). At $\delta = 0.5$, the high-gain interaction is three times the low-gain interaction.

The implementation uses the approved `cov_full` construction. Let

$$\boldsymbol{\Sigma}_x = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i \mathbf{x}_i^\top$$

be the empirical second-moment matrix of realized agent types. First draw a symmetric Gaussian matrix $\mathbf{B}_0$ and recenter it to zero trace, then remove its weighted projection onto $\mathbf{A}$ under the inner product

$$\langle \mathbf{M}, \mathbf{N} \rangle_{\boldsymbol{\Sigma}_x} = \operatorname{tr}(\boldsymbol{\Sigma}_x \mathbf{M} \boldsymbol{\Sigma}_x \mathbf{N}).$$

Specifically,

$$\mathbf{B}_{\text{raw}} = \mathbf{B}_0 - \frac{\operatorname{tr}(\boldsymbol{\Sigma}_x \mathbf{B}_0 \boldsymbol{\Sigma}_x \mathbf{A})}{\operatorname{tr}(\boldsymbol{\Sigma}_x \mathbf{A} \boldsymbol{\Sigma}_x \mathbf{A})} \mathbf{A}, \qquad \mathbf{B} = \frac{\mathbf{B}_{\text{raw}}}{\lVert \mathbf{B}_{\text{raw}} \rVert_F}.$$

This construction makes $\mathbf{B}$ weighted-orthogonal to $\mathbf{A}$ under the bilinear form defined above, $\operatorname{tr}(\boldsymbol{\Sigma}_x \mathbf{B} \boldsymbol{\Sigma}_x \mathbf{A}) = 0$, while preserving symmetry. $\mathbf{B}$ is generally indefinite, not SPD. This is intentional: only the sign of $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j$ matters for regime assignment, so the orientation of the separating operator matters, not positive definiteness.

The gain modulates the *strength* of the base interaction without changing its sign. Among pairings with similar base interactions $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j$, those in the high-gain regime are worth substantially more than those in the low-gain regime. This difference is the source of the broker's informational advantage (§1e).

#### 1d. What controls the nature of the matching problem

- **$d_\gamma$ (active dimensions).** When $d_\gamma = d$, the type curve spans all $d$ dimensions, creating maximum diversity in the type space and the interaction effects that depend on it. When $d_\gamma < d$, the curve is confined to a lower-dimensional subspace.

- **$\rho$ (mixing weight).** At high $\rho$, general quality dominates. At low $\rho$, the gain-modulated interaction dominates.

- **$\delta$ (gain strength).** Controls the magnitude of the regime effect. At $\delta = 0$, the gain is 1 for all pairings and the DGP reduces to a simple interaction without regimes. At $\delta > 0$, the true interaction results from a mixture of two regimes. Larger $\delta$ produces a larger gap between high-gain and low-gain pairings, making the regime more consequential for match rankings.

- **$\mathbf{A}$ and $\mathbf{B}$ (interaction and regime operators).** $\mathbf{A}$ determines the base interaction structure; $\mathbf{B}$ determines the regime boundary. $\mathbf{A}$ is SPD. $\mathbf{B}$ is symmetric and constructed to be weighted-orthogonal to $\mathbf{A}$ under the realized type second moment $\boldsymbol{\Sigma}_x$. For a fixed agent $i$, the base interaction $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j = \mathbf{a}_i^\top \mathbf{x}_j$ (where $\mathbf{a}_i = \mathbf{A} \mathbf{x}_i$) is linear in $\mathbf{x}_j$. The regime boundary ($\mathbf{b}_i^\top \mathbf{x}_j = 0$, where $\mathbf{b}_i = \mathbf{B} \mathbf{x}_i$) is therefore orthogonalized away from the payoff operator in the weighted geometry induced by realized types, reducing systematic alignment between payoff ranking and regime assignment.

- **$\sigma_\varepsilon$ (noise scale).** The match-level noise $\sigma_\varepsilon = 0.10$ should be interpreted relative to the actual variance of $f$, which depends on the parameter configuration. The typical magnitude of dot products on the unit sphere in $\mathbb{R}^d$ is $O(1/\sqrt{d})$. The effective signal-to-noise ratio should be measured empirically at initialization.

#### 1e. The information gap between single- and cross-agent data

The regime-dependent gain (§1c) creates an informational gap between single-agent and cross-agent data. This gap has three important characteristics:

1. The gap is inherent to the DGP, not purely model-related.

2. The gap is fundamental, not merely statistical.

    - A purely statistical advantage depends asymptotically on data volume only and erodes as agents accumulate data. A fundamental gap involves an identification problem that single-agent data cannot solve regardless of sample size.

3. The gap affects match selection.

    - Agents use predictions to rank and select counterparties.
    - The gap causes single- and cross-agent data to produce *different rankings* among top candidates, not just more accurate point estimates or better predictions for candidates that would never be selected.

These characteristics correspond to assumptions being made through model design.

**Why the regime creates a fundamental information gap.** For a fixed agent $i$, the gain-modulated interaction produces outcomes from a *mixture* of two linear functions of $\mathbf{x}_j$. Some partners are in the high-gain regime ($g = 1 + \delta$) and others in the low-gain regime ($g = 1 - \delta$), but it is hard for agent $i$ to determine which regime each match fell into. The regime boundary (where $\mathbf{b}_i^\top \mathbf{x}_j = 0$, with $\mathbf{b}_i = \mathbf{B}\mathbf{x}_i$ because $\mathbf{B}$ is symmetric) is along a direction in $\mathbf{x}_j$ space that the agent does not know. 

The agent data is generated by the mixture:

$$q_{ij} \approx \begin{cases} (1 + \delta) \cdot \mathbf{a}_i^\top \mathbf{x}_j & \text{if } \mathbf{b}_i^\top \mathbf{x}_j > 0 \\ (1 - \delta) \cdot \mathbf{a}_i^\top \mathbf{x}_j & \text{if } \mathbf{b}_i^\top \mathbf{x}_j < 0 \end{cases}$$

(omitting the quality component and noise for clarity). 

A linear model $\boldsymbol{\beta}^\top \mathbf{x}_j$ fitted on this mixture learns an *average slope* that is systematically wrong for both regimes. To separate the two regimes, the agent needs to detect that the slope of the relationship between $\mathbf{x}_j$ and outcomes changes along the direction $\mathbf{b}_i$. This is an unsupervised change-point detection problem in $d$ dimensions that requires qualitatively more sophisticated analysis than simple linear regression, regardless of sample size.

**Why the broker can resolve the regime.** The broker observes $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ triples across many different pairings. The regime depends on $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j$, which is a *bilinear* function of both types. The broker's network receives symmetric additive and pairwise-product features (§2c), so cross-agent data make the regime boundary directly learnable from variation in both $\mathbf{x}_i$ and $\mathbf{x}_j$. The regime boundary that is hidden from the individual agent (because it requires conditioning on $\mathbf{x}_i$, which is fixed in single-agent data) is *visible* in the broker's input (because $\mathbf{x}_i$ varies across observations in cross-agent data).

**Why the gap affects match selection.** The gain operates multiplicatively on the base interaction. Among an agent's candidates with a high true interaction component, some are in the high-gain regime and others in the low-gain regime. The agent's linear model, fitting the average slope, ranks these candidates similarly. But their true match qualities differ by a factor of $(1 + \delta) / (1 - \delta)$ (3:1 at $\delta = 0.5$). The broker, knowing the regime, can identify which top candidates are high-gain and rank them above the low-gain candidates. This produces *different selections* from the same candidate pool.

### 2. Learning

After demand and channel choices are realized, but before offer formation, agents and the broker update prediction models on their accumulated histories. Those updated models are then used to rank candidates and evaluate offers in the shared market. The broker retrains at most once per period when it has new observations. Agents retrain on a deterministic alternating-parity schedule, accumulating all new observations until their next retraining period (see `simulation_pseudocode.tex`, `PeriodUpdate`).

#### 2a. Architecture and fitting

Both agents and the broker use the same architecture: a fully-connected network with one hidden layer, ReLU activations, and a single linear output:

$$\hat{q}(\mathbf{z}) = \mathbf{w}_2^\top \text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) + b_2$$

where $\mathbf{z}$ is the input feature vector, $\mathbf{W}_1$ is the hidden-layer weight matrix, $\mathbf{b}_1$ is the hidden bias vector, $\mathbf{w}_2$ is the output-layer weight vector, and $b_2$ is the output bias. Agents receive raw partner type vectors. The broker receives a symmetric pair representation defined in §2c.

**Fitting.** Each period, the network weights are updated by minimizing MSE using vanilla gradient descent on the full batch with a fixed learning rate $\eta_{\mathrm{lr}}$ (default 0.03). Gradients are computed by automatic differentiation through DifferentiationInterface with Enzyme. No explicit regularization is applied: at the data scales the broker and agents accumulate, adding weight decay has no measurable effect on held-out fit, so it is omitted for simplicity.

**Initialization.** At $t = 0$ each network is trained from random weights for $E_{\text{init}}$ gradient steps (default 200) on its seed history. The output bias $b_2$ is initialized to $Q$ (the DGP offset, §1a) rather than zero so that an untrained network predicts the population-mean match quality. This avoids a large negative-bias artifact for fresh entrants whose network has not yet been trained, and is irrelevant for mature networks (the first training steps on any real data move $b_2$ to its fitted value). All other weights follow He initialization.

**Adaptive schedule.** The network is updated each period with warm start from the previous period's weights. The number of gradient steps adapts to the ratio of new observations to total history:

$$E_t = \max\!\left(50, \; \left\lceil E_{\text{init}} \cdot \frac{n_{\text{new}}}{n_{\text{total}}} \right\rceil\right)$$

where $n_{\text{new}}$ is the number of observations added this period and $n_{\text{total}} = |\mathcal{H}^t|$ is the current history size. The floor of 50 ensures that mature networks continue to receive enough gradient updates per period to converge close to the DGP's best-achievable fit, rather than stagnating far below it.

**Training window.** To avoid diluting new observations in a large full-batch gradient, each training period uses at most the $W = 500$ most recent observations from the agent's or broker's history. The warm start preserves what was learned from older data. This sliding window ensures that the gradient reflects recent experience while remaining large enough to contain a representative cross-section of match types.

**Prediction.** Given a fitted network, the prediction for a candidate match is a single forward pass. An agent evaluates $\hat{q}_i(\mathbf{x}_j)$ for candidate partners $\mathbf{x}_j$, ranks feasible candidates by predicted or known-partner quality, and emits up to its current demand count in directed offers (§5a). Because $f$ is symmetric, the same model serves both roles: evaluating potential counterparties (as demander) and evaluating incoming proposals (as counterparty). The broker evaluates pair-level predictions over the broker-accessible unordered pair set and emits directed offers for broker demanders from that shared ranking (§5b).

#### 2b. Agent $i$'s model

**History.** $\mathcal{H}_i^t = \{(\mathbf{x}_j, q_{ij})\}_{m=1}^{n_i}$ records the other party's type and the realized match output from every match $i$ has participated in, regardless of role. Because $f$ is symmetric (§1a), observations from both roles pool into a single history.

**Input and capacity.** The agent's network takes the partner's type as input: $\mathbf{z} = \mathbf{x}_j$ ($d = 8$ inputs in the baseline). The hidden width is derived from type dimensionality, $h_A = 2d$, so the baseline uses 16 hidden units. Total parameters are $h_A(d + 2) + 1$, equal to 161 at $d = 8$.

**Why this architecture.** The regime-dependent gain (§1c) makes the agent's local prediction problem *piecewise linear*: for a fixed agent $i$, the target function is $f_i(\mathbf{x}_j) \approx (1 \pm \delta) \cdot \mathbf{a}_i^\top \mathbf{x}_j + \text{quality}$, with two different slopes on either side of a hyperplane boundary $\mathbf{b}_i^\top \mathbf{x}_j = 0$ that the agent does not know. A one-hidden-layer ReLU network is the natural function approximator for this structure: each ReLU unit computes a hinge function $\max(\mathbf{w}^\top \mathbf{x}_j + b, 0)$, and a small number of such units can represent piecewise linear functions with learned breakpoints.

With sufficient data, the network can in principle learn the piecewise linear structure. The ReLU units can discover the regime boundary as one of their activation thresholds. In practice, with sparse data, the network produces a smooth approximation that averages across regimes. As the agent accumulates observations, the approximation improves, but the identification problem persists: detecting a change in slope along an unknown direction in $\mathbb{R}^d$ from noisy data requires substantially more observations than fitting a single linear relationship.

#### 2c. Broker's model

**History.** $\mathcal{H}_b^t = \{(\mathbf{x}_i, \mathbf{x}_j, q_{ij})\}_{m=1}^{n_b}$ records both parties' types and the realized match output from every match the broker has mediated. At initialization, it is seeded from existing roster-roster ties that the broker observes without adding new agent-agent edges (§11c).

**Input and capacity.** The broker's network takes a symmetric pair feature vector:

$$
\psi(\mathbf{x}_i,\mathbf{x}_j)
= \left[
\mathbf{x}_i+\mathbf{x}_j;\;
\operatorname{vech}\!\left(\frac{\mathbf{x}_i\mathbf{x}_j^\top+\mathbf{x}_j\mathbf{x}_i^\top}{2}\right)
\right].
$$

The half-vectorization $\operatorname{vech}$ keeps the lower-triangular entries of the symmetric matrix, so the broker input dimension is $d_B = d + d(d+1)/2$, equal to 44 at $d = 8$. The broker hidden width is derived from type dimensionality, $h_B = 8d$, so the baseline uses 64 hidden units. Total parameters are $h_B(d_B + 2) + 1$, equal to 2945 at $d = 8$.

**Fitting.** The broker's network must learn a predictive approximation to the bilinear interaction structure $\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_j$ and the regime boundary $\mathbf{x}_i^\top \mathbf{B} \mathbf{x}_j$. The additive block captures symmetric one-body quality structure, while the half-vectorized symmetric outer-product block exposes the pairwise complementarity terms directly. The one-hidden-layer ReLU network then learns a nonlinear mapping from this symmetric pair representation to match quality.

Because $\psi(\mathbf{x}_i,\mathbf{x}_j)=\psi(\mathbf{x}_j,\mathbf{x}_i)$, each observed unordered pair contributes one training row. No ordered-pair symmetry augmentation is used.

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
- **Resource capture** charges no placement fee to the captured client. The broker earns or loses the principal spread between realized value and the acquisition price (§12).

The common friction is independent of realized match quality. Under resource capture (§12), $\phi$ still enters the broker's capture decision as the standard-placement fee the broker forgoes by becoming principal.

An economically important asymmetry in the illustrative markets is **search-risk transfer**. Self-search typically requires the agent to incur time, attention, or internal business-development costs for each sought relationship position whether or not the search succeeds: calling dealers, screening counterparties, traveling to trade events, preparing offers, or canvassing foreign buyers. By contrast, broker compensation is often at least partly contingent on success: a broker or intermediary is usually paid when a relationship clears, not merely for having searched. In that sense, outsourcing shifts part of the risk of failed search from the agent to the intermediary. This creates a motive for brokerage that is distinct from pure informational superiority. Even when the broker and the agent faced the same cost level $\lambda_c$, the broker could still be valuable by absorbing failed-search risk.

### 4. Network Structure and Turnover

Agents interact through a single undirected network $G$ that determines each agent's search opportunities and structural position.

#### 4a. Network initialization

$G$ is initialized as a small-world graph (Watts & Strogatz, 1998). Agents are arranged on a ring in random order, each connected to its $k_G = 6$ nearest neighbors on the ring, and each edge is rewired with probability $p_{\text{rewire}} = 0.1$. This produces the high clustering and short path lengths characteristic of small-world graphs. Agents are placed on the ring in random order (rather than, e.g., sorted by type) so that the initial network is not type-assortative: neighbors at period 0 are representative of the broader population, which avoids inflating baseline match quality through an artificially favorable neighborhood structure. An optional PC1-sorted variant is retained for robustness checks.

The broker is a permanent node in $G$, connected to all standing roster members, all current-period broker clients, and agents currently engaged in broker-channel matches (§7). The broker node has no type vector and is excluded from matching candidate pools, but is included in network measure computations (§10).

#### 4b. Match tie formation

Each realized match (whether through self-search or brokered) adds an undirected edge between the demander and counterparty in $G$, if one does not already exist. Ties persist unless one of the nodes exits, as former counterparties remain connected after their match dissolves. This is the only mechanism of network densification.

#### 4c. Agent turnover

Agents exit independently each period with probability $\eta_{\mathrm{exit}}$ (default 0.02), yielding an expected agent lifetime of 50 quarters (12.5 years).

Exiting agents are replaced by entrants with fresh types sampled from the curve at a random position $u \sim \mathrm{Unif}[0,1]$ plus noise (same procedure as initialization), empty experience histories, self-satisfaction initialized from new neighbors' self-satisfaction (word-of-mouth), and broker-satisfaction set to the current broker reputation. The exiting agent's node in $G$ is removed (along with all its edges). 

The entrant is added to $G$ with $\lfloor k_G/2 \rfloor$ edges to agents sampled from the type neighborhood (probability $\propto \exp(-\|\mathbf{x}_{i'} - \mathbf{x}_j\|^2)$). Entrants join with fewer connections than the initial network degree $k_G$ to reflect the disadvantage of being new to a market: established agents have accumulated connections through prior matches, while entrants start with only a few type-similar contacts. New entrants with sparse networks are more likely to need the broker's matching service.

### 5. Search

At the start of each period, each agent draws active relationship demand

$$d_i \sim \text{Binomial}(K,\; p_{\text{demand}}),$$

where $K$ is the maximum number of active demands the agent can initiate in the period and $p_{\text{demand}}$ defaults to 0.50. If $d_i > 0$, the agent chooses **one channel for the batch** of current-period demand (§6): self-search or broker. The period contains one shared binding-offer market. Self-searching demanders propose from self-search lists, outsourcing demanders propose from broker lists, and all accepted offers form relationships in the same shared market.

#### 5a. Self-search

Agent $i$'s self-search candidate pool has two components:

**Known neighbors.** Direct network neighbors in $G$ with no active current-period relationship with $i$ and at least one previously observed match with $i$ (equivalently, a stored partner mean). For each such neighbor $j$, the agent evaluates quality using the **average of realized outcomes** from prior matches with $j$: $\bar{q}_{ij} = \frac{1}{n_{ij}} \sum q_{ij}^{(m)}$, where $n_{ij}$ is the number of times $i$ and $j$ have matched. This is a direct empirical estimate, not a model prediction. Initial histories include all non-broker graph neighbors (§11c), so initial direct ties are known neighbors from period 1. Later graph neighbors can still lack a stored partner mean if a tie exists without a direct realized relationship.

**Strangers.** Each self-searching demander $i$ independently samples its own pool $U_i^t$ of $\min(n_{\mathrm{strangers}}, N)$ agents uniformly without replacement from the population, where $n_{\mathrm{strangers}} = 10$ (default). The demander can evaluate members of this pool that are not current neighbors and are not already current-period counterparties. The agent has no prior history with these candidates and evaluates them using its **prediction model**: $\hat{q}_i(\mathbf{x}_j)$ (§2b). Strangers represent cold outreach: attending trade events, browsing listings, or following up on indirect referrals. Independent pools avoid creating artificial same-period attention shocks in which all self-searching demanders evaluate the same small set of strangers.

Agent $i$ orders feasible self-search candidates by this demander-side evaluation and emits up to $d_i$ directed offers to candidates whose evaluation exceeds $r$.

#### 5b. Broker-mediated search

When agent $i$ outsources to the broker, the broker includes agent $i$ in its client list for the current period. Outsourcing does not alter standing roster membership (§7), but current broker clients are added to the broker's one-period access overlay. The broker therefore allocates over a hybrid access set rather than over the standing roster alone.

At the end of Step 1 (after all outsourcing decisions), the broker observes its full client list $D^t$ (the set of demanders who outsourced this period) and forms its accessible counterparty set

$$\mathcal{A}_b^t = \text{Roster}^t \cup D^t.$$

Agents already on the standing roster remain on it whether or not they outsource in the current period; current clients expand access only for the current period and do not become lagged standing members for that reason.

For every unordered pair $\{i,j\}$ such that one side is a broker-client demander and the other side is in $\mathcal{A}_b^t$, the broker computes predicted match quality. For each broker-client demander $i$, it keeps only the highest-valued accessible counterparties above $r$, up to $i$'s residual active broker demand. The selected directed broker offers are then processed in the same deterministic score order that a full global pair ranking would induce. If both sides are broker-client demanders and both have remaining active demand, the same unordered pair can therefore generate reciprocal broker offers.

**Implementation note.** The code may realize these rules with performance-oriented scratch buffers and caches, provided the stochastic object is unchanged: each self-searching demander samples an independent stranger pool, standard broker-side offers are selected by per-client residual quota before the selected union is ordered, current-period duplicate-pair exclusion may be implemented with an exact period-local index, and neural-network training still uses the same data windows and gradient steps. The implementation organizes scratch state into subsystem workspaces, including a directed-offer book for offer construction and acceptance, and a period ledger for demand, satisfaction, payment, and accepted-match buffers. Resource-capture planning still uses the full sorted broker pair list because whole-lot feasibility depends on each client's complete ranked accessible set.

#### 5c. Shared offer acceptance

All directed offers enter one shared market, grouped by unordered pair.

1. If pair $\{i,j\}$ contains reciprocal offers $i \to j$ and $j \to i$, the relationship is accepted automatically.
2. If pair $\{i,j\}$ contains one directed offer $i \to j$, receiver $j$ evaluates $i$ using $\bar{q}_{ji}$ for known partners and $\hat{q}_j(\mathbf{x}_i)$ otherwise. The offer is accepted iff this value exceeds $r$.
3. Each accepted unordered pair forms at most one current-period relationship. The realized match output is drawn once for the relationship.

There is no counterparty-side capacity conflict resolution and no outer round loop. A demander can have at most $d_i$ outgoing offers, but can also receive any number of acceptable incoming offers. This removes `K` as a counterparty-capacity constraint while retaining it as the active-demand bound.

### 6. The Outsourcing Decision

A **calibration reference** $\bar{q}_{\text{cal}} = E[q]$ is computed once at initialization from a Monte Carlo sample of random agent pairs (§11c). This is the unconditional mean match output, used to scale the reservation value $r$, broker fee $\phi$, and self-search cost $c_s$ (§11b). It is not used to initialize satisfaction indices or broker reputation; those are initialized from actual seed data (see below).

#### 6a. Satisfaction tracking

Each agent $i$ maintains a satisfaction index $s_{i,c}^t$ for each search channel $c \in \{\text{self}, \text{broker}\}$. These scores summarize past matching outcomes and drive the outsourcing decision.

The index is an exponentially weighted moving average (recency weight $\omega = 0.2$) of realized match value, net of search costs:

$$s_{i,c}^{t+1} = (1 - \omega)\,s_{i,c}^t + \omega \cdot \tilde{q}$$

where $\tilde{q}$ is the satisfaction input for the period. The averaging unit is the agent's **requested relationship demand** $d_i$. Satisfaction is updated once per period for each agent with active demand, and only for the channel chosen for that demand. Realized outputs enter through accepted directed offers made by that agent. A reciprocal relationship can therefore update two different channel-specific satisfaction states, one for each directed offer. Unfilled requested demands contribute zero output through the denominator $d_i$.

| Channel | Satisfaction input $\tilde{q}$ |
|---------|-------------------------------|
| Self-search | $\dfrac{\sum q_{ij}}{d_i} - c_s$, summing over accepted directed offers made by $i$ through self-search |
| Standard brokered (base model) | $\dfrac{\sum q_{ij} - \phi \cdot n_{i,\text{broker success}}}{d_i}$, summing over accepted directed offers made by $i$ through the broker |
| Brokered with resource capture | $\dfrac{\sum q_{ij}^{\text{std}} + \sum p_i^{\text{capture}} - \phi \cdot n_{i,\text{broker std success}}}{d_i}$, where $p_i^{\text{capture}}$ is the acquisition payment received by captured origin client $i$ |

This implies an intentional asymmetry in total-failure episodes. If a brokered batch fails completely, then $\tilde{q}=0$ and broker satisfaction decays toward zero. If a self-search batch fails completely, then $\tilde{q}=-c_s$ because the per-position search effort was paid despite filling no position. Satisfaction indices are not floored: they can go negative. The EWMA's recency weighting ensures recovery from negative values within a few good observations.

Under the approved simplification, $s_{i,\text{self}}^t$ is interpreted as the reduced-form value of the entire internal-search channel. It summarizes realized self-search outcomes, including cases where the agent reused known partners directly, rather than separating out a distinct contemporaneous "known partners" score at decision time.

**Initialization from seed data.** At initialization, each agent's self-satisfaction is set to the mean of its seed match outcomes (see `simulation_pseudocode.tex`, `Initialize`), not to an arbitrary constant. Each agent's broker-satisfaction is set to the broker's seed-data reputation (§6c). This grounds the initial outsourcing decision in actual data: agents with good neighbors start with high self-satisfaction and are harder for the broker to recruit, while agents with poor neighbors are more open to outsourcing.

**Fresh entrants.** New agents entering via turnover (§4) initialize self-satisfaction as the mean of their new neighbors' self-satisfaction (word-of-mouth: the entrant inherits the local opinion about self-search quality). Broker-satisfaction is set to the current broker reputation (the market's current opinion). The `tried_broker` flag is false, so the entrant uses broker reputation for its first outsourcing decision.

**`tried_broker` flag semantics.** The flag flips from false to true the first time the agent chooses the broker channel for any demand in a period, regardless of whether the broker's proposal led to a successful placement. Once true, the agent uses its own $s_{i,\text{broker}}^t$ rather than the broker's reputation for subsequent decisions (§6b). The rationale is that after selecting the broker once, the agent's personal EWMA has started to absorb information about that channel, including failed broker episodes that update satisfaction through a zero realized input, so reputation stops being the better signal.

#### 6b. Decision rule

Each period, an agent with $d_i$ requested relationship positions compares two scores:

- **$\text{score}_{\text{self}}$** $= s_{i,\text{self}}^t$: the EWMA satisfaction from past self-search outcomes. This is a reduced-form internal-search score and is interpreted as already incorporating the value of exploiting known partners under the self-search channel.
- **$\text{score}_{\text{broker}}$** $= s_{i,\text{broker}}^t$ if the agent has tried the broker, otherwise the broker's reputation $\text{rep}_b^t$.

The agent outsources if $\text{score}_{\text{broker}} > \text{score}_{\text{self}}$; it self-searches if $\text{score}_{\text{broker}} < \text{score}_{\text{self}}$. Only at the exact boundary $\text{score}_{\text{broker}} = \text{score}_{\text{self}}$ is the channel chosen by a uniform coin flip between self-search and broker.

This simplification treats the self-search channel as a single reduced-form outside option. Agents do not separately compute a contemporaneous "best known partners" score at decision time; instead, the value of having discovered good partners is assumed to be reflected over time in realized self-search outcomes and therefore in $s_{i,\text{self}}^t$.

The search-risk-transfer asymmetry sharpens this comparison. Self-search exposes the agent to the risk of paying for effort on requested relationship positions that yield no placement, whereas standard brokerage shifts more of that downside onto the intermediary because compensation is tied more closely to successful matching. As a result, outsourcing can be attractive not only because the broker has better information or broader access, but also because it converts some search cost from a non-contingent expenditure into a contingent payment. This mechanism is especially relevant for agents facing uncertain fill rates, sparse networks, or highly lumpy demand.

**Initial conditions.** Self-satisfaction is initialized from each agent's seed match outcomes over all initial non-broker graph neighbors. Broker-satisfaction is initialized to the broker's seed-data reputation, computed from existing roster-roster ties observed by the broker. Both values are grounded in actual data, not an arbitrary constant. Since the broker's seed reputation and the typical agent's seed self-satisfaction are close but not identical, the first period's outsourcing decisions reflect genuine (if noisy) differences in local match quality rather than a symmetric coin flip. Agents with above-average self-satisfaction prefer self-search; those with below-average self-satisfaction are more open to outsourcing. The broker's early client base is thus self-selected rather than random.

#### 6c. Broker reputation

$$\text{rep}_b^{t+1} = \begin{cases} \frac{1}{|D^t|} \sum_{i \in D^t} s_{i,\text{broker}}^{t+1} & \text{if } D^t \neq \emptyset \\[4pt] \text{rep}_b^{t} & \text{otherwise} \end{cases}$$

where $D^t$ is the set of agents who outsourced to the broker this period. When the broker has current clients, reputation is updated to the mean of their (post-update) broker satisfaction. When it has no clients, the value is held from the previous period. Reputation is initialized from the mean of the broker's seed match outcomes (see `simulation_pseudocode.tex`, `Initialize`).

### 7. Broker Roster

The broker maintains a **roster** of agents it knows and can propose as counterparties when mediating matches.

**Initialization.** The roster is seeded with a fixed target size

$$R^* = \lceil \alpha_R N \rceil, \qquad \alpha_R = 0.20,$$

by drawing $R^*$ agents uniformly at random from the population (default 200 at $N = 1000$). This ensures the broker can serve early outsourcers without frequent no-match failures that would drive broker satisfaction down before the broker has a chance to demonstrate value. The broker's history is seeded from existing roster-roster ties in the initial graph, capped at 100 observed ties. Broker seeding does not create new agent-agent ties in $G$ (§11c).

**Standing roster with replenishment.** The broker maintains this roster as a standing access base. At the start of each period, after prior-period active matches are cleared and before current-period demand is realized, each current roster member independently exits the roster with exogenous probability $p_{\text{roster}}$ (default $0.02$). The broker then replenishes uniformly at random from agents not currently on the roster until the target size $R^*$ is restored. Formally, if $\widetilde{\text{Roster}}^t$ is the post-churn roster,

$$\widetilde{\text{Roster}}^t = \{i \in \text{Roster}^{t-1} : u_i^t > p_{\text{roster}}\}, \qquad u_i^t \overset{iid}{\sim} \mathrm{Unif}[0,1],$$

then the broker samples without replacement from $\{1,\ldots,N\}\setminus \widetilde{\text{Roster}}^t$ until $|\text{Roster}^t| = R^*$, or until the population is exhausted. Standing-roster membership is therefore independent of current outsourcing decisions: outsourcing does not place an agent onto the standing roster, and being matched through the broker does not remove the agent from it.

**Current-client overlay.** In each period, the broker also maintains the one-period client set $D^t$ of agents who outsourced in that period. The broker's effective counterparty access set for period $t$ is therefore $\mathcal{A}_b^t = \text{Roster}^t \cup D^t$. This restores an endogenous access channel, because current outsourcing expands the set of agents the broker can use as counterparties in that period without requiring a lagged client-memory mechanism. The period stranger pool is a self-search opportunity set, not a broker-access set.

**Broker edges in $G$.** Broker-node edges are synchronized to the standing roster, the current client set, and agents currently engaged in broker-channel matches. This means the broker is always adjacent in $G$ to its maintained access base and its current broker clients, while current broker-mediated relationships are also represented in the period graph even when the matched agents were not already on the standing roster. Because turnover removes exiting agents immediately but replenishment occurs at the next period start, the internal standing roster can temporarily fall below $R^*$ between the exit step and the next refresh.

### 8. Match Lifecycle

Matches are bilateral relationships within a period. Once a match is accepted in the shared offer market, both parties observe the realized match output, and the relationship remains active until the next period begins. The same unordered pair cannot form a second active relationship in the same period.

**At market finalization, for each accepted match:**
1. Realized output is drawn: $q_{ij} = Q + f(\mathbf{x}_i, \mathbf{x}_j) + \varepsilon_{ij}$.
2. Both parties add the observation to their histories: $i$ adds $(\mathbf{x}_j, q_{ij})$ to $\mathcal{H}_{i}$; $j$ adds $(\mathbf{x}_i, q_{ij})$ to $\mathcal{H}_{j}$.
3. If either directed offer used the broker channel, the broker adds $(\mathbf{x}_i, \mathbf{x}_j, q_{ij})$ to $\mathcal{H}_b$ once.
4. An edge is added between $i$ and $j$ in $G$ (if not already present).

These updates take effect after the shared market has accepted all current-period offers.

**Before the next period begins:** clear the current-period match lists $M_i^t$ and $M_j^t$. The next period begins with no active current-period relationships.

### 9. Simulation Pseudocode

The implementation-level pseudocode is consolidated in `simulation_pseudocode.tex`. That document defines the notation and dimensions used by the algorithms and gives compact pseudocode for initialization, the period update, the shared active-demand offer market, optional client-origin resource capture, satisfaction updates, and period metrics.

### 10. Performance Measures

Computed on $G$ (which includes the broker as a permanent node; §4a) each measurement period. Broker structural measures are recorded after current-period matching and pure diagnostics, before agent entry/exit, so they describe the network position on which period-$t$ brokerage occurred. No agent uses these measures in its decisions; they are outputs for analysis.

#### Network measures

**Betweenness centrality.** Standard Freeman betweenness (Freeman, 1977) computed on $G$ using the Brandes (2001) algorithm adapted for single-node computation on undirected unweighted graphs. Neighbor iteration uses a compressed sparse row (CSR) adjacency structure built once per measurement call, with pre-allocated per-thread BFS workspaces for allocation-free parallel execution. The broker's betweenness is the fraction of all shortest paths that pass through the broker node, with the standard undirected normalization:

$$\mathrm{BC}_b = \frac{1}{(n-1)(n-2)} \sum_{u \neq b} \sum_{v \neq u} \frac{\sigma_{uv}(b)}{\sigma_{uv}}$$

where $n = N+1$, $\sigma_{uv}$ is the number of shortest paths from $u$ to $v$, and $\sigma_{uv}(b)$ is the number passing through $b$. The double sum counts each undirected pair from both directions; dividing by $(n-1)(n-2)$ rather than $\binom{n}{2}$ corrects for this double-counting (Brandes, 2001, p. 9). As matches create direct ties, shortest paths increasingly bypass the broker, reducing betweenness: the structural erosion that the theory predicts.

**Burt's constraint.** Computed on the broker's ego network (Burt, 1992):

$$\mathrm{Constraint}_b = \sum_j \left(p_{bj} + \sum_{h \neq b,j}
p_{bh}\, p_{hj}\right)^2$$

where $p_{bj} = 1/d_b$ is the proportion of the broker's ties invested in node $j$ (for the unweighted network), and $p_{hj} = 1/d_h$ is the proportion of intermediary $h$'s ties invested in $j$ (Everett & Borgatti, 2020). Note that the indirect term uses the intermediary's degree, not the ego's. Low constraint = broker spans structural holes. High constraint = broker's contacts are interconnected.

**Effective size.** The number of non-redundant contacts in the broker's ego network. Using the Borgatti (1997) simplification for binary undirected networks: $\text{ES}_b = d_b - 2t_b / d_b$, where $d_b$ is the broker's degree and $t_b$ is the number of ties among the broker's neighbors (not counting ties to the broker). Equivalently: $\text{ES}_b = |N(b)| - \sum_j p_{bj} \sum_{h \neq b, h \in N(b)} m_{jh}$ where $p_{bj} = 1/d_b$ and $m_{jh} = 1$ if $j$ and $h$ are connected.

#### Prediction quality

**Winner's curse / selection bias.** Both agents and the broker rank feasible counterparties by *predicted* match quality and emit offers to the top candidates. When predictions are noisy, selected candidates' predictions are systematically inflated relative to the noiseless true output $Q + f(\mathbf{x}_i, \mathbf{x}_j)$, because selection picks up positive prediction errors. This is the classic winner's curse.

**Holdout model quality.** Each period, up to 100 agents with non-empty histories are sampled without replacement. For each sampled agent $i$, $\min(40, N-1)$ non-self partners $j$ are sampled without replacement, and both agent $i$'s neural network and the broker's neural network predict the noiseless true output $Q + f(\mathbf{x}_i, \mathbf{x}_j)$ for each partner. Per-agent $R^2$, RMSE, bias, and rank correlation are computed for each model, then averaged across the sampled agents. The implementation uses the standard $R^2 = 1 - \text{SSE}/\text{SST}$ definition, equivalently $1 - \text{MSE}/\operatorname{Var}_{\text{pop}}(q)$ with the population variance denominator. Because both models are evaluated on the same agent-partner sets, the resulting metrics are directly comparable: any gap reflects the models' relative quality, not differences in evaluation samples. Holdout sampling uses a deterministic diagnostics RNG derived from the simulation seed and period; it does not consume the main simulation RNG.

**Selected-sample metrics.** Four metrics are computed each period over accepted directed offers by channel (self-search or brokered) that period. The directed-offer basis is intentional: an accepted reciprocal relationship can represent two active search decisions, possibly through different channels.

- *Selected $R^2$* $= 1 - \text{SSE}/\text{SST} = 1 - \text{MSE}/\operatorname{Var}_{\text{pop}}(q)$. Because matched counterparties are those with the highest predictions, this sample is subject to the winner's curse: predictions are systematically inflated relative to outcomes, depressing $R^2$.

- *RMSE* $= \sqrt{\frac{1}{n}\sum(\hat{q} - q)^2}$. Tracks prediction error on the output scale.

- *Bias* $= \frac{1}{n}\sum(\hat{q} - q)$. Tracks systematic over- or underprediction. Positive bias is expected in the selected sample due to the winner's curse.

- *Selected rank correlation* (Spearman's $\rho_S$). Measures whether the agent ranks matched counterparties correctly by realized output. The rank correlation is less affected by the winner's curse than $R^2$ because it is invariant to monotone transformations.

**Minimum variance threshold.** When $\text{Var}(q) < \sigma_\varepsilon^2 / 6 \approx 0.0017$ at the default $\sigma_\varepsilon=0.10$, the realized output variance in the sample is too small relative to the noise floor for $R^2$ to be informative. Below this threshold, $R^2$, bias, and rank correlation return NaN so that the row is treated uniformly as "insufficient signal" in downstream aggregation.

**Summary of prediction quality metrics:**

| Metric | What it measures | Selection bias? | Primary use |
|--------|-----------------|-----------------|-------------|
| Holdout $R^2$ and RMSE | Model quality (approximation of $Q + f$) | None (random sample, noiseless truth) | Informational advantage |
| Selected rank correlation | Matching decision quality (correct ordering) | Mild (order is more robust than level) | Allocation effectiveness |
| Selected $R^2$ and RMSE | Prediction accuracy on accepted directed offers | Strong (winner's curse) | Economic outcomes |

The broker-agent gap in holdout $R^2$ is the purest measure of the informational advantage. The gap in selected rank correlation shows whether the advantage translates into better matching decisions.

#### Other measures

All period outputs are recorded after current-period matching, satisfaction, reputation, confidence, and diagnostics updates, but before agent entry/exit turnover. This timing ensures period-$t$ measurements describe the market state produced by period-$t$ matching rather than the replacement state prepared for period $t+1$.

**Access vs. assessment decomposition.** For each accepted broker-directed offer, record whether receiver $j$ was a direct neighbor of offer sender $i$ in $G$ at the time of the offer. If not: access value (the sender could not have found this counterparty through its own network). If yes: assessment value (the sender could have found this counterparty but the broker predicted match quality better).

**Match quality by channel.** Average realized match output $\bar{q}_c^t$ per period, where $c \in \{\text{self}, \text{brokered}\}$.

**Mean channel satisfaction.** Cross-agent means of the two satisfaction states, $N^{-1}\sum_i s_{i,\text{self}}^t$ and $N^{-1}\sum_i s_{i,\text{broker}}^t$. These summarize how the market's recent realized experience with each channel evolves over time.

**Outsourcing rate.** Fraction of requested relationship positions that are outsourced to the broker: outsourced requested positions / total requested positions. A demander-level outsourcing share (fraction of demanders choosing the broker channel) is retained as a secondary diagnostic in the code, but the requested-position share is the primary quantity because the model's demand object is the relationship position.

**Standing roster size.** Number of agents currently on the broker's standing roster (§7). In the recorded period outputs, this is measured after the start-of-period refresh and before Step 5 entry/exit, so it typically equals the target $R^*$. Internally, the roster can dip below target immediately after exits and is replenished at the next period start.

**Broker access size.** Number of distinct agents in the broker's within-period access set, $|\mathcal{A}_b^t| = |\text{Roster}^t \cup D^t|$, where $D^t$ is the set of current-period broker clients. In the recorded period outputs, this is measured after current outsourcing decisions have formed $D^t$ and before Step 5 entry/exit. This is the meaningful quantity for how many agents the broker can search over in period $t$. Because current clients can already be on the standing roster, broker access size is generally smaller than standing roster size plus the number of current broker clients.

**Counterparty concentration.** The median and maximum number of current-period counterparties per agent are recorded each period. These are diagnostics, not constraints: the model no longer caps incoming accepted offers.

**Whole-network degree summaries.** Mean, median (the 0.5 quantile), minimum, and maximum agent-node degree are recorded each period. These summarize network densification among market participants and exclude the broker node from the degree distribution.

**Resource-capture diagnostics.** Under Model 1, the recorded period outputs include: whether the broker satisfied the capture readiness gate; the broker's raw live error $\kappa_b^t$ and scaled live error $\kappa_b^t/(\bar{q}_{\text{cal}} - r)$; the number of captured origin clients; the number of captured positions; the number of accepted and rejected principal positions; principal acceptance rate; principal-mode share, defined as captured positions divided by outsourced requested positions; mean principal surplus and loss rate over all captured positions, with rejected positions counted as realized zero; and principal exposure RMSE over accepted and rejected principal positions.

## Part II. Parameters, Calibration, and Initialization

### 11. Parameters

#### 11a. Parameter table

Parameters are organized into four categories reflecting their role in the analysis.

**Structural constants.** Define the model's mechanisms. Values are set by design rationale and not varied.

| Symbol | Meaning | Value | Notes |
|--------|---------|-------|-------|
| $d$ | Type dimensionality | 8 | Fixed |
| $k_G$ | Network mean degree | 6 | Watts-Strogatz ring lattice degree |
| $p_{\text{rewire}}$ | Network rewiring probability | 0.1 | Watts-Strogatz rewiring |
| $\omega$ | Satisfaction recency weight (§6a) | 0.2 | EWMA weight |
| $p_{\text{demand}}$ | Active-demand probability | 0.50 | $d_i \sim \text{Binomial}(K, p_{\text{demand}})$ |
| $n_{\mathrm{strangers}}$ | Period stranger-pool size | 10 | Sampled uniformly once per period |
| $\sigma_x$ | Type noise scale | 0.5 | Expected distance from agent to curve position |
| $\alpha_R$ | Target roster share (§7) | 0.20 | Standing roster target size is $R^* = \lceil \alpha_R N \rceil$ |

**Calibration parameters.** Set during model development. Constant in production runs.

| Symbol | Meaning | Default | Notes |
|--------|---------|---------|-------|
| $r$ | Outside option | $0.60 \cdot \bar{q}_{\text{cal}}$ | Constant for all agents; calibrated at initialization |
| $\eta_{\mathrm{lr}}$ | Learning rate | 0.03 | Vanilla gradient descent, full-batch, no weight decay |
| $E_{\text{init}}$ | Initial training steps | 200 | Full convergence at initialization; in production periods each agent retrains every other period on a deterministic parity schedule, with steps $\max(50, \lceil E_{\text{init}} \cdot n_{\text{new}} / n_{\text{total}} \rceil)$ |
| $W$ | Training window | 500 | Train on at most $W$ most recent observations (sliding window) |
| $b_2^{(0)}$ | Initial output bias | $Q$ | Untrained networks predict population-mean quality rather than zero |
| $\sigma_\varepsilon$ | Match output noise SD | 0.10 | |
| $\delta$ | Regime gain strength (§1c) | 0.5 | $\delta = 0$: no regime effect; $\delta = 1$: maximum gain contrast |
| $\lambda_c$ | Shared search-cost rate | 0.15 | $\phi = \lambda_c\cdot(\bar{q}_{\text{cal}} - r)$, $c_s = \lambda_c\cdot(\bar{q}_{\text{cal}} - r)$; $c_s$ is a self-search cost per demanded relationship position, $\phi$ a successful standard-placement fee; §11b |
| $p_{\text{roster}}$ | Standing-roster churn probability (§7) | 0.02 | Each roster member is dropped independently at the start of a period, before uniform replenishment back to $R^*$ |

Neural-network hidden widths are derived from $d$ rather than calibrated as model parameters: $h_A = 2d$ for agents and $h_B = 8d$ for the broker.

**Phase diagram axes.** Primary parameters of interest.

| Symbol | Meaning | Default | Sweep |
|--------|---------|---------|-------|
| $d_\gamma$ | Active dimensions | 8 | {2, 4, 6, 8} |
| $\rho$ | Quality-interaction mixing weight | 0.50 | {0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0} |

**Model 1 parameters.** Apply only under resource capture (§12).

| Symbol | Meaning | Default | Notes |
|--------|---------|---------|-------|
| `enable_principal` | Resource capture toggle | false | Enables client-origin whole-lot resource capture (§12) |
| `capture_min_error_obs` | Minimum broker-controlled error observations before capture | 100 | The broker must have enough live standard-brokerage or principal exposure errors before it can become principal |
| `capture_error_threshold` | Capture confidence threshold | 0.65 | Capture is allowed when $\kappa_b^t / (\bar{q}_{\text{cal}} - r)$ is at or below this threshold |

**OAT sensitivity parameters.** Varied one at a time while holding all others at defaults.

| Symbol | Meaning | Default | Sweep | Notes |
|--------|---------|---------|-------|-------|
| $K$ | Maximum active demands | 5 | {1, 2, 5, 10, 20, 50} | Upper bound for $d_i$; not a counterparty-capacity limit |
| $p_{\text{demand}}$ | Per-position demand probability | 0.50 | {0.10, 0.25, 0.50, 0.75, 0.90} | Higher values produce a thicker, faster-moving market |
| $\eta_{\mathrm{exit}}$ | Agent entry/exit rate | 0.02 | {0.01, 0.02, 0.05, 0.10} | |
| $\delta$ | Regime gain strength | 0.5 | {0, 0.25, 0.50, 0.75} | $\delta = 0$: no regime effect (pure statistical advantage) |

The activity parameters $p_{\text{demand}}$ and $K$ jointly determine the market regime. Because $K$ bounds active demand, the expected outgoing demand volume scales with $K \cdot p_{\text{demand}}$: agents in high-demand environments initiate more offers per period, reflecting a thicker, faster-moving market. Different combinations map to the illustrative domains:

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
| $T_{\text{burn}}$ | Burn-in periods (discarded) | 30 | n/a |
| $M$ | Network measure interval | 20 | n/a |

#### 11b. Search-cost calibration

The two channel costs are calibrated jointly from the average match surplus scale $(\bar{q}_{\text{cal}} - r)$ using a shared search-cost rate $\lambda_c$:

$$
\phi = \lambda_c \cdot (\bar{q}_{\text{cal}} - r), \qquad
c_s = \lambda_c \cdot (\bar{q}_{\text{cal}} - r).
$$

In the default $\lambda_c = 0.15$, both channels use the same friction level. The two quantities are computed once at initialization and held constant thereafter, but they enter realized payoffs asymmetrically: $c_s$ is charged on each requested self-search relationship position regardless of fill, whereas $\phi$ is charged only on successful standard brokered placements.

#### 11c. Initial conditions

The initialization procedure is specified in `simulation_pseudocode.tex` (`Initialize`). The key design choices are:

- Agent types are drawn at random positions on the sinusoidal curve with noise, then projected to the unit sphere (§0).
- The matching function parameters ($\mathbf{c}$, $\mathbf{A}$, $\mathbf{B}$) are drawn once and held fixed (§1).
- Calibration quantities ($\bar{q}_{\text{cal}}$, $r$, $\phi$, $c_s$) are computed from 10,000 random agent pairs (§11b).
- Each agent's history is seeded with one observation for every initial non-broker neighbor in $G$, ensuring initial predictions reflect the full local network.
- The broker's roster is seeded at the fixed target size $R^* = \lceil 0.20 \cdot N \rceil$, and its history is seeded from up to 100 existing roster-roster graph ties. Broker history seeding observes existing relationships and does not densify the initial graph.
- All neural networks are trained from random weights for $E_{\text{init}}$ steps on their seed histories before the first period (§2a). These seed histories initialize predictive capability. Resource capture, when enabled, still waits for the live confidence gate in §12 before the broker can act as principal.

#### 11d. Reproducibility

All model-event randomness flows from a single integer seed. The seed determines: type draws, the realization of $G$, matching function parameters ($\mathbf{c}$, $\mathbf{A}$, $\mathbf{B}$), broker seed roster, standing-roster churn and replenishment draws, and all subsequent model events. Holdout diagnostics use deterministic seed-and-period-derived sampling so measurement does not perturb the model-event RNG stream. Simulations are fully reproducible given (parameter dictionary, seed).

## Part III. Model Variant: Resource Capture

### 12. Client-Origin Whole-Lot Resource Capture

Resource capture is Model 1. It is disabled by default and activated by `enable_principal = true`. The broker can become principal only for opportunities brought by current broker clients. It does not acquire non-client capacity. The roster remains important as the access set against which the broker can try to place captured client-origin opportunities.

#### 12a. Capture readiness

The broker tracks a live mean absolute error scale $\kappa_b^t$ from broker-controlled exposure errors. Standard accepted broker offers contribute $|\hat{q}_b(\{i,j\}) - q_{ij}|$. Principal accepted positions contribute the same error. Principal rejected positions contribute $|\hat{q}_b(\{i,j\}) - 0|$, because the broker committed to the position but generated no output.

Resource capture can operate in period $t$ only if:

$$
n_{\text{broker error}}^t \geq n_{\min}, \qquad
\frac{\kappa_b^t}{\bar{q}_{\text{cal}} - r} \leq \kappa_{\max} .
$$

Defaults are $n_{\min} = 100$ and $\kappa_{\max} = 0.65$. The first condition prevents capture before the broker has enough live error observations; the second requires prediction error to be small relative to the calibrated surplus scale.

#### 12b. Whole-lot acquisition

Let $D_+^t$ be current broker clients with positive residual offer quota and let $\mathcal{A}_b^t = \text{Roster}^t \cup D^t$ be broker access. For each client $i \in D_+^t$, the broker considers acquiring the entire current broker-channel lot of size $d_i$. Partial capture is not allowed.

The acquisition price per captured position is

$$
p_i = \begin{cases}
|\mathcal{H}_i^t|^{-1}\sum_{q \in \mathcal{H}_i^t} q, & |\mathcal{H}_i^t| > 0 \\
\bar{q}_{\text{cal}}, & |\mathcal{H}_i^t| = 0 .
\end{cases}
$$

If $p_i < r$, the lot is ineligible. The broker ranks recipients $j \in \mathcal{A}_b^t \setminus \{i\}$ by the symmetric broker prediction $\hat{q}_b(\{i,j\})$. A lot is feasible if the broker can plan $d_i$ distinct recipients with $\hat{q}_b(\{i,j\}) > r$. A feasible lot is captured only if expected principal surplus exceeds the standard brokerage fees the broker would forgo:

$$
\sum_{j \in \mathcal{J}_i^t} \hat{q}_b(\{i,j\}) - d_i p_i > d_i \phi,
$$

where $\mathcal{J}_i^t$ is the planned recipient set. Candidate lots are processed in descending expected net gain. Before executing a lot, the broker replans against already used pairs and already captured origins so that the whole-lot feasibility condition still holds. Once a client has been captured as an origin, it is unavailable both as a standard-market candidate and as a later same-period principal recipient.

#### 12c. Principal offers and inventory risk

When lot $i$ is captured, the broker pays or guarantees $p_i$ for each of the $d_i$ positions and sets $i$'s remaining standard broker demand to zero. The broker then offers the captured client's features to each planned recipient $j$. Recipient $j$ accepts the principal offer iff $j$ evaluates $i$ above the reservation threshold using the same receiver evaluation rule as standard one-sided offers.

If $j$ accepts, $q_{ij}$ is realized, the broker receives $q_{ij} - p_i$, and the broker records $(\mathbf{x}_i,\mathbf{x}_j,q_{ij})$ in its history. If $j$ rejects, no output is realized and the broker receives $-p_i$. Rejections are recorded as zero-output principal exposures for surplus, loss, and confidence diagnostics, but not as broker training observations.

#### 12d. Lock-in and market thinning

Principal positions do not create direct $i$-$j$ network ties. The origin client $i$ does not observe the recipient's type. Recipient $j$ observes the realized outcome if it accepts, but cannot add $i$ as a known tie and cannot use $q_{ij}$ as known direct history with $i$. Neither agent receives a normal history observation or partner-mean update from a principal position. The broker therefore keeps learning while client learning and network growth are blocked.

Captured origin clients are removed from all standard self-search and standard broker-offer candidate lists for the rest of the period. This implements market thinning: valuable current client-origin opportunities are diverted out of the open direct-matching market before standard offers are formed.

#### 12e. Satisfaction and diagnostics

Captured origin clients update broker-channel satisfaction using the acquisition payment they receive. No broker fee is charged on captured positions. For broker client $i$:

$$
\tilde{q}_{i,\text{broker}} =
\frac{\sum q_{ij}^{\text{std}} + d_i^{\text{capture}} p_i - \phi n_{i,\text{broker std success}}}{d_i}.
$$

Model 1 diagnostics record captured origin clients, captured positions, accepted and rejected principal positions, principal acceptance rate, principal-mode share over outsourced requested positions, principal surplus and loss rate, capture readiness, raw and scaled broker error, and principal exposure RMSE. Visualizations may also derive lock-in quantities from accepted principal positions: each accepted principal position corresponds to one direct $i$-$j$ tie not formed and two agent history observations not recorded. These derived lock-in quantities are not separate saved output columns.

---

## Part IV. Outstanding Design Choices

### 13. Deferred Design Choices

The following design choices are deferred for future work. They are described at a conceptual level to guide subsequent development.

#### 13a. Data Capture (Model 2)

Under data capture (Proposition 3b), the broker sells access to its prediction model as a per-period subscription service. Subscribing agents use the broker's model when evaluating strangers during self-search (§5a), while continuing to match directly, learn from outcomes, and form ties. For known neighbors, subscribers still use their own historical averages. The broker earns per-period subscription revenue $\mu$ rather than per-match fees.

Data capture produces the gradual trajectory of Proposition 3b: agents keep learning and forming ties, structural erosion continues, and the broker's advantage narrows as subscribers improve their own predictions.

**Open design questions:**

**Does the broker observe outcomes of subscriber-directed matches?** If not, the broker's learning slows under data capture: subscribers use the broker's model to find better matches with strangers, but the broker doesn't see the outcomes. This creates a natural ceiling on the broker's model quality. If the broker does observe outcomes (e.g., through a reporting requirement in the subscription contract), the ceiling disappears and data capture dynamics change.

**Does subscription replace or supplement the agent's own model for stranger evaluation?** If the subscription replaces the agent's neural network entirely (the agent uses the broker's predictions for all strangers), the agent becomes dependent and its own model atrophies. If the subscription supplements (e.g., the agent uses the better of its own prediction and the broker's for each stranger), the agent's model continues to improve alongside the broker's. The replacement version is simpler and produces stronger capture dynamics; the supplement version is more realistic.

**Can subscribers also use the broker for standard placement simultaneously?** If yes, the broker can earn revenue from both subscription fees and placement fees, and subscribers benefit from both better predictions and access to the broker's roster. If no, subscription and brokerage are mutually exclusive channels.

#### 13b. Resource Capture Extensions

The baseline resource-capture mechanism is client-origin, whole-lot, and same-period only (§12). A future extension could allow cross-period inventory. That extension would require explicit holding costs, depreciation, expiry, or liquidation rules for unplaced inventory and would materially strengthen the broker's warehousing role.

#### 13c. Prediction Confidence and Uncertainty

The current model does not track per-prediction posterior uncertainty. All match predictions are point estimates. Model 1 uses the reduced-form broker confidence state $\kappa_b^t$ as an entry condition for principal mode (§12a), not as a pair-specific posterior uncertainty measure.

**Bayesian last layer.** A natural extension of the current neural network architecture (§2a): the hidden layer remains a deterministic feature extractor trained by gradient descent, but the output layer is replaced with Bayesian linear regression. Given hidden features $\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1)$ from the training data, the posterior over output weights $\mathbf{w}_2$ is available in closed form (conjugate Gaussian). For a new input $\mathbf{z}^*$, the predictive distribution is $\mathcal{N}(\boldsymbol{\mu}_{\text{post}}^\top \mathbf{h}^*, \; \sigma_\varepsilon^2 + \mathbf{h}^{*\top} \boldsymbol{\Sigma}_{\text{post}} \mathbf{h}^*)$, where the second variance term $\mathbf{h}^{*\top} \boldsymbol{\Sigma}_{\text{post}} \mathbf{h}^*$ is the *epistemic* uncertainty (large when the input is far from training data in feature space, small when it is well-covered). Implementation cost is minimal: one $h \times h$ matrix inversion per agent per period (at $h = 16$, this is trivial).

**Uses of per-prediction uncertainty:**
- *Match selection.* An upper confidence bound (UCB) rule (select the partner with the highest $\hat{q} + \kappa \cdot \hat{\sigma}$) would balance exploitation (high predicted quality) with exploration (high uncertainty), generating more informative data and accelerating learning.
- *Resource-capture decision.* A future capture mechanism could use pair-specific prediction uncertainty to avoid acquired positions where $q_{ij}$ is highly uncertain and inventory risk is greatest.
- *Outsourcing decision.* An agent whose average predictive uncertainty is high might rationally prefer the broker even when satisfaction scores are comparable.
- *Measuring the informational advantage.* The epistemic uncertainty gap between agent and broker (the broker's $\Sigma_{\text{post}}$ is smaller because it has more diverse training data) directly quantifies the informational advantage at the prediction level.

Deferred because point predictions are sufficient for the base brokerage mechanism and keep the model parsimonious. A Bayesian last layer would enrich the dynamics by adding pair-specific epistemic uncertainty, and could be added without changing the hidden-layer training procedure.

#### 13d. Pricing Alternatives

The base model uses a fixed successful-placement fee $\phi$ on standard brokered matches. Two alternative pricing mechanisms are noted for future exploration.

**Surplus-proportional fee.** $\phi = \alpha \cdot \hat{q}_b([\mathbf{x}_i; \mathbf{x}_j])$. The broker charges a fraction of its predicted match quality. This creates a recognition gap: the broker's revenue depends on its own prediction, while the agent's satisfaction depends on realized quality. Better predictions increase broker revenue, strengthening the incentive to invest in prediction accuracy.

**Prediction-based fee.** $\phi = \alpha \cdot (\hat{q}_b - \hat{q}_i)$. The broker charges for the prediction improvement it provides over the agent's own model. This directly prices the informational advantage but requires the broker to know (or estimate) the agent's prediction quality.

Both alternatives create richer dynamics but add parameters and complicate the satisfaction comparison between channels. The fixed-fee design isolates the informational channel by removing price as a margin of competition.

#### 13e. Other Design Choices

**Alternative acquisition pricing.** The baseline acquisition price is the origin client's historical mean, with lots below reservation ineligible (§12b). A future variant could use negotiated prices or fixed outside-option prices, but those alternatives would change how hard capture is and are not part of the baseline model.

## Figures

Unless otherwise stated, the base-model and capture exploration scripts generate
simulation figures with $N = 800$, $T = 200$, and 5 seeds.

**Fig. 1.** The informational mechanism.
- *Purpose:* Establishes the core mechanism: the broker learns faster than individual agents, the gap widens with matching complexity, and this drives increasing outsourcing (Propositions 1.1, 1.2, 1.3).
- *Content:* All panels at default parameters ($d_\gamma = 8$, $\rho = 0.50$), using the base active-demand offer market.
  - Panel A: time on the horizontal axis, prediction quality (holdout $R^2$) on the vertical axis. One line for the broker, one for the average agent. The broker-agent gap reflects the informational advantage and its dynamics over time. An inset shows the effect of varying $d_\gamma$.
  - Panel B: time on the horizontal axis, outsourcing rate on the vertical axis.
  - Panel C: time on the horizontal axis, average realized match output by channel (self-search, brokered).

**Fig. 2.** Decoupling of structural position from informational advantage.
- *Purpose:* The central empirical implication. Shows that betweenness centrality declines while the broker's informational advantage grows (Proposition 2.1).
- *Content:* Time on the horizontal axis, dual vertical axes for broker betweenness centrality and broker prediction quality.

**Fig. 3.** Access vs. assessment decomposition over time.
- *Purpose:* Traces the shift from network access to information assessment as the dominant source of broker value (Propositions 1.3a, 1.3b).
- *Content:* Time on the horizontal axis, fraction of brokered matches on the vertical axis, decomposed into access value (counterparty was not in demander's network) and assessment value (counterparty was reachable but broker predicted better).

**Fig. 4.** Capture dynamics and the lock-in mechanism.
- *Purpose:* Shows the client-origin transition from standard brokerage to principal activity and the lock-in mechanism of Proposition 3a.
- *Content:* Time on the horizontal axis. Panels track principal-mode share over outsourced demand, captured origins and positions, principal acceptance rate, broker scaled error and readiness, principal surplus and loss rate, and derived lock-in effects from accepted principal positions.

**Fig. 5.** Phase diagram.
- *Purpose:* Maps the conditions under which capture occurs, identifying regions of no capture, partial capture, and full capture as a function of matching complexity (Proposition 2.2).
- *Content:* Heatmap or contour plot showing the broker-agent prediction quality gap and principal-mode share under the client-origin capture mechanism.

#### SI figures

**Fig. S1.** Prediction quality decomposition.
- *Content:* Three sub-panels: $R^2$, bias, and rank correlation over time (broker and average agent).

**Fig. S2.** Attributional vs. relational channel (Proposition 1.2).
- *Content:* $\rho$ on horizontal axis; broker-agent gap in holdout $R^2$; outsourcing rate at steady state.

**Fig. S3.** OAT parameter sweeps.
- *Content:* Grid of panels varying $\eta_{\mathrm{exit}}$, $\delta$, $p_{\text{demand}}$, $K$ while holding others at defaults.

**Fig. S4.** Network visualization snapshots.
- *Content:* The network $G$ at early, middle, and late periods. Broker node positioned centrally. Under Model 1, captured positions suppress direct client-recipient ties.

**Fig. S5.** Broker risk profile.
- *Purpose:* Shows the frequency and magnitude of inventory losses under client-origin capture.
- *Content:* Time on the horizontal axis, with principal surplus, principal loss rate, principal acceptance rate, and scaled broker error.

**Fig. S6.** Network degree diagnostics in the base-model exploration.
- *Purpose:* Tracks how the overall connectivity of $G$ evolves as the market densifies, complementing the broker-centered structural-hole measures.
- *Content:* Three time-series panels with time on the horizontal axis and agent-network degree statistics on the vertical axis: mean and median degree in a shared panel, minimum degree, and maximum degree (computed over agent nodes only, excluding the broker). A fourth panel shows the histogram of the agent-node degree distribution in the last recorded period, pooled across seeds.

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
