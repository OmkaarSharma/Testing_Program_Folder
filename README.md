# Chaotic Chameleon Swarm Algorithm for Economic Load Dispatch with AC-Accurate Transmission Losses

A Python implementation of the **Chaotic Chameleon Swarm Algorithm (CCSA)** for the **Economic Load Dispatch (ELD)** problem, coupled to a **Fast Decoupled Load Flow (FDLF)** solver so that transmission losses are computed from a converged AC power flow at every fitness evaluation rather than approximated by a $B$-coefficient loss formula.

Validated on two systems:

| System | Buses | Branches | Generators | Demand |
|---|---|---|---|---|
| IEEE 30-bus | 30 | 42 | 6 | 283.40 MW |
| Indian Utility 62-bus | 62 | 89 | 19 | 2908.00 MW |

---

## Table of Contents

- [Motivation](#motivation)
- [Problem Formulation](#problem-formulation)
- [Fast Decoupled Load Flow](#fast-decoupled-load-flow)
- [Chaotic Chameleon Swarm Algorithm](#chaotic-chameleon-swarm-algorithm)
- [Chaotic Economically-Guided Repair Operator](#chaotic-economically-guided-repair-operator)
- [Experimental Protocol](#experimental-protocol)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Implementation Notes and Caveats](#implementation-notes-and-caveats)
- [References](#references)

---

## Motivation

Most metaheuristic ELD studies handle transmission losses through Kron's loss formula

$$
P_L \;=\; \sum_{i}\sum_{j} P_{G,i}\,B_{ij}\,P_{G,j} \;+\; \sum_i B_{0i}P_{G,i} \;+\; B_{00},
$$

whose $B$-coefficients are derived from a **single linearisation about one operating point**. During optimisation the swarm visits dispatch vectors far from that point, so the loss estimate — and therefore the feasibility of the power-balance constraint — degrades exactly where the search is most active.

This implementation replaces the loss surrogate with a converged AC solution. At each fitness evaluation the FDLF solver returns

$$
P_L(\mathbf{P}_G) \;=\; \sum_{(k,m)\in\mathcal{B}} \Big[\Re\{S_{km}\} + \Re\{S_{mk}\}\Big],
$$

so the loss term is consistent with the network state, generator reactive limits are enforced through PV$\rightarrow$PQ bus-type switching, and the reported dispatch corresponds to a physically realisable operating point.

---

## Problem Formulation

### Decision vector

$$
\mathbf{P}_G = \left[P_{G,1},\, P_{G,2},\, \ldots,\, P_{G,N_G}\right]^{\!\top} \in \mathbb{R}^{N_G},
\qquad N_G \in \{6,\,19\}
$$

### Objective function

**IEEE 30-bus** — quadratic fuel cost with valve-point loading effect (VPE):

$$
F(\mathbf{P}_G) \;=\; \sum_{j=1}^{N_G}
\Big[\,\underbrace{a_j P_{G,j}^2 + b_j P_{G,j} + c_j}_{\text{smooth quadratic}}
\;+\;
\underbrace{\left| e_j \sin\!\big( f_j (P_{G,j}^{\min} - P_{G,j}) \big) \right|}_{\text{valve-point ripple}}\,\Big]
$$

**Indian Utility 62-bus** — smooth quadratic only:

$$
F(\mathbf{P}_G) \;=\; \sum_{j=1}^{19} \left( a_j P_{G,j}^2 + b_j P_{G,j} + c_j \right)
$$

The VPE term makes the 30-bus cost surface non-convex and non-differentiable, which is precisely the regime where gradient methods fail and population-based search is justified.

### Constraints

**1 — Active power balance (with AC losses):**

$$
\sum_{j=1}^{N_G} P_{G,j} \;-\; \sum_{i=1}^{N_B} P_{D,i} \;-\; P_L(\mathbf{P}_G) \;=\; 0
$$

**2 — Generator active limits with spinning reserve margin:**

$$
P_{G,j}^{\min} \;\le\; P_{G,j} \;\le\; P_{G,j}^{\mathrm{eff}},
\qquad
P_{G,j}^{\mathrm{eff}} \;=\; P_{G,j}^{\min} + 0.90\left(P_{G,j}^{\max} - P_{G,j}^{\min}\right)
$$

Each unit is held to 90 % of its operating range, so the system carries

$$
R_{\text{spin}} \;=\; 0.10 \sum_{j=1}^{N_G}\left(P_{G,j}^{\max} - P_{G,j}^{\min}\right)
$$

MW of distributed spinning reserve. The reported utilisation is measured against the **full** nameplate range $\left(P^{\max}-P^{\min}\right)$, so a unit pinned at its effective ceiling appears at $\approx 89\text{–}90\,\%$.

**3 — Generator reactive limits:**

$$
Q_{G,j}^{\min} \;\le\; Q_{G,j} \;\le\; Q_{G,j}^{\max}
$$

enforced *twice*: structurally inside the FDLF solver by bus-type switching, and economically through a quadratic penalty in the fitness.

**4 — Nodal power flow equations** (satisfied by construction at every evaluation):

$$
P_i \;=\; V_i \sum_{k=1}^{N_B} V_k \left( G_{ik}\cos\delta_{ik} + B_{ik}\sin\delta_{ik} \right)
$$

$$
Q_i \;=\; V_i \sum_{k=1}^{N_B} V_k \left( G_{ik}\sin\delta_{ik} - B_{ik}\cos\delta_{ik} \right),
\qquad \delta_{ik} = \delta_i - \delta_k
$$

### Penalised fitness

Constraints are handled by static exterior penalties:

$$
\Phi(\mathbf{P}_G) \;=\; F(\mathbf{P}_G)
\;+\; \lambda \,\Delta P^{\,2}
\;+\; \lambda \sum_{j=1}^{N_G}\Big[\max\big(0,\;Q_{G,j}-Q_{G,j}^{\max}\big)^2 + \max\big(0,\;Q_{G,j}^{\min}-Q_{G,j}\big)^2\Big]
\;+\; 10^{9}\,\mathbb{1}\!\left[\text{FDLF diverged}\right]
$$

where

$$
\Delta P \;=\; \left| \sum_j P_{G,j} - \sum_i P_{D,i} - P_L \right|,
\qquad
\lambda = \begin{cases}
10^{4} & \text{IEEE 30-bus}\\[2pt]
10^{6} & \text{Indian 62-bus}
\end{cases}
$$

The $10^9$ divergence term guarantees that any dispatch driving the load flow to non-convergence is dominated by *every* feasible candidate, so infeasible regions are never exploited by the swarm.

---

## Fast Decoupled Load Flow

The solver implements the classical Stott–Alsaç decoupling. Under the standard transmission-network assumptions $r_{km} \ll x_{km}$, $\delta_{km}$ small, and $V \approx 1$ p.u., the Jacobian's off-diagonal blocks vanish and the Newton update separates into two constant-matrix half-sweeps:

$$
\mathbf{B}' \, \Delta\boldsymbol{\delta} \;=\; \frac{\Delta \mathbf{P}}{V},
\qquad\qquad
\mathbf{B}'' \, \Delta \mathbf{V} \;=\; \frac{\Delta \mathbf{Q}}{V}
$$

with the **XB** formulation used here:

$$
B'_{km} = -\frac{1}{x_{km}}, \qquad
B'_{kk} = \sum_{m \in \Omega_k} \frac{1}{x_{km}}
$$

$$
B''_{km} = -\frac{1}{x_{km}}, \qquad
B''_{kk} = \sum_{m \in \Omega_k} \left( \frac{1}{x_{km}} + \frac{b^{sh}_{km}}{2} \right)
$$

$\mathbf{B}'$ spans all non-slack buses; $\mathbf{B}''$ spans the PQ set. Both are **constant across iterations**, so each is LU-factorised once via `scipy.linalg.lu_factor` and every subsequent iteration costs only a triangular solve, $\mathcal{O}(n^2)$ instead of $\mathcal{O}(n^3)$.

### Network matrix caching

$\mathbf{Y}_{\text{bus}}$, $\mathbf{G}$, $\mathbf{B}$, and the $\mathbf{B}'$ LU factors depend on topology alone, never on the dispatch. They are built once and cached:

```python
cache_key = (id(bus_data), id(branch_data), int(slack_bus_idx))
```

The cache is pre-warmed in the parent process before forking, so all 8 workers inherit the factorisation. Since CCSA calls the load flow $\mathcal{O}(10^5)$ times per trial, this is the single largest speedup in the pipeline. $\mathbf{B}''$ is deliberately **excluded** from the cache — it must be rebuilt whenever Q-limit enforcement changes the PQ set.

### Reactive limit enforcement by bus-type switching

After convergence, each PV bus is checked:

$$
Q_{G,j} \;=\; \big(Q_{i}^{\text{calc}} + Q_{D,i}\big)\cdot S_{\text{base}}
$$

If $Q_{G,j} > Q_j^{\max}$ or $Q_{G,j} < Q_j^{\min}$, the bus is reclassified PV $\rightarrow$ PQ with $Q$ clamped at the violated limit, $\mathbf{B}''$ is refactorised for the new PQ set, and the flow is re-solved. Up to 10 enforcement passes are permitted. This is the same mechanism used in MATPOWER and PSS/E. Crucially, the optimiser, the fitness function and the repair operator are untouched by it — only the physics layer changes.

### Loss computation (full $\pi$-model)

Once converged, with $\mathbf{V} = V\,e^{\,j\delta}$, complex flows at both ends of every branch are

$$
S_{km} \;=\; V_k \left[ y_{km}\left( V_k - V_m \right) + \frac{j\,b^{sh}_{km}}{2} V_k \right]^{*},
\qquad
S_{mk} \;=\; V_m \left[ y_{km}\left( V_m - V_k \right) + \frac{j\,b^{sh}_{km}}{2} V_m \right]^{*}
$$

and total system losses follow as

$$
P_L \;=\; \sum_{(k,m)} \Re\{S_{km} + S_{mk}\},
\qquad
Q_L \;=\; \sum_{(k,m)} \Im\{S_{km} + S_{mk}\}
$$

Convergence tolerance $\varepsilon = 10^{-4}$ p.u. on $\max|\Delta P|$ and $\max|\Delta Q|$; iteration cap 50. Bus angles are wrapped to $(-\pi, \pi]$ and PQ voltages clipped to $[0.5, 1.5]$ p.u. each sweep for numerical hygiene.

---

## Chaotic Chameleon Swarm Algorithm

CSA (Braik, 2021) models three behaviours of a chameleon hunting: **searching** for prey, **rotating the eyes** to localise it, and **projecting the tongue** to capture it. The chaotic variant replaces every uniform pseudo-random draw with iterates of the logistic map.

### Chaotic sequence generator

$$
x_{n+1} \;=\; A\,x_n\,(1 - x_n), \qquad A = 4.0
$$

At $A = 4$ the map is fully chaotic on $(0,1)$ with invariant density $\rho(x) = \left[\pi\sqrt{x(1-x)}\right]^{-1}$. The seed $x_0$ is drawn from OS entropy (`os.urandom`) and rejected unless

$$
x_0 \in (0.01,\, 0.99) \quad\text{and}\quad \min_{\tau \in \{0,\,0.25,\,0.5,\,0.75,\,1\}} |x_0 - \tau| \ge 0.01
$$

The excluded set contains the fixed points and short-period orbits of the map; seeding near them collapses the sequence to a periodic cycle and destroys the diversity the algorithm depends on.

### Control parameters

| Symbol | 30-bus | 62-bus | Meaning |
|---|---|---|---|
| $n$ | 20 | 50 | Population size |
| $d$ | 6 | 19 | Dimensionality ($=N_G$) |
| $T$ | 300 | 300 | Iterations |
| $P_p$ | 0.10 | 0.10 | Search/exploration switch probability |
| $p_1,\,p_2$ | 0.25, 1.50 | 0.25, 1.50 | Search-phase coefficients |
| $\gamma,\,\alpha,\,\beta$ | 1, 3.5, 3 | 1, 3.5, 3 | Exploration decay shape |
| $c_1,\,c_2$ | 1.75, 1.75 | 1.75, 1.75 | Hunting acceleration coefficients |
| $\rho$ | 1 | 1 | Inertia exponent |

### Phase 1 — Searching for prey

$$
y_{i,j}^{t+1} =
\begin{cases}
y_{i,j}^{t} \;+\; p_1\left(P_{i,j}^{t} - G_j^{t}\right) r_2 \;+\; p_2\left(G_j^{t} - y_{i,j}^{t}\right) r_1, & r_i \ge P_p \\[8pt]
y_{i,j}^{t} \;+\; \mu \left[ \left(u_j - l_j\right) r_3 + l_j \right] \cdot \operatorname{sgn}(r_4 - 0.5), & r_i < P_p
\end{cases}
$$

where $P_i$ is chameleon $i$'s personal best, $G$ the global best, $l_j = P_{G,j}^{\min}$, $u_j = P_{G,j}^{\mathrm{eff}}$, and all $r_\bullet$ are successive logistic-map iterates. The exploration amplitude decays as

$$
\mu \;=\; \gamma \left( e^{-\alpha t / T} \right)^{\beta}
$$

so the random-walk branch is aggressive early and effectively silent by $t \approx T/2$.

### Phase 2 — Eye rotation

The chameleon rotates its eyes about the swarm centroid $\bar{\mathbf{y}}^t = \frac{1}{n}\sum_i \mathbf{y}_i^t$:

$$
\mathbf{y}_i^{t+1} \;=\; \mathbf{R}(\theta,\,k)\left( \mathbf{y}_i^{t} - \bar{\mathbf{y}}^{t} \right) + \bar{\mathbf{y}}^{t}
$$

$\mathbf{R}$ is built as a product of Givens rotations over every coordinate plane not containing the selected axis $k \in \{0,1\}$:

$$
\mathbf{R}(\theta,\,k) \;=\; \prod_{\substack{(i,j)\,\in\,\binom{\mathcal{A}_k}{2}}} \mathbf{G}_{ij}(\theta),
\qquad
\mathcal{A}_k = \{0,\dots,d-1\}\setminus\{k\},
\qquad
\theta = 2\pi\,\chi_t
$$

Because rotation is norm-preserving about the centroid, this phase redistributes the population **without** collapsing its spread — it is a diversity-preserving move, not an intensification move.

### Phase 3 — Hunting (tongue projection)

$$
v_{i,j}^{t+1} \;=\; \omega\, v_{i,j}^{t} \;+\; c_1\left(G_j - y_{i,j}^{t}\right) r_1 \;+\; c_2\left(P_{i,j} - y_{i,j}^{t}\right) r_2
$$

$$
y_{i,j}^{t+1} \;=\; y_{i,j}^{t} \;+\; \frac{\left(v_{i,j}^{t+1}\right)^2 - \left(v_{i,j}^{t}\right)^2}{2\,a}
$$

The position update is the kinematic relation $s = \left(v_f^2 - v_i^2\right)/2a$ — the tongue's projectile displacement — rather than the $\Delta t \cdot v$ used in PSO. Inertia and tongue acceleration evolve as

$$
\omega \;=\; \left(1 - \frac{t}{T}\right)^{\rho \sqrt{t/T}},
\qquad
a \;=\; 2590 \left( 1 - e^{-\log(t+1)} \right)
$$

### Diversity restart

If the global best stagnates for 20 consecutive iterations, the worst-performing members are reinitialised chaotically over $[P^{\min}, P^{\mathrm{eff}}]$:

$$
n_{\text{restart}} = \max\!\left(2,\; \left\lfloor n/4 \right\rfloor\right) \;\text{(30-bus)},
\qquad
n_{\text{restart}} = \max\!\left(1,\; \left\lfloor n/10 \right\rfloor\right) \;\text{(62-bus)}
$$

The elite and the incumbent global best are never disturbed.

---

## Chaotic Economically-Guided Repair Operator

Penalty terms alone converge slowly on an equality constraint as tight as power balance, so every candidate is projected onto the feasible manifold before evaluation. Given the FDLF loss at the current point, the target generation and imbalance are

$$
P_{\text{target}} = \sum_i P_{D,i} + P_L(\mathbf{P}_G),
\qquad
\Delta = P_{\text{target}} - \sum_j P_{G,j}
$$

If $|\Delta| < 1$ MW the candidate is returned unchanged. Otherwise the adjustable set is

$$
\mathcal{A} =
\begin{cases}
\{\,j : P_{G,j} < P_{G,j}^{\mathrm{eff}}\,\} & \Delta > 0 \\
\{\,j : P_{G,j} > P_{G,j}^{\min}\,\} & \Delta < 0
\end{cases}
$$

and the imbalance is distributed according to **incremental cost**

$$
\lambda_j \;=\; \frac{\partial F_j}{\partial P_{G,j}} \;=\; 2 a_j P_{G,j} + b_j
$$

with chaotically perturbed weights

$$
w_j \;=\;
\begin{cases}
\dfrac{\chi_j}{\lambda_j}, & \Delta > 0 \quad \text{(load the cheap units first)}\\[10pt]
\lambda_j\,\chi_j, & \Delta < 0 \quad \text{(back off the expensive units first)}
\end{cases}
\qquad
P_{G,j} \leftarrow \operatorname{clip}\!\left( P_{G,j} + \Delta\,\frac{w_j}{\sum_{k \in \mathcal{A}} w_k},\; P_{G,j}^{\min},\; P_{G,j}^{\mathrm{eff}} \right)
$$

where $\chi_j$ is a fresh logistic-map iterate. This is a stochastic relaxation of the classical equal-incremental-cost condition $\lambda_1 = \lambda_2 = \cdots = \lambda_{N_G}$: the repair pushes the dispatch *toward* the merit order without deterministically enforcing it, so the operator restores feasibility while preserving population diversity. Repair is applied after **each** of the three CSA phases.

---

## Experimental Protocol

Metaheuristic results from a single run are not evidence. The benchmark runs **30 independent trials** per system and reports the full distribution.

Trial decorrelation uses **burn-in offsets** on the chaotic stream: trial $k$ discards $100k$ iterates of the logistic map after seeding,

$$
x^{(k)}_{\text{start}} \;=\; \underbrace{L^{\,100k}}_{\text{burn-in}}\!\left( x_0^{(k)} \right),
\qquad L(x) = 4x(1-x)
$$

Because the map has positive Lyapunov exponent $\lambda_{\mathrm{L}} = \ln 2 \approx 0.693$, two streams separated by 100 iterates are numerically independent, guaranteeing that the 30 trials sample genuinely distinct search trajectories rather than perturbations of one.

Trials execute across **8 worker processes** via `multiprocessing.Pool.imap_unordered`. BLAS thread counts are pinned to 1 (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, …) — oversubscribing 16 logical threads against 8 processes causes cache thrashing and drops CPU utilisation to roughly 40 % on memory-bandwidth-bound triangular solves.

---

## Results

### IEEE 30-bus (283.40 MW demand, 30 trials)

| Metric | Best | Mean | Worst | Std Dev |
|---|---|---|---|---|
| **Fuel cost ($/h)** | 834.8087 | 860.6465 | 934.5993 | 31.3258 |
| **Active loss (MW)** | 7.6020 | 9.7278 | 13.8504 | 1.6785 |

Total wall-clock: **132.34 s** (00h 02m 12s), 8 workers.

**Best dispatch — trial 30, burn-in offset 2900:**

| Gen | Bus | $P_G$ (MW) | $P^{\min}$ | $P^{\max}$ | $P^{\mathrm{eff}}$ | Util (%) | Cost ($/h) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 124.7998 | 50.00 | 200.00 | 185.00 | 49.87 | 308.0059 |
| 2 | 2 | 69.8666 | 20.00 | 80.00 | 74.00 | 83.11 | 207.6899 |
| 3 | 5 | 23.7423 | 15.00 | 50.00 | 46.50 | 24.98 | 58.9732 |
| 4 | 8 | 29.3541 | 10.00 | 35.00 | 32.50 | 77.42 | 102.5872 |
| 5 | 11 | 20.5074 | 10.00 | 30.00 | 28.00 | 52.54 | 72.0361 |
| 6 | 13 | 23.7893 | 12.00 | 40.00 | 37.20 | 42.10 | 85.5162 |

```
Total Generation   : 292.0595 MW
Total Demand       : 283.4000 MW
Total Active Losses:   8.6597 MW
P-balance Error    :   0.000186 MW
Total Fuel Cost    : 834.8087 $/h
```

Voltage profile stays within $[0.9979,\,1.0513]$ p.u. across all 30 buses; no bus violates the 0.95 lower bound.

### Indian Utility 62-bus (2908.00 MW demand, 30 trials)

| Metric | Best | Mean | Worst | Std Dev |
|---|---|---|---|---|
| **Fuel cost ($/h)** | 14232.5850 | 14536.1844 | 14851.8895 | 154.7433 |
| **Active loss (MW)** | 68.5804 | 80.9553 | 101.5077 | 7.1100 |

Total wall-clock: **2244.50 s** (00h 37m 24s), 8 workers.

**Best dispatch — trial 4, burn-in offset 300:**

| Gen | Bus | $P_G$ (MW) | $P^{\min}$ | $P^{\max}$ | $P^{\mathrm{eff}}$ | Util (%) | Cost ($/h) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 87.2517 | 50.00 | 300.00 | 275.00 | 14.90 | 741.6016 |
| 2 | 2 | 280.1017 | 50.00 | 450.00 | 410.00 | 57.53 | 1581.9202 |
| 3 | 3 | 158.8196 | 50.00 | 450.00 | 410.00 | 27.20 | 819.0088 |
| 4 | 4 | 89.0502 | 0.00 | 100.00 | 90.00 | 89.05 | 105.5175 |
| 5 | 5 | 138.8214 | 50.00 | 300.00 | 275.00 | 35.53 | 774.2065 |
| 6 | 8 | 226.1427 | 50.00 | 450.00 | 410.00 | 44.04 | 1275.8436 |
| 7 | 11 | 127.8188 | 50.00 | 200.00 | 185.00 | 51.88 | 748.9432 |
| 8 | 13 | 209.2268 | 50.00 | 500.00 | 455.00 | 35.38 | 1420.4530 |
| 9 | 16 | 110.6860 | 0.00 | 600.00 | 540.00 | 18.45 | 823.2530 |
| 10 | 17 | 90.0000 | 0.00 | 100.00 | 90.00 | 90.00 | 119.2000 |
| 11 | 23 | 139.2939 | 50.00 | 150.00 | 140.00 | 89.29 | 375.1829 |
| 12 | 26 | 44.5724 | 0.00 | 50.00 | 45.00 | 89.14 | 120.8533 |
| 13 | 32 | 272.7430 | 50.00 | 300.00 | 275.00 | 89.10 | 937.8809 |
| 14 | 34 | 132.5344 | 0.00 | 150.00 | 135.00 | 88.36 | 376.0993 |
| 15 | 37 | 149.2889 | 0.00 | 500.00 | 450.00 | 29.86 | 926.5245 |
| 16 | 49 | 139.0957 | 50.00 | 150.00 | 140.00 | 89.10 | 371.7983 |
| 17 | 50 | 82.6032 | 0.00 | 100.00 | 90.00 | 82.60 | 97.2710 |
| 18 | 56 | 270.0298 | 50.00 | 300.00 | 275.00 | 88.01 | 785.1699 |
| 19 | 61 | 235.8156 | 100.00 | 600.00 | 550.00 | 27.16 | 1831.8574 |

```
Total Generation   : 2983.8959 MW
Total Demand       : 2908.0000 MW
Total Active Losses:   75.9013 MW
P-balance Error    :    0.005372 MW
Total Fuel Cost    : 14232.5850 $/h
```

Two generators (buses 23 and 61) sit exactly at $Q^{\max} = 150$ MVAr — these are the buses the solver switched PV$\rightarrow$PQ during reactive-limit enforcement, and the clamp is exact rather than approximate. Bus 53 records the minimum voltage at 0.9633 p.u.

### Reading the distributions

The 30-bus results are visibly **bimodal**: 19 trials converge into the 834–851 $/h band while 8 land in the 890–935 $/h band, and the higher-cost cluster is systematically accompanied by higher losses ($\approx$ 11.4–13.9 MW versus $\approx$ 8.3–9.2 MW). This is not noise — it is a distinct local basin, and the valve-point ripple is the natural suspect. The coefficient of variation is 3.6 %.

The 62-bus distribution is unimodal with a coefficient of variation of 1.06 %, which is the expected signature of a smooth quadratic objective in higher dimensions: no cost ripple, no basin structure, just dispersion around a single attractor. Power-balance error stays below $6\times10^{-3}$ MW on all 30 trials — roughly two parts in $10^{6}$ of demand.

---

## Repository Structure

```
.
├── fdlf_ccsa_30_bus.py              # IEEE 30-bus: CCSA + FDLF, valve-point objective
├── fdlf_ccsa_62_bus.py              # Indian 62-bus: CCSA + FDLF, quadratic objective
├── results/
│   ├── IEEE_30_BUS_SYSTEM.txt       # 30-trial benchmark log
│   └── Indian_Utility_62_bus.txt    # 30-trial benchmark log
└── README.md
```

Both scripts are self-contained: network data, cost coefficients, the FDLF solver, the optimiser and the reporting layer live in a single file with no external data dependencies.

---

## Usage

### Requirements

```
python >= 3.8
numpy
scipy
```

```bash
pip install numpy scipy
```

### Running

```bash
python fdlf_ccsa_30_bus.py     # ~2 minutes on 8 cores
python fdlf_ccsa_62_bus.py     # ~37 minutes on 8 cores
```

Each script prints per-trial results as they complete, then a full benchmark table, the statistical summary, and a detailed report for the best trial (dispatch, voltage profile, generator reactive output, power balance verification).

### Tuning

```python
n_cham = 50          # population size
iters  = 300         # iterations per trial
N_WORKERS = 8        # parallel processes — match physical, not logical, cores
```

`N_WORKERS` should track **physical** core count. Setting it to the logical thread count degrades throughput because the LU solves are memory-bandwidth-bound.

To capture a run:

```bash
python fdlf_ccsa_62_bus.py | tee results/run_$(date +%Y%m%d_%H%M).txt
```

---

## Implementation Notes and Caveats

Stated plainly, since these matter for anyone reproducing or extending the work:

**Modelling assumptions on the 62-bus system.** Reactive demand is synthesised as $Q_{D,i} = 0.3\,P_{D,i}$ and line charging as $b^{sh}_{km} = 0.3\,x_{km}$, both uniform across the network. Only $r$ and $x$ come from the source data. The 30-bus system uses tabulated $Q_D$ and $b^{sh}$ values throughout, so its load flow is the more faithful of the two. Any conclusion about 62-bus *reactive* behaviour inherits these assumptions.

**Transformer taps.** Off-nominal tap ratios are not modelled. Several 30-bus branches with $r = 0$ are transformers, represented here as pure series reactances with $t = 1.0$. Reported losses are therefore mildly optimistic relative to a tap-modelled solution.

**Cost verification column.** For the 30-bus system the per-generator `Cost($/h)` column in the dispatch table prints the quadratic term only, while the objective driving the optimisation includes valve-point loading. The "Sum of Gen Costs" verification line will consequently differ from "Total Fuel Cost" by the aggregate VPE contribution (834.8084 vs 834.8087 in the reported run, where the sine terms happen to fall near a zero crossing). The 62-bus system has no VPE, so its two figures agree exactly.

**Utilisation column.** `Util_eff(%)` is computed against the full nameplate range $\left(P^{\max}-P^{\min}\right)$, not against $P^{\mathrm{eff}}$, despite the header. A unit pinned at its effective ceiling therefore reads $\approx 89\text{–}90\,\%$ rather than 100 %.

**Cache keying.** The FDLF cache keys on `id()` of the bus and branch lists. This is valid within a process and survives `fork`, but would silently mis-key if the data structures were rebuilt or mutated mid-run. It is safe as written; it is not safe to copy into a codebase that regenerates network data dynamically.

**No security constraints.** Branch thermal limits are not enforced, so this solves ELD, not SCOPF. Adding $|S_{km}| \le S_{km}^{\max}$ as a further penalty term is the natural next step and requires no change to the solver architecture — branch flows are already computed for the loss calculation.

**On the metaheuristic itself.** The FDLF coupling, the Q-limit switching, and the incremental-cost repair operator are the substantive engineering contributions here. The chaotic-map substitution improves diversity relative to a uniform RNG, but comparisons against PSO, DE, or GA under an identical loss model — with a Wilcoxon signed-rank test on the 30-trial distributions — would be needed before claiming algorithmic superiority. This repository does not make that claim.

---

## References

1. M. S. Braik, "Chameleon Swarm Algorithm: A bio-inspired optimizer for solving engineering design problems," *Expert Systems with Applications*, vol. 174, 2021.
2. B. Stott and O. Alsaç, "Fast Decoupled Load Flow," *IEEE Transactions on Power Apparatus and Systems*, vol. PAS-93, no. 3, pp. 859–869, 1974.
3. R. A. M. van Amerongen, "A general-purpose version of the fast decoupled load flow," *IEEE Transactions on Power Systems*, vol. 4, no. 2, pp. 760–770, 1989.
4. A. J. Wood, B. F. Wollenberg, and G. B. Sheblé, *Power Generation, Operation, and Control*, 3rd ed., Wiley, 2013.
5. R. D. Zimmerman, C. E. Murillo-Sánchez, and R. J. Thomas, "MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education," *IEEE Transactions on Power Systems*, vol. 26, no. 1, pp. 12–19, 2011.
6. D. C. Walters and G. B. Sheblé, "Genetic algorithm solution of economic dispatch with valve point loading," *IEEE Transactions on Power Systems*, vol. 8, no. 3, pp. 1325–1332, 1993.

---

## Author

**Omkaar Sharma** — B.Tech Electrical Engineering, Institute of Technology, Nirma University, Ahmedabad
Under the supervision of **Dr. SantoshKumar Vora**, Department of Electrical Engineering, Nirma University.

---

## License

Released under the MIT License. The IEEE 30-bus data is standard published test-system data; the 62-bus system is a test case widely used in the economic dispatch literature.
