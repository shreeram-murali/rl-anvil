<!-- Converted from Exploration_in_RL (4).pdf — 29 pages -->

## Page 1

## SUN: Reaching for Novelty in
## Reinforcement Learning
### Anonymous Author(s)
### Affiliation
### Address
### email
## Abstract
1
### Exploration in reinforcement learning (RL) remains a fundamental challenge.
2
### Recent goal-conditioned RL strategies (which select goals to encourage broader
3
### state coverage) have shown promising results but lack a unified objective that
4
### accounts for both novelty and reachability — some states appear desirable yet
5
### remain unattainable at the agent’s current capabilities. In this paper, we introduce a
6
### reachability-aware goal-selection framework that explicitly integrates these two
7
### aspects, and that can be seamlessly incorporated into any off-policy RL algorithm.
8
### To this aim, we propose SUccessor-to-Novelty (SUN), an indicator derived from
9
### successor value functions to identify goals that are both novel and reachable. We
10
### prove that SUN recovers count-based bonuses in the limit, bounds short-horizon
11
### hitting probabilities, and provably rejects unreachable goals. We further present a
12
### practical implementation based on a novel multi-step relabeling, and accurate yet
13
### lightweight pseudocounts to avoid the overhead of classic methods. We back up all
14
### our claims with thorough benchmarks: SUN consistently outperforms state-of-the-
15
### art methods in standard and novel environments with unreachable or hard-to-reach
16
### states, irreversible transitions, obstacles, mazes, and unbounded spaces.
17 1 Introduction
18 Exploration is fundamental to reinforcement learning (RL): without effective exploration, agents
19 collect uninformative data and fail to learn. Classical dithering schemes, such as ε-greedy and entropy
20 regularization, ignore environment structure and are sample-inefficient. Provably efficient algorithms
21 [3, 23, 42] offer strong guarantees but do not scale to large state spaces. Intrinsic motivation methods
22 [8, 32, 33] require careful tuning and are non-stationary by construction: as the agent explores, the
23 intrinsic reward shifts beneath the policy trained on it, destabilizing learning [8].
24 A more recent family casts exploration as goal-conditioned RL (GCRL) [9, 27], where the agent
25 follows a goal-conditioned policy trained on a stationary goal-reaching objective. Different goal-
26 selection mechanisms lead to different exploration strategies, but most of the existing work captures
27 only half the picture. Density-based methods such as MEGA [34], Skew-Fit [35], GoalGAN [17],
28 and Hindsight Goal Generation [37] score goals by novelty, committing to rare goals that may be
29 unreachable. Conversely, methods based on distances or success probabilities [20, 38] optimize
30 reachability alone, neglecting rare but achievable goals. Neither extreme captures the right intuition:
31 a useful exploration goal is one that is novel and reachable. Figure 1 summarizes this problem.
32 In this paper, we address this gap with the following contributions.
33 (1) We present a GCRL exploration framework with a goal-selection mechanism to identify goals
34 that are both novel and reachable. Its core is the SUccessor-to-Novelty (SUN) indicator: reachability
35 is estimated via successor value functions [12], and novelty via pseudocounts. SUN is compatible
36 with any off-policy RL algorithm; in this paper, we instantiate it with DQN [28] and TD3 [18].
### Submitted to the 19th European Workshop on Reinforcement Learning (EWRL 2026). Do not distribute.

---

## Page 2

# Figure 1: Reachability or novelty are not
# enough. At every episode, the agent spawns in
# one of two isolated rooms. After exploring for
# some time, the second room has been rarely vis-
# ited due to its lower spawning rate. Heatmaps
# show the score assigned to each tile by different
# goal-selection scores when the agent is in the top-
# left corner (red boxes mark the selected goal). Novelty alone (e.g., visit counts inverse) picks tiles
# in the second room, which the agent cannot reach. Reachability alone (e.g., distance) picks the
# agent’s current tile, leading to no exploration. Only novelty and reachability combined selects the
# least-visited tile within reach. While simple, this example shows the importance of considering both
# reachability and novelty in exploration, and raises the central question of this paper: how to encode,
# learn, and combine reachability and novelty? Our SUN indicator provides principled answers.
37 (2) We introduce Last-HER(λ), a multi-step goal relabeling scheme combining Hindsight Experience
38 Replay (HER) [2] and TD(λ) traces [50], for training the successor value functions.
39 (3) We present an accurate yet lightweight pseudocount that avoids the overhead of density-based
40 methods, enabling provably efficient exploration with O(1) query cost.
41 (4) We show that SUN is the value of a goal-conditioned count-bonus reward, gives a closed-form
42 lower bound on the short-horizon hitting probability, and provably suppresses unreachable goals .
43 (5) We introduce new benchmarking environments with unreachable states and irreversible transitions
44 that directly stress-test reachability-aware exploration, and show that SUN consistently outperforms
45 state-of-the-art methods on these and standard benchmarks.
# Together, these contributions establish SUN as a principled and practical solution to the long-
# standing tension between novelty and reachability in exploration.
46
47 SUN exploration fits in the field of reward-free exploration and goal-conditioned RL, and is especially
48 close to the work of Tarbouriech et al. [46] (AdaGoal) and Diaz-Bone et al. [13] (DISCOVER) in
49 its use of successor value functions to drive exploration. AdaGoal and DISCOVER also balance
50 novelty and reachability, but estimate novelty through critic-ensemble disagreement — which is
51 computationally expensive and is prone to novelty collapse once the critics converge. SUN avoids
52 these issues, and we show that it consistently outperforms both methods.
53 The proposed multi-step goal relabeling builds on hindsight relabeling [2] and is close to recent
54 contrastive approaches [15]. Unlike them, though, it takes advantage of RL eligibility traces [50] for
55 more effective backpropagation of the positive signal. The scheme is algorithm-agnostic and can be
56 applied to any GCRL algorithm.
57 2 Problem Setting
58 Optimal exploration. A reward-free Markov Decision Process (MDP) is defined by the tuple
### ′
59
# , A, P, p0, where S is the state space, A is the action space, P(s |s, a) is the transition function,
60 and p0is the initial state distribution. The objective is to explore the state space “optimally” without
61 any task-specific reward. Two main lines of work formalize this notion of optimality differently.
62 The first line targets the state-visitation distribution: the goal is to learn a policy whose induced
63 distribution maximizes a desired criterion, typically the entropy [1, 21, 22, 25, 29, 53]. A maximum-
64 entropy state-visitation distribution corresponds to uniform coverage of the state space, and provably
65 efficient algorithms exist for this objective in tabular MDPs.
66 The second line frames exploration as goal-conditioned RL (GCRL) [26, 45, 46] and the agent learns
67 goal-conditioned policies π(a|s, g) with g ∈ G. The goal space G may be a subset of the state space
### 1
68 S, of the joint state-action space S × A, or of a learned representation thereof. A goal g is said to
69 be reachable from a reference state s0if there exists a policy π that reaches g from s0in bounded
70 expected time. The objective is then to learn policies that can reliably visit every goal reachable given
71 different reference states. This formulation directly captures the intuition that exploration should
1The goal space is environment-dependent. For example, discrete actions may highlight relevant dynamics
### (e.g., “pick” or “push” may terminate the episode) and exploration should explicitly consider them. On the
### contrary, the state alone may be sufficient if it carries all information (e.g., agent pose in control tasks).

---

## Page 3

72 focus on states the agent can actually reach, but does not specify a target visitation distribution.
73 Both objectives are principled but solving them exactly requires machinery — e.g., Frank-Wolfe
74 schemes for max-entropy [21], PAC-style algorithms for reachable coverage [45, 46] — that does
75 not scale to deep RL. Practical methods therefore approximate these objectives with greedy or local
76 heuristics: thanks to careful goal-selection mechanisms, following π(a|s, g) induces a state-visitation
77 distribution with broad and uniform state coverage. Our work follows this pragmatic line: we design
78 a goal-selection rule that, at each step, prefers goals that are both underrepresented in the agent’s
79 current visitation distribution and reachable under the current policy.
80 Successor Value Functions. In GCRL literature, reachability is often encoded with the succes-
81 sor value function (SVF) [6, 12, 15, 24], which generalizes the value function and represents the
82 cumulative γ-discounted occurrence of a goal g under a policy π, i.e.,
## hX
| ∞ | i |
|---|---|
|  |  |
### V (s , g) = Ek−t
## tγ 1{sk=g} π, P, st, (1)
### k=t
83 where γ ∈ [0, 1) and 1{sk=g} is the reward function returning 1 if sk= g and 0 otherwise. The
### π
85 standard value functions, with V (st, g) = maxaQ (st, a, g). In GCRL, once the goal is given
84 state-action analogue Q (st, at, g) is defined likewise, and both admit a Bellman recursion as in
### π
### π
86 (e.g., a desired robot pose or an environment coordinate), greedily following the SVF leads the agent
87 to it, as the value increases the fewer steps are needed to reach the goal.
88 This same mechanism extends naturally to reward-free exploration: select a goal appropriately —
89 unlike in GCRL, the goal is not given by the task — and then follow the SVF to reach it. The
90 goal-selection is what determines whether exploration is optimal: a well-designed mechanism would
91 guarantee coverage and uniformity over the goal space G ⊆ S ×A.
## With these tools in hand, our method must address three concrete subproblems. First, how to
## train SVFs efficiently in the GCRL setting. Second, how to compute a novelty signal that is both
## accurate and cheap enough to query at scale. Third, how to combine the reachability and novelty
## signals so that goal-selection mechanism prefers goals that are both novel and reachable.
92
93 3 Exploration via SUN
94 We present our answer to the three subproblems above: SUN (SUccessor-to-Novelty), a goal-selection
95 indicator that combines an SVF-based reachability signal with a novelty signal in a single score.
### SUN(g|st) ≜ Vπ(st, g) · ν(g), gt= arg max SUN(g|st), (2)
### g∈G
96
### π
97 where V (st, g) is the SVF estimating reachability of g from the current state stunder the goal-
98 conditioned policy, and ν(g) is a novelty signal. Computing the arg max in Eq. (2) is not feasible
99 in continuous or large goal spaces, so we restrict it to a finite set of candidates {Ct⊂ G} ∪ {gt−1}
100 sampled from a replay buffer [28]. This choice pairs naturally with off-policy algorithms, which
101 already maintain a buffer for training. Two key design choices distinguish SUN from prior work.
### π
103 the same goal would be selected for the whole episode (Appendix A.7). However, V is learned
102 Per-step goal selection. If V were exact, acting greedily with respect to it would be optimal and
### π
104 and approximate, and the agent may commit to unreachable goals (potentially not exploring at all).
105 Similarly, if transitions are stochastic, the agent may suddenly be in states where the previously-
106 selected goal is no longer reachable. Action-level noise (e.g., ε-greedy) could mitigate this, but would
### π
108 introduces a different source of noise: as V is updated, the score of each candidate shifts, and the
109 selected goal and action can change accordingly. This is noise that self-attenuates as V converges.
112 SUN score, making the goal-selection unstable even when V is accurate. Including g
## t−1preserves
113 stability and ensures the selected goal changes only when a candidate genuinely outranks it.
107 require tuning (too much noise could cancel the benefits of V entirely). Per-step selection instead
### π
### π
110 Note that goals from Ctare always compared against the previous goal gt−1. Without this comparison,
111 the random sampling of Ctat each step would induce spurious goal switches independent of the
### π
### 2
2Per-step goal selection also offers a flexible interface for injecting noise into exploration. The agent can
### randomly forget the current goal to escape commitments to overestimated targets, or reselect it only every few
### steps when the task requires consistent action sequences. We discuss these variants in Appendix C.

---

## Page 4

114 Multiplicative indicator. A common alternative is to combine reachability and novelty additively,
### π
115 e.g., V (s, g) + ν(g), which admits standard UCB-style confidence bounds and PAC guarantees
116 when ν is based on visit counts (see Appendix I). However, additive formulations are sensitive to
117 the relative scale of the two terms and typically require a tuning coefficient to balance them. SUN
118 multiplicative form sidesteps this scaling problem entirely: the two signals share a common “zero”
119 (an unreachable or already-saturated goal scores zero on either factor and is rejected regardless of the
120 other) and a common scale (both are non-negative and bounded). In Appendix I.6 we compare SUN
121 against an additive UCB-style variant and show that the multiplicative indicator performs better.
122 The rest of this section is organized as follows. Section 3.1 gives the theoretical properties that justify
123 the design; Section 3.2 describes SVFs training via our multi-step goal relabeling scheme; Section 3.3
124 introduces our novelty estimator ν; and Section 3.4 summarizes SUN and discusses related work.
125 3.1 Theoretical Properties
126 We informally describe four SUN properties that justify the design above; proofs are in Appendix A.
127 (a) Count-bonus equivalence. If ν(g) = 1/ngwhere ngare goal visit counts, SUN is the value
### k−t
128
# function of a reward inversely proportional to ng, i.e., E[Σk=tγ 1{sk=g}/ng]. Thus, SUN is
129
# not simply a combination of two signals, but the value of a principled count-bonus objective.
### 130 (b) Reachability guarantee. The reachability factor Vπ(s, g) controls hitting time: a high SVF value
### π
### 131 implies a high probability of reaching g within a short horizon, Prπg[τg≤ n] ≥ Vg(s, g)−γn+1,
132
# where τgis the hitting time. The horizon scales as O(log(1/V )/(1 − γ)).
### π
133 (c) Unreachable goals are suppressed. If no policy in the agent’s class can reach g, then V (s, g) = 0
134
# for all such policies, and SUN(g|s) = 0 regardless of ν(g). This is the formal counterpart of
135
# Figure 1: novelty alone selects unreachable goals, while SUN does not.
### 136 (d) Stability of per-step selection. Under exact Vπand deterministic dynamics, SUN arg max remains
137
# constant until the goal is reached: per-step selection coincides with episodic goal-selection.
138 3.2 Multi-Step Goal Relabeling via Eligibility Traces
### π
139 Training the SVF is a self-supervised process: given tuples (st, at, gt, st+1), V can be trained with
140 TD learning using the reward in Eq. (1), i.e., rt= 1{st=g}. Effective training requires tuples in which
141 the agent both reaches and fails to reach the goal, i.e., positive and negative samples. Hindsight
142 Experience Replay (HER) [2] provides these signals by relabeling the goal of a trajectory with a state
143 from the trajectory itself. The selected state fires the reward and yields explicit positive samples,
144 while other transitions are implicit negatives. States from the replay buffer can also be sampled
### 3
145 as random negative goals. HER is summarized in Figure 2(a). However, HER suffers from three
146 limitations.
147 • First, the reward 1{st=g} fires only when st= g exactly, allowing for at most one positive sample
148
# per trajectory. To make learning tractable, HER and most GCRL algorithms relax the condition to
149
# ∥st−g∥ < ϵg, accepting states “close enough” to the goal as positives. In GCRL the threshold ϵgis
150
# often given by the task; in reward-free exploration it is not, introducing a sensitive hyperparameter
151
# that must be tuned per environment.
152 • Second, the positive/negative boundary is a hard cut: two temporally adjacent states can receive
153
# opposite labels despite being nearly identical, so HER does not exploit the temporal structure it
154
# operates on.
155 • Third, the same state may appear as both a positive and an implicit negative of two temporally-close
156
# goals, producing a mixed training signal.
157 Last-HER(λ). Following Yang et al. [51], we address these issues by combining HER with TD(λ)
158 updates [50], which naturally produce a soft positive signal across time. Given a trajectory, L-HER(λ)
159 relabels the goal with the trajectory’s last state and uses the original hard indicator reward; only the
### T−t
160 last sample’s reward fires, but the TD(λ) target at each preceding state inherits a fraction (γλ)
# of
161 this terminal signal, producing a graded positive update along the entire trajectory. This eliminates
162 implicit negatives and ambiguous reward signals. The full scheme is summarized in Figure 2(c).
3Training is off-policy by construction (for both positives and negatives): the relabeled goal is not the goal
### under which the trajectory was collected.

---

## Page 5

s1s2s3s4s5s6(a) HER. Given a trajectory, we sample different goals (blue) to fire positive re-reward: 0 0 0 1 0 0 wards. Since rewards would be sparse, it ...is common practice in GCRL to relax theSVF reward 1{st=g} and have it fire also
for states that are “close enough” to the s1s2s3s4s5s6goal, i.e., ||st− g|| < εg. In GCRL, the threshold εgis given by the task, but in reward: 0 0 0 0 0 1 reward-free RL it must be carefully tuned.
imposes a hard cut between positives and s1s2s3s4s5s6negatives. This begs our first idea: to combine goal relabeling with RL eligibil-reward: 0 0 0 1 0 0 ity traces. Even though rewards are still mostly 0, the positive reward that fires for ...strajectoryt= g is backpropagated throughout thewithin the same update. This
der goal s6, even though they are close.
many independent positive samples.
s s s s s s 1 2 3 4 5 6 g˜1 (d) Explicit negatives. For all methods, ...(red) are paired with the full trajectory. NoK random off-trajectory goals g˜1, . . . , g˜K
trajectory state achieves any g˜i, so all K s1s2s3s4s5s6 g˜Kpairs serve as hard negatives.
Figure 2: Goal-relabeling strategies. In HER, one trajectory is relabeled with multiple goals, producing hard positive and negatives. HER(λ) introduces soft positives, and L-HER(λ) further removes the implicit negatives. Random negatives can be additionally sampled for all versions.
While the idea of multi-step HER has already been presented by Yang et al. [51], to the best of our knowledge the effect of implicit negatives has not been discussed in RL literature. As demonstrated in Appendix H.2, L-HER(λ) significantly improves the performance of all the baselines. First, by always selecting the trajectory’s last state as the positive goal and propagating the signal via TD(λ), each trajectory contributes a single coherent positive trace rather than many independent positive samples. This eliminates the ambiguity caused by HER’s implicit negatives, where the same state may appear as both a positive and an implicit negative of two temporally-close goals. Second, the TD(λ) target naturally interpolates between positive and negative, so neighbouring states with similar value receive similar targets — precisely the temporal smoothness HER’s hard cut discards.
163

---

## Page 6

164 Shuffling negatives. To increase the diversity of random goals, we apply a per-feature shuffling: with
165 probability p
### shuffle, each feature of a random goal is replaced independently by the same feature drawn
166 from a different sample in the buffer. This breaks correlations between feature dimensions while
167 preserving the per-feature marginal distributions, exposing the SVF to off-distribution combinations
168 that the buffer would not otherwise produce. Figure 3 illustrate our shuffling.
169 3.3 Novelty via Lightweight Pseudocounts
170 The novelty signal ν(g) in Eq. (2) can be instantiated in many ways [8, 33]. A principled choice
171 is visit counts or density estimate [5, 44], so that goals visited rarely receive a high novelty score.
172 Formally, ν(g) = 1/ng, where ngis the number of times g has been visited. However, in continuous
173 spaces ngcannot be tracked exactly and must be approximated by a pseudocount. Moreover, since
174 we query ν(g) against many candidate goals Ctat every step, the pseudocount must be lightweight to
175 compute. Standard approaches such as kernel density estimation (KDE) or neural density models do
176 not satisfy this requirement. We instead propose a pseudocount that amortizes its cost into buffer
177 insertion: counts are precomputed and stored alongside each buffer entry, making queries cheap.
178 For each entry in the replay buffer at index i, we store a count niof entries within a neighborhood of
179 radius ρ of siin standardized feature space. When a new sample is inserted, we compute its count
180 (number of neighbors plus one for itself) and increment the neighbors’ count. Figure 4 illustrates the
181 procedure; Appendix B gives the standardization scheme and the implementation details.
182 This is a fixed-radius nearest-neighbor density estimator with the cost moved from query time to
183 insertion time. The benefits of this design are the following.
184 • The radius ρ is a single hyperparameter, applied in the standardized space. Standardization makes
185
### a single scalar radius meaningful across features: without it, a separate radius would be needed for
186
# each feature dimension to account for differences in native scale.
# 187 • By incrementing the count of every neighbor of the new sample, the stored counts remain up to date
188
### across the entire buffer without ever recomputing them from scratch. At query time, the novelty of
189
# a candidate goal g is read directly from the replay buffer: ν(g) = 1/ng. This has cost O(1).
# 190 • The insertion cost is O(N·M) in the buffer size N and goal dimensionality M. Classic KDE costs
191
# O(N·M·B) per step for B candidate goals, and density-model can be even more expensive.
# In our setting, where the goal is reselected at every environment step, classic methods are
### prohibitively expensive while our pseudocount remains tractable. By moving the cost from query
# to insertion time and standardizing across features, we obtain novelty estimator that is accurate,
# lightweight, and well-suited to the goal-selection workload. We validate the accuracy of our
# pseudocount in Appendix H.3.
192
193 3.4 Summary and Related Work
194 We now have all the components to practically describe SUN exploration. At every step st, the
195 current goal gtis selected using Eq. (2), with candidates (and their respective pseudocounts) sampled
196 from the replay buffer. Then the agent acts according to at∼ π(· |st, gt), the new sample is inserted
197 into the replay buffer, and the pseudocount is updated. In parallel, the SVF and the policy are updated
198 following any off-policy algorithm, e.g., DQN [28] or TD3 [18]. Pseudocode is in Appendix C.
199 Note that this scheme can be applied to classic reward-driven RL: the goal-conditioned policy and the
200 SVF drive exploration and collect environment rewards, which are then used to train the task-specific
201 value function and policy (off-policy).
### f
### f
### f
### f
### f
### f
### f
### 1 2 3 4 1 2 3 f4
# Figure 3: Per-feature cross-sample shuf-
(1) (1)
# g a1 a2 a3 a4 g˜ c1 a2 d3 b4fling. Each goal has each of its features
# g(2)b1 b2 b3 b4 g˜(2)a1 d2 b3c4independently drawn from the same fea-ture index of a randomly chosen buffer
(3)
| c | c | c | c | (3) |
|---|---|---|---|---|
|  |  |  |  | c |
### 1 2 3 4 b 2 a3 d
# g g˜ 1 4 goal — feature fjstays in slot fj, but its
# g(4)d1 d2 d3 d4 g˜(4)d1 b2c3 a4value comes from a different sample. Col-
# ors track the source sample of each value.

---

## Page 7

## 1 2 1 2 Figure 4: Pseudocount via replay buffer neighbors. Each
## Insert ρ buffer entry stores a count of its neighbors within radius ρ (in
1 2 1 3 3standardized feature space). When a new sample is inserted,
2 3
## 3 3its count is set to the number of neighbors within ρ plus one
4 4
## (for itself), and each neighbor’s count is incremented.
202 Reachability vs. novelty. Throughout Sections 2–3, we have discussed how SUN relates to prior
203 work along several axes: reward-free exploration [21, 26, 45, 46], GCRL [2, 38], hindsight and
204 contrastive relabeling [2, 15, 55], and count- or density-based novelty [5, 8, 34, 35, 44]. Here, we
205 focus on the two methods closest in spirit to SUN: AdaGoal [46] and DISCOVER [13]. Both select
### π
### π
206 goals using an ensemble of SVFs {V1, . . . , VK}, with mean μ(s, g) and standard deviation σ(s, g).
207 Formally,
### SUN gt= arg max Vπ(st, g)/ ng
### g∈C ∪{g
| t | t−1} |
|---|---|
| t   = arg max μ ( | s 0 , g ) + |
| g ∈C 0 |  |
| t   = arg max σ ( | s 0 , g ) |
| g ∈C 0 |  |
208 The mean μ serves as a reachability signal, and the disagreement σ as an epistemic-uncertainty signal
209 (goals on which the ensemble disagrees are those the agent cannot reliably reach). AdaGoal explicitly
210 frames this as “selecting uncertain goals” and proves PAC guarantees; DISCOVER frames the same
211 quantity as “novelty” since uncertainty correlates with under-exploration.
212 Two main differences distinguish SUN from this line of work. First, AdaGoal and DISCOVER
213 select the goal once at the beginning of the episode, and resort to random exploration after the goal
214 is reached. SUN instead reselects the goal at every environment step from a freshly-resampled
215 candidate set C , with g
## t t−1always included to ensure stability. Second, SUN replaces the ensemble
216 with a lightweight pseudocount, removing the computational cost of training and querying K critics
217 in parallel and eliminating the sensitive hyperparameter β that ensemble disagreement introduces.
218 This matters because ensemble disagreement is prone to novelty collapse as the critics converge: as
219 σ → 0, the disagreement signal vanishes regardless of how often a goal has been visited, and the
220 goal-selection rule collapses to pure reachability. Pseudocounts do not suffer from this failure mode
221 — they directly track visits and remain informative. We confirm this empirically in Appendix H.3.
222 As a final remark, L-HER(λ) is itself algorithm-agnostic and can be applied to AdaGoal and DIS-
223 COVER as well. In Appendix H.2 we show it improves their performance, demonstrating that this
224 contribution extends beyond SUN itself.
225 Hindsight relabeling. HER [2] has spawned a large family of variants, most operating in the GCRL
226 setting where a task reward is given. These works differ primarily in which transitions or goals get
227 replayed: energy-based prioritization [54], curriculum-guided goal selection [16], and diversity-based
228 trajectory sampling [11], among others. None addresses the reward-free setting we consider.
229 The closest work to L-HER(λ) is MHER(λ) Yang et al. [51]: multi-step returns with exponentially
230 decayed weights to mitigate the off-policy bias. However, MHER retains standard HER’s practice
231 of sampling any trajectory state as the hindsight goal preserving implicit negatives (Figure 2(b)).
232 L-HER(λ) instead always selects the last state as goal, eliminating implicit negatives by construction
233 and addressing credit-assignment ambiguity.
234 Finally, we note that our L-HER(λ) is orthogonal to other HER variants. For example, it can be
235 combined with USHER [39], that corrects statistical hindsight bias via importance sampling.
4This is the pure-exploration variant of DISCOVER (referred to as “achievability + novelty” in the original
### paper). The full objective is designed for GCRL, and additionally combines this with a direction term toward a
### task-specific goal.
5Two notes. First, this is the deep-RL variant of AdaGoal, where σ approximates a prediction-error upper
### bound. The tabular version further constrains the arg max to the expected hitting time as a proxy for reachability,
### but the deep-RL variant does not. Second, Tarbouriech et al. apply the softmax operator rather than max. Here,
### we use max for consistency with SUN and DISCOVER, but all methods can use softmax as well.

---

## Page 8

# Figure 5: Main results. SUN uses L-HER(λ), AdaGoal and DISCOVER HER(λ). Note that full
# coverage in Gridworlds is easy, but exploring uniformly (high entropy) is not.
236 4 Experiments
237 We select benchmarks that illustrate three challenges in RL exploration: (a) unreachable states or
238 irreversible transitions, (b) large or unbounded goal spaces, and (c) hard exploration (hard-to-reach
239 states, obstacles, mazes). Full details are in Appendix D.
240 • Gridworlds (a, c): ThreeRoom is a version of Figure 1 with three larger rooms; FourRoomStuck
241
# is a variation of the classic four-room [43] in which the bottom-left room can be entered but not
242
# exited; GridMaze is a maze with a narrow passage near the agent’s spawning position. These
243
# environments are a novel contribution of this paper.
244 • Classic control (a, b, c). MountainCar is a well-known hard-exploration benchmark. In CartPole,
245
### episodes terminate easily if the pole falls, so most of the cart-position space is hard to visit (reaching
246
# the corners requires to keep the pole upright). LunarLander space is unbounded (the agent can fly
247
# as high as it can). Acrobot has unreachable joint configurations.
248 • GCRL control (b, c): PointMaze-S/H and AntMaze-S/H, ArmPush-H [7] are locomotion environ-
249
# ments of increasing difficulty with corridors and dead-ends.
250 We evaluate SUN against a random uniform exploration baseline, AdaGoal [46] and DISCOVER [13].
251 Not only are these two close to SUN, but they have achieved state-of-the-art performance and
252 outperformed algorithms like MEGA [34] and HER [2]. Full details of the setup are in Appendix F.
253 In Appendix H, we further ablate SUN components (HER, pseudocount, per-step goal-selection) to
254 evaluate their contribution to its performance. Appendix G has visitation heatmaps.
255 We evaluate two metrics over the goal space: coverage (fraction of goal space visited) and entropy
256 (entropy of goal visit counts normalized to [0, 1]). For continuous spaces, we discretize the goal space
257 into 50 bins per dimension (only for computing these evaluation metrics, not for learning). All results
258 are averaged across ten seeds, and we report means and 95% confidence intervals.

---

## Page 9

259 Results. Figure 5 shows that the two metrics are complementary: coverage measures whether a state
260 has been visited at least once, while entropy measures how uniformly visits are distributed. A method
261 can saturate coverage while still concentrating most of its samples in a small region of the state space,
262 which is precisely what happens in the gridworld environments (top row of Figure 5): all four methods
263 reach perfect coverage, yet their entropy values differ substantially, with SUN consistently the highest.
264 This gap is most pronounced on FourRoomStuck and GridMaze, where the agent starts from a fixed
265 position and must traverse bottlenecks to spread its visits evenly across the state space. Coverage
266 alone hides this difficulty; entropy reveals it. Continuous-control environments further reveal that
267 not only does SUN attain the best performance, but it is also significantly more sample-efficient:
268 SUN curves rise faster and reach and plateau only at (near-)full coverage. Furthermore, AdaGoal and
269 DISCOVER entropy curves actually decrease after a while. We attribute AdaGoal and DISCOVER
270 behaviors to critic disagreement vanishing once the baselines have explored the easily-reachable
271 region, removing the signal that drives their goal-selection, whereas SUN’s pseudocount-based
272 novelty signal remains informative (see Appendix H.4). Note that SUN performs poorly is ArmPush.
273 We are unsure about the reasons, but we suspect it may be due to the need to tune the pseudocount
274 radius hyperparameters (we did not tune any tuning at all).
275 5 Discussion
276 In this paper, we tackled the challenge of exploration in RL via goal-conditioned policies, focusing on
277 the selection of goals that are simultaneously novel and reachable. We argued that existing methods
278 cannot fully capture the tension between the two and end up neglecting one or the other. We thus
279 proposed SUN, a unified and principled goal-selection rule grounded in SVFs and counts. We further
280 introduced L-HER(λ), a multi-step goal relabeling scheme that strengthens the supervision of SVFs,
281 along with a lightweight pseudocount that scales to per-step goal-selection in continuous spaces.
282 Finally, we validated SUN on standard and new benchmarks, where it consistently outperforms
283 state-of-the-art methods AdaGoal and DISCOVER.
284 Strengths. The strength of SUN lies in its principled and modular design. Its score admits a natural
285 interpretation as the optimal value of a count-bonus exploration objective, driving the agent toward
286 rarely-visited states within reach. This is made possible by our lightweight pseudocount, which avoids
287 the overhead of classical pseudocount and density-based approaches while preserving their accuracy.
288 Our results against AdaGoal and DISCOVER, which instead rely on critic-ensemble disagreement
289 and are therefore prone to novelty collapse, confirm this. Moreover, L-HER(λ) relabeling significantly
290 improves SVF training, as evidenced by the gains it also yields for AdaGoal and DISCOVER.
291 Limitations and Future Work. First, although principled, count-based novelty is ineffective in large
292 spaces, such as image observations, regardless of their pseudocount approximations. Since most
293 observed states are effectively unique, counts become nearly uniform across the buffer and do not
294 provide a useful novelty signal. In such regimes, exploration likely requires richer signals, such as
295 learned curiosity [32, 36] or mutual information between trajectories and learned latents [14, 40].
296 Combining these signals with SUN’s reachability factor is a promising direction.
297 Second, SUN selects the goal greedily based on the SUN score of the candidate alone, without
298 accounting for the states traversed to reach it. A longer path through many novel states may be
299 preferable to a shorter path to a marginally rarer goal — a distinction the current formulation cannot
300 make. Extending SUN indicator to a cumulative form would more directly target a maximum-entropy
301 state distribution [21, 29]. We see this as a natural next step.
302 Finally, SUN draws candidate goals from the replay buffer, which means it can only propose goals it
303 has already encountered. While the negative shuffling in Section 3.2 provides some signal for out-of-
304 distribution goals during training, the selection rule itself remains tied to the buffer manifold and
305 cannot extrapolate to genuinely unseen states. A natural way to relax this is to exploit the geometry
306 of the goal space. Recent work on temporal distances and quasimetric value functions [30, 49]
307 learns goal-space metrics that generalize beyond observed pairs, Laplacian-style representations [41]
308 provide a similar latent geometry, and studies on out-of-distribution generalization in goal-conditioned
309 RL [52] characterize design choices that enable extrapolation to unseen goals. Combining SUN’s
310 reachability factor with such learned metrics or generalization-aware training — sampling goals
311 as points in a continuous metric space rather than from the buffer — is a promising direction for
312 unblocking the buffer-manifold limitation.

---

## Page 10

313 References
314 [1] J. Adamczyk, A. Kamoski, and R. V. Kulkarni. Maximum entropy exploration without the
315
# rollouts. arXiv:2603.12325, 2026.
316 [2] M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin,
317
# O. Pieter Abbeel, and W. Zaremba. Hindsight experience replay. In Advances in Neural
318
# Information Processing Systems (NeurIPS), 2017.
# 319 [3] P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem.
320
# Machine Learning, 47(2-3):235–256, 2002.
# 321 [4] M. G. Azar, I. Osband, and R. Munos. Minimax regret bounds for reinforcement learning. In
322
# International Conference on Machine Learning (ICML), 2017.
323 [5] M. G. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos. Unifying
324
### count-based exploration and intrinsic motivation. In Advances in Neural Information Processing
325
# Systems (NeurIPS), 2016.
326 [6] L. Blier, C. Tallec, and Y. Ollivier. Learning successor states and goal-dependent values: A
327
# mathematical viewpoint. arXiv:2101.07123, 2021.
328 [7] M. Bortkiewicz, W. Pałucki, V. Myers, T. Dziarmaga, T. Arczewski, Ł. Kuci´nski, and B. Ey-
329
# senbach. Accelerating goal-conditioned reinforcement learning algorithms and research. In
330
# International Conference on Learning Representations (ICLR), 2025.
331 [8] Y. Burda, H. Edwards, A. Storkey, and O. Klimov. Exploration by random network distillation.
332
# In International Conference on Learning Representations (ICLR), 2019.
# 333 [9] C. Colas, T. Karch, O. Sigaud, and P.-Y. Oudeyer. Autotelic agents with intrinsically motivated
334
# goal-conditioned reinforcement learning: a short survey. Journal of Artificial Intelligence
335
# Research (JAIR), 74:1159–1199, 2022.
336 [10] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes. Supervised learning of universal
337
# sentence representations from natural language inference data. In Conference on Empirical
338
# Methods in Natural Language Processing (EMNLP), 2017.
339 [11] T. Dai, H. Liu, K. Arulkumaran, G. Ren, and A. A. Bharath. Diversity-based trajectory and
340
# goal selection with hindsight experience replay. In Pacific Rim International Conference on
341
# Artificial Intelligence (PRICAI), 2021.
342 [12] P. Dayan. Improving generalization for temporal difference learning: The successor representa-
343
# tion. Neural Computation, 5(4):613–624, 1993.
344 [13] L. Diaz-Bone, M. Bagatella, J. Hübotter, and A. Krause. DISCOVER: Automated curricula for
345
# sparse-reward reinforcement learning. arxiv:2505.19850, 2025.
# 346 [14] B. Eysenbach, A. Gupta, J. Ibarz, and S. Levine. Diversity is all you need: Learning skills
347
# without a reward function. In International Conference on Learning Representations (ICLR),
348
# 2019.
349 [15] B. Eysenbach, T. Zhang, S. Levine, and R. R. Salakhutdinov. Contrastive learning as goal-
350
# conditioned reinforcement learning. In Advances in Neural Information Processing Systems
351
# (NeurIPS), 2022.
# 352 [16] M. Fang, T. Zhou, Y. Du, L. Han, and Z. Zhang. Curriculum-guided hindsight experience replay.
353
# In Advances in Neural Information Processing Systems (NeurIPS), 2019.
354 [17] C. Florensa, D. Held, X. Geng, and P. Abbeel. Automatic goal generation for reinforcement
355
# learning agents. In International Conference on Machine Learning (ICML), 2018.
356 [18] S. Fujimoto, H. van Hoof, and D. Meger. Addressing function approximation error in Actor-
357
# Critic methods. In International Conference on Machine Learning (ICML), 2018.
358
# [19] I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In
359
# International Conference on Machine Learning (ICML), 2013.

---

## Page 11

360 [20] K. Hartikainen, X. Geng, T. Haarnoja, and S. Levine. Dynamical distance learning for semi-
361
# supervised and unsupervised skill discovery. In International Conference on Learning Repre-
362
# sentations (ICLR), 2016.
363
# [21] E. Hazan, S. Kakade, K. Singh, and A. Van Soest. Provably efficient maximum entropy
364
# exploration. In International Conference on Machine Learning (ICML), 2019.
365 [22] A. K. Jain, L. Lehnert, I. Rish, and G. Berseth. Maximum state entropy exploration using
366
### predecessor and successor representations. Advances in Neural Information Processing Systems
367
# (NeurIPS), 2023.
# 368 [23] T. Jaksch, R. Ortner, and P. Auer. Near-optimal regret bounds for reinforcement learning.
369
# Journal of Machine Learning Research (JMLR), 11:1563–1600, 2010.
# 370 [24] T. D. Kulkarni, A. Saeedi, S. Gautam, and S. J. Gershman. Deep successor reinforcement
371
# learning. arXiv:1606.02396, 2016.
372 [25] L. Lee, B. Eysenbach, E. Parisotto, E. Xing, S. Levine, and R. Salakhutdinov. Efficient
373
# exploration via state marginal matching. arXiv:1906.05274, 2019.
374 [26] S. H. Lim and P. Auer. Autonomous exploration for navigating in MDPs. In Conference on
375
# Learning Theory, 2012.
# 376 [27] M. Liu, M. Zhu, and W. Zhang. Goal-conditioned reinforcement learning: Problems and
377
# solutions. arXiv:2201.08299, 2022.
378 [28] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves,
379
# M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep rein-
380
# forcement learning. Nature, 518(7540):529–533, 2015.
381 [29] M. Mutti, L. Pratissoli, and M. Restelli. Task-agnostic exploration via policy gradient of a
382
# non-parametric state entropy estimate. In AAAI Conference on Artificial Intelligence, 2021.
# 383 [30] V. Myers, C. Zheng, A. Dragan, S. Levine, and B. Eysenbach. Learning temporal distances: Con-
384
# trastive successor features can provide a metric structure for decision-making. In International
385
# Conference on Machine Learning (ICML), 2024.
386
# [31] Y. Pan, K. Banman, and M. White. Fuzzy tiling activations: A simple approach to learning
387
### sparse representations online. In International Conference on Learning Representations (ICLR),
388
# 2021.
389
# [32] S. Parisi, V. Dean, D. Pathak, and A. Gupta. Interesting object, curious agent: Learning
390
# task-agnostic exploration. In Advances in Neural Information Processing Systems (NeurIPS),
391
# 2021.
392 [33] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell. Curiosity-driven exploration by self-
393
# supervised prediction. In International Conference on Machine Learning (ICML), 2017.
394 [34] S. Pitis, H. Chan, S. Zhao, B. Stadie, and J. Ba. Maximum entropy gain exploration for long
395
# horizon multi-goal reinforcement learning. In International Conference on Machine Learning
396
# (ICML), 2020.
397 [35] V. Pong, M. Dalal, S. Lin, A. Nair, S. Bahl, and S. Levine. Skew-Fit: State-covering self-
398
# supervised reinforcement learning. In International Conference on Machine Learning (ICML),
399
# 2020.
400 [36] R. Raileanu and T. Rocktäschel. RIDE: Rewarding Impact-Driven Exploration for Procedurally-
401
# Generated Environments. In International Conference on Learning Representations (ICLR),
402
# 2020.
403 [37] Z. Ren, K. Dong, Y. Zhou, Q. Liu, and J. Peng. Exploration via hindsight goal generation. In
404
# Advances in Neural Information Processing Systems (NeurIPS), 2019.
# 405 [38] T. Schaul, D. Horgan, K. Gregor, and D. Silver. Universal value function approximators. In
406
# International Conference on Machine learning (ICML), 2015.
407
### [39] L. Schramm, Y. Deng, E. Granados, and A. Boularias. USHER: Unbiased sampling for hindsight
408
# experience replay. In Conference on Robot Learning (CoRL), 2022.

---

## Page 12

409 [40] A. Sharma, S. Gu, S. Levine, V. Kumar, and K. Hausman. Dynamics-aware unsupervised
410
# discovery of skills. In International Conference on Learning Representations (ICLR), 2020.
# 411 [41] D. Shehmar, M. Schlegel, M. E. Taylor, and M. C. Machado. Laplacian representations for
412
# decision-time planning. arXiv:2602.05031, 2026.
413
# [42] A. L. Strehl and M. L. Littman. An analysis of model-based interval estimation for Markov
414
# decision processes. Journal of Computer and System Sciences (JCSS), 74(8):1309–1331, 2008.
# 415 [43] R. S. Sutton, D. Precup, and S. Singh. Between MDPs and semi-MDPs: A framework for
416
### temporal abstraction in reinforcement learning. Artificial intelligence, 112(1-2):181–211, 1999.
417 [44] H. Tang, R. Houthooft, D. Foote, A. Stooke, O. X. Chen, Y. Duan, J. Schulman, F. DeTurck, and
418
# P. Abbeel. #Exploration: A study of count-based exploration for deep reinforcement learning.
419
# In Advances in Neural Information Processing Systems (NeurIPS), 2017.
# 420 [45] J. Tarbouriech, M. Pirotta, M. Valko, and A. Lazaric. Improved sample complexity for in-
421
# cremental autonomous exploration in MDPs. In Advances in Neural Information Processing
422
# Systems (NeurIPS), 2020.
423 [46] J. Tarbouriech, O. D. Domingues, P. Ménard, M. Pirotta, M. Valko, and A. Lazaric. Adaptive
424
# multi-goal exploration. In International Conference on Artificial Intelligence and Statistics
425
# (AISTATS), 2022.
426
# [47] M. Towers, A. Kwiatkowski, J. Terry, J. U. Balis, G. De Cola, T. Deleu, M. Goulão,
427
# A. Kallinteris, M. Krimmel, A. KG, et al. Gymnasium: A standard interface for reinforcement
428
# learning environments. arXiv:2407.17032, 2024.
429 [48] H. van Hasselt. Double Q-learning. In Advances in Neural Information Processing Systems
430
# (NeurIPS), 2010.
# 431 [49] T. Wang, A. Torralba, P. Isola, and A. Zhang. Optimal goal-reaching reinforcement learning via
432
# quasimetric learning. In International Conference on Machine Learning (ICML), 2023.
433 [50] C. J. Watkins and P. Dayan. Q-learning. Machine Learning, 8(3-4):279–292, 1992.
434
# [51] R. Yang, J. Lyu, J. Yan, F. Luo, D. Luo, L. Li, and X. Li. Bias-reduced multi-step hindsight
435
# experience replay for efficient multi-goal reinforcement learning. arXiv:2102.12962, 2021.
436 [52] R. Yang, Y. Lin, X. Ma, H. Hu, C. Zhang, and T. Zhang. What is essential for unseen goal
437
### generalization of offline goal-conditioned rl? In International Conference on Machine Learning
438
# (ICML), 2023.
439 [53] C. Zhang, Y. Cai, L. Huang, and J. Li. Exploration by maximizing Renyi entropy for reward-free
440
# RL framework. In AAAI Conference on Artificial Intelligence, 2021.
441 [54] R. Zhao and V. Tresp. Energy-based hindsight experience prioritization. In Conference on
442
# Robot Learning (CoRL), 2018.
443 [55] C. Zheng, R. Salakhutdinov, and B. Eysenbach. Contrastive difference predictive coding. In
444
# International Conference on Learning Representations (ICLR), 2024.

---

## Page 13

445 Appendices
446 A Theoretical Properties 14
447
# A.1 Setup and notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
448
# A.2 Count-Bonus Equivalence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
449
# A.3 Hitting-Probability Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
450
# A.4 Rejection of Unreachable Goals . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
451
# A.5 Log-space Decomposition and Connections to Prior Work . . . . . . . . . . . . . . 16
452
# A.6 Extension to Stochastic Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . 16
453
# A.7 Per-Step Goal-Selection: Theoretical Consistency and Practical Motivation . . . . 16
454 B Pseudocount Radius and Standardization 18
455 C Practical Notes 18
456 D Environment Details 19
457 E Source Code and Compute Details 20
458 F Training Hyperparameters 20
459
# F.1 Gridworlds and Continuous Control . . . . . . . . . . . . . . . . . . . . . . . . . 20
460
# F.2 GCRL Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
461 G Heatmaps 23
462 H Ablations 24
463
# H.1 Ablation: Per-Step vs. Episodic Goal Selection . . . . . . . . . . . . . . . . . . . 24
464
# H.2 Ablation: HER vs. L-HER(λ) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
465
# H.3 Ablation: Pseudocounts vs. True Counts . . . . . . . . . . . . . . . . . . . . . . . 26
466
# H.4 AdaGoal and DISCOVER Novelty Collapse . . . . . . . . . . . . . . . . . . . . . 26
467 I SUN-UCB: PAC and Empirical Analysis 27
468
# I.1 Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
469
# I.2 SUN-UCB algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
470
# I.3 PAC bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
471
# I.4 Proof of Theorem 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
472
# I.5 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
473
# I.6 Additive vs. multiplicative SUN . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

---

## Page 14

474 A Theoretical Properties
475 This section analyzes the SUN selection rule under an oracle: the SVF V (s, g) and the count ngare
### π
476 exact, the replay buffer is frozen, and the goal-conditioned policy π acts to reach g. The analysis is
477 structural — it characterizes what SUN does given perfect estimates, not what it learns from samples.
478 Within this scope, we establish three properties (Sections A.2–A.4) that together formalize how SUN
479 balances novelty and reachability, and we draw connections to existing methods through a log-space
480 decomposition (Section A.5).
481 A.1 Setup and notation
482 Let μ(g) denote the replay marginal over goals — the empirical distribution of states stored in the
483 buffer. The pseudocount ngfrom Section 3.3 is a finite-sample estimate of μ(g) up to a normalization
484 constant; under the oracle, we treat ngas exact and μ(g) > 0 for every candidate. We define the
485 log-rarity potential
## Φ(g) ≜ log μ(g), (3)
486 which is monotone in ng: small ngcorresponds to small μ(g), hence small (very negative) Φ(g). The
487 SUN novelty score ν(g) = 1/ngin Section 3 therefore corresponds to −Φ(g) in log-space, up to a
488 constant.
489 Throughout the analysis, π denotes the goal-conditioned policy targeting g, and τg≜ inf{t ≥ 0 :
## 490 st= g} is its (random) hitting time of g from a starting state s. The SVF (Eq. 1) under π admits the
491 equivalent form
### π
### V (s, g) = Eπ[γτg1{τg<∞}], (4)
492 when g is terminal (i.e., once reached, the episode is considered ended). This identity will be used
493 repeatedly.
494 Assumption 1 (Oracle setting). From now on, we assume the following conditions are true.
495 (C1) The state space S is finite.
496 (C2) Transitions are deterministic.
### π
497 (C3) The replay marginal μ, the SVF V , and the count ngare frozen during the analysis.
## 498 (C4) Under π, the trajectory reaches g in kπ(s, g) ∈ {1, 2, . . . } steps if the goal is reachable, and
499
## the episode is considered ended at the first hit (i.e., goal-reaching transitions are terminal). If
500
## the goal is unreachable goal, kπ(s, g) = ∞.
501 Assumption 1 mirrors the structural setup used in the autonomous-exploration literature [26, 45, 46].
502 Section A.6 relaxes (C2)–(C3) to stochastic dynamics with almost-sure hitting.
503 A.2 Count-Bonus Equivalence
504 The first result links SUN to the count-based intrinsic motivation literature [5]: SUN is exactly the
505 value of a goal-specific count-bonus reward.
506 Theorem 1 (Count-bonus equivalence). Under Assumption 1, for any goal g and any policy π, define
### g
507 the goal-specific reward rt≜ 1{st=g}/ng, where ngis the (frozen) count from Assumption 1(C4).
508 Then
### EπXγtrtgs0= s = Vπn(s, gg) = SUN(g|s). (5)
### t≥0

---

## Page 15

509 Proof. By Assumption 1(C4), ngis frozen during the analysis and therefore independent of the 510 trajectory. It thus factors out of the expectation:
# X
# t gXt1{st=g}
# Eπγ rts0= s = Eπγ ngs0= s (6)
t≥0 t≥0
1 Xt
# = Eπγ 1{st=g} s0= s (7)
# ng
Vπ(s, g)
# =
# = SUN(g|s), (8)
# ng
511 where the last equality uses the definition of the SVF (Eq. 1) and the SUN score (Eq. 2).
516 Remark 1 (Frozen vs. online counts). The equivalence holds when ngis treated as fixed during 517 the trajectory — the standard frozen-replay setting (C4). When counts are updated online (as in 518 our practical algorithm and in the standard count-bonus literature), the surrogate reward becomes 519 non-stationary and the equivalence holds only approximately, with the size of the gap controlled by 520 how much ngchanges over the trajectory.
521 A.3 Hitting-Probability Bound
π
π 525 reaches g, it stays. In this setting, V (s, g) = E[γτg1{τg<∞}] ≤ 1 (Section A.1), and a high SVF
526 guarantees a high hitting probability over a short horizon.
π Pr[τg≤ n] ≥ V (s, g) − γn+1. (9) π n+1 π π
529 horizon is n = O(log(1/V )/(1 − γ)).
528 In particular, choosing n such that γ
# ≤ V (s, g)/2 gives Prπ[τg≤ n] ≥ V (s, g)/2. This
π
π π 530 Proof. Under Assumption (C4), V (s, g) admits the equivalent form V (s, g) = E[γτg1{τg<∞}]
π V (s, g) = E[γτg1{τg<∞}] (10)
τ = E[γg1{τg≤n}] + E[γτg1{n<τg<∞}]. (11)
533 For the first term, γτg≤ 1 when τg≤ n, so
E[γτg1{τg≤n}] ≤ E[1{τg≤n}] = Pr[τg≤ n]. (12) π τ 534 For the second term, γg≤ γn+1when τg> n, so
τ n+1 E[γg1{n<τg<∞}] ≤ γ Pr[τg> n] ≤ γn+1. (13) π π n+1
535 Combining the two bounds gives V (s, g) ≤ Pr [τ ≤ n] + γ
| π g | , and rearranging yields |
|---|---|
| π   |  |
|   n ]   ≥   V   ( s, g )   −   | γ n +1 .   (14) |
|  |  |
|  |  |
|  |  |
|  | π   |
| /γ )   ≥   1   −   γ   for   γ   ∈ |   (0 ,   1) . |
|   and inversely in   | 1   −   γ : high-SVF goals are hit quickly |
536 For the horizon claim, set γ
# ≤ V /2 and solve: n + 1 ≥ log(2/V )/log(1/γ), which gives
π
π
539 with high probability, while low-SVF goals require long horizons. This is the formal version of the 540 intuition that the SVF is a reachability signal.

---

## Page 16

541 A.4 Rejection of Unreachable Goals
542 The third result is the dual of Theorem 2: goals that no policy can reach receive zero score, so SUN
543 never selects them.
544 Theorem 3 (Unreachability rejection). Let Π be any class of policies. If g is unreachable from s
545 under every π ∈ Π — that is, Prπ[τg< ∞] = 0 for all π ∈ Π — then
### sup Vπ(s, g) = 0, SUN(g|s) = 0, (15)
### π∈Π
546 regardless of ng.
# 547 Proof. For any π ∈ Π, the unreachability hypothesis Prπ[τg< ∞] = 0 means the reward 1{τg<∞}
548 is zero with probability one. Hence
### π
### V (s, g) = Eπ[γτg1{τg<∞}] = 0, (16)
# 549 and SUN(g|s) = 0/ng= 0.
550 This property fails for novelty-only scores (1/ngalone), which assign maximal value to never-visited
551 unreachable goals — exactly the failure mode shown in Figure 1. SUN is therefore guaranteed to
552 ignore goals that the agent cannot reach, regardless of how rare they are.
553 A.5 Log-space Decomposition and Connections to Prior Work
554 A useful consequence of the count-bonus form is that log SUN decomposes additively:
### log SUN(g|s) = log Vπ(s, g) − Φ(g). (17)
### π
### 555 Under Assumption 1, V (s, g) = γkπ(s,g)−1for reachable g, so the score takes the closed form
#  
# 
# log SUN(g|s) = − kπ(s, g) − 1 λ − Φ(g), λ ≜ −log γ > 0. (18)
556 SUN therefore takes the form of a soft Lagrangian: −Φ(g) is the rarity reward, (kπ− 1)λ is the
557 discounted distance cost, and λ is derived from the discount γ rather than tuned. This is structurally
558 similar to AdaGoal [46], which solves a hard-constrained version with a user-specified radius. The
559 two are not equivalent — AdaGoal uses sample-variance epistemic uncertainty as its novelty signal,
560 while SUN uses distributional rarity — but both balance the same two ingredients.
561 The log-space form also yields a state-dependent admissibility condition: a goal g is preferred over
562 staying at s if and only if
# Φ(s) − Φ(g)
# k (s, g) < 1 +
| π | . (19) |
|---|---|
|  | λ   |
|  | s ) ) is never preferred, and the maximum |
|  | Φ( g ) .   This is a tighter analogue of the |
| "neither too easy nor too hard" intuition behind AdaGoal and DISCOVER [ |   13   ], with the trade-off |
|  |  |
567 A.6 Extension to Stochastic Dynamics
568 Theorems 1–3 hold beyond deterministic dynamics. Replacing (C2)–(C3) with stochastic transitions
### π
569 and the assumption that τgis almost-surely finite for reachable g, the equivalent SVF form V (s, g) =
### π
### 570 E[γτg] continues to hold, and all three theorems transfer verbatim. The closed-form V = γkπ−1
571 in Eq. 18 no longer holds in general — kπbecomes a random hitting time — but the log-space
572 decomposition (Eq. 17) is unchanged.
573 A.7 Per-Step Goal-Selection: Theoretical Consistency and Practical Motivation
### π
574 A natural concern with the SUN formulation is that the SVF V (s, g) is defined under a policy π
575 that pursues g for the entire trajectory, while SUN reselects the goal at every step. We show that this
576 concern dissolves in two ways: in the deterministic, oracle setting, per-step reselection coincides
577 with fixed-goal commitment (Proposition 1), so the two semantics agree exactly. In stochastic or
578 approximate settings, they diverge — but the divergence is exactly what per-step reselection is
579 designed to handle.

---

## Page 17

π ∗
582 under πgt∗ to reach st+1, then gtremains the SUN argmax at st+1:
g∈G
# ng
583 By induction, the goal remains fixed throughout the episode until gtis reached.
∗
585 reduces its hitting time by exactly one: kπ(st+1, gt) = kgt∗ − 1. For any other goal g, the triangle
∗
gt∗= arg max Vπ(st+1, g). (20)
∗
∗
π 587 kπ(st+1, g) ≥ kg− 1. Applying V (s, g) = γkπ(s,g)−1:
π Vπ(st+1, gt∗) = γ−1V (st, gt∗),
∗
| ngt | ngt∗ |
|---|---|
| π   |  |
| V   ( s t +1 , g ) ≤ | − 1   V   π   ( s t , g ) ∀ |
| n |  |
| g | n g |
|   n g   change between steps |   t   and   t   + 1 |
| during the analysis. (Even outside this assumption, the count |   n g t ∗   |
|  |  |
|  |   g t   at   s t |
∗
# 592 Theorem 3 ensures unreachable goals remain excluded.
π
594 produce the same trajectory. The fixed-goal semantics of V and the per-step semantics of SUN are 595 therefore consistent.
596 Stochastic dynamics. Under stochastic transitions, the next state s
# t+1is random. The hitting-time
∗ ∗
598 from gt. This is precisely where per-step reselection becomes valuable: a fixed-goal policy would 599 continue pursuing gteven after a stochastic transition that makes another goal more attractive, while
597 identity k (s , g ) = k (s , g ) − 1 holds only in expectation, and the argmax at s
# π t+1 t π t t t+1may differ
∗ ∗
600 SUN re-optimizes given the realized next-state. Per-step reselection is therefore not a deviation from 601 the theoretical analysis but its natural extension to stochastic settings.
SUN per-step goal-selection is therefore the right strategy in both regimes. When Vπis
# inaccurate (as in the early stages of training), it injects the noise needed to avoid committing
deterministically to the wrong goal (or acting wrongly toward the right goal). When Vπis
# accurate (as it naturally becomes during learning), Proposition 1 shows that it coincides with
# fixed-goal commitment and is therefore optimal.
602

---

## Page 18

603 B Pseudocount Radius and Standardization
604 The pseudocount of Section 3.3 relies on a single scalar radius ρ in standardized feature space. We
605 elaborate on the standardization step here.
606 Standardization. Let σmdenote the standard deviation of feature m across all currently stored buffer
607 entries. Naively dividing each feature by its σmcenters all features at unit variance, so that a single
608 radius ρ has the same meaning across features. However, this fails when a feature is nearly constant:
609 a small σmin the denominator amplifies tiny variations in that feature, so two points that are almost
610 identical in dimension m end up far apart in the standardized space.
611 To prevent this amplification, we floor σmat the median standard deviation across features:
′ σ˜m= max(σm, medianm(σm′ )), (21)
612 and standardize using σ˜mrather than σm. As a final safeguard, if σ˜m= 0 (the feature is exactly
613 constant across the buffer), we fall back to σ˜m= 1. The squared distance used by the pseudocount
614 then becomes
### ∥s − s′∥2/σ˜= Xm(smσ−˜m2s′m)2. (22)
615 Note that the standard deviation {σm} is recomputed on each insertion from the current buffer. In
616 our experiments, this added negligible overhead since the dominant cost is the pairwise distance
617 computation. For very large buffers, σmcan be recomputed less frequently (e.g., every n insertions)
618 without affecting the reported results.
619 C Practical Notes
## Algorithm 1: SUN Exploration Scheme
### 1 for each environment step do
### (i)
### 2 sample candidate goals C = {g }
## t i=1...Band their pseudocounts from D
### 3 compute n
## gt−1against D (Section 3.3)
### 4 select goal: g ← arg max
## t g∈Ct∪{gt−1}SUN(g|st), where ν(g) = 1/ng
### 5 explore: a ∼ π (· | s , g ) and s
## t θ t t t+1∼ P(· | st, at)
### 6 insert (st, at, st+1) and update pseudocounts in D (Section 3.3)
### 7 sample trajectories of length T batch from D and do multi-step relabeling (Section 3.2)
### π
### π
### 9 update π to act greedily with respect to V or Q
### 8 update V or Q on the relabeled batch with any off-policy algorithm
### π
### π
620 Algorithm 1 summarizes SUN exploration. The pseudocode is intentionally generic, as training can
621 be carried out with any off-policy algorithm. For example, DQN would update at every step, whereas
622 TD3 would update at the end of each episode. Finally, a few practical notes.
623 Action-level noise. Ideally, the exploration policy would act greedily with respect to the SVF.
### π
624 However, since V is inaccurate early in training, it may be appropriate to inject a small amount of
625 noise into π(a|st, g). More details are in Appendix F.
626 Replay buffer eviction. If the replay buffer is large enough (as in our experiments), no data eviction
627 occurs, so the stored counts exactly reflect the current buffer at all times. If eviction were enabled (e.g.
628 in a fixed-size FIFO buffer), the principled option is to symmetrically decrement the counts of the
### 6
629 evicted sample’s neighbors, preserving exactness at the same per-step cost as insertion. A simpler
630 approximation is to leave counts as-is, but this introduces an upward bias on the evicted sample’s
631 neighbors — their stored counts no longer reflect the current buffer.
632 SVF reward. In RL literature, the reward in Eq. (1) is sometimes replaced by alternatives that
633 target the same quantity through different reward shapings. For example, returning −1 until siis
634 reached and 0 thereafter recovers a negative-distance interpretation [2, 38]. However, those rewards
635 performed worse in our experiments.
6This is why n
### is computed at every step in Algorithm 1, and not read from the buffer (g
### gt−1 t−1 may have
### been evicted).

---

## Page 19

636 D Environment Details
# Figure 6: Gridworlds introduced
### in this paper. Black tiles are empty,
### gray tiles are walls; cyan tiles mark
# starting positions and green tiles
# mark terminal positions (episode
# ends). In Four-Room-Stuck, red
# arrows denote one-way tiles.
### (a) Three-Room (b) Four-Room-Stuck (c) Grid-Maze
637 Gridworlds. Novel environments shown in Figure 6. The observation is a one-hot encoding of the 638 agent’s tile. The goal space is S × A, i.e., we exploration should do every actions in all states.
639 • Three-Room: three rooms separated by walls. The agent spawns non-uniformly across three 640
# positions (cyan tiles): 47.5% chance in the first room, 47.5% in the second, 5% in the third. There
641
# are four actions: left, right, up, down. Episode horizon: 100 steps.
642 • Four-Room-Stuck: a variation of the classic four-room in which the bottom-left room can be 643
# entered but not exited (red arrows mark irreversible-transition tiles). There are four actions: left,
644
# right, up, down. Episode horizon: 200 steps. Episodes also end if the agent reaches the green tile.
645 • Grid-Maze: maze with narrow passages. There are nine actions: left, right, up, down, up-left, 646
# down-left, up-right, down-right, stay. Exploration is challenging due to the narrow passages near
647
### the agent’s starting position, that can be traversed only with “up-right”. Episode horizon: 200 steps.
648
# Episodes end if the agent does “stay” in the green tile.
# ˜
# 649 Classic control. Open-sourced classic RL benchmarks [47]. The goal space is S × A, where S˜ is a
7
650 subset of the state space that depends on the environment.
◦ ◦
653
# [−12 , 12 ] respectively (the episode terminates if the agent moves out of these bounds).
654 • Lunar Lander: the (x, y) coordinate of the lander. The x coordinate is bounded in [−1, 1] (the 655
# episode terminates if the lander moves out of bounds); y is unbounded.
656 • Mountain Car: the position x ∈ [−1.2, 0.6] and the velocity x˙ ∈ [−0.07, 0.07].
657 GCRL Control. Open-sourced GCRL benchmarks [7]. The goal space is the (x, y) planar position 658 of the agent, bounded by the maze layout (Figure 7). The action is not part of the goal space.
# Figure 7: GCRL environments.
# PointMaze-S (a) is a continu-
### ous point-navigation task with ve-
# locity controls implemented in
# the MuJoCo physics engine; the
# figure shows the maze layout,
# and the agent’s starting position
# (a) PointMaze-S (b) AntMaze-S (c) AntMaze-H (green dot).
7Gymnasium [47] documented bounds are just “advisory” and do not correspond to the actual bounds of ◦ the space the agent can visit. For example, CartPole documented bounds are [−4.8, 4.8] and [−24 , 24◦], but ◦ episodes end if the cart moves outside [−2.4, 2.4] or if the pole falls out of [−12 , 12◦]. Similarly, LunarLander
### episodes terminate if x is outside of [−1, 1], even though its documented bounds are [−2.5, 2.5].

---

## Page 20

659 E Source Code and Compute Details
660 We ran experiments on SLURM-based clusters, always saving all data and statistics available (e.g., 661 visit and goal counts maps). Wall-clock time per run varies with environment dimensionality, training 662 steps, and algorithm. Runs were parallelized whenever possible. 663 For gridworlds and classic control, code is in PyTorch and ran on Intel E5-2683 v4 Broad-664 well @ 2.1GHz CPUs. Fastest runs took 10 minutes, longest 10 hours (all on one CPU core). 665 For GCRL control, code is in Jax and ran on a mix of NVIDIA V100 (32 GB) and A100 (80 GB). 666 Operations are JIT-compiled and execute end-to-end on a single GPU per run. Fastest runs took 30 667 minutes, longest 7 hours.
668 Source code will be released.
669 F Training Hyperparameters
## Hyperparameters have not been tuned. For GCRL control, we used the official implementation
## of DISCOVER (TD3). For gridworlds and continuous control, we implemented a simple version
## of DQN and used common hyperparameters without any tuning.
670 671 Table 1 shows the number of warm-up steps at the beginning of learning, during which the agent acts 672 randomly to populate the replay buffer. 673 The replay buffer stores all samples collected, i.e., no data is ever evicted. 674 When sampling trajectories of length T, we draw one sample form the buffer and append all the 675 following T − 1 samples. These sequences may have samples from different trajectories, but are kept 676 separate thanks to truncation flags.
## Table 1: Warm-up steps per environment.
Three-Room Four-Room-Stuck Maze Acrobot Cart Pole L. Lander M. Car PointMaze-S AntMaze-S AntMaze-H 5K 50K 50K 10K 10K 20K 20K 1K 1K 1K
## Table 2: Hyperparameters used in our experiments.
Critic final activation softplus Loss Huber HER future-relabel prob. 0.7 Gradient norm clipped at 1.0
677 F.1 Gridworlds and Continuous Control
π
678 Actions are discrete, thus we learn Q (s, a, gs, ga). Note that the goal is composed of a state 679 component gs(environment-specific, see Section D), and an action component ga. That is, we 680 explicitly learn SVFs whose goal is to perform specific actions in specific states. Consequently, the 681 Q-network takes state and the goal-state as input, and outputs the action-value for every goal-action.

---

## Page 21

### π
683 ε-greedily with respect to Q , with ε = 0.1. Action-level noise is needed with neural-network ap-
682 We use Double-DQN to train all Q [48]. Once the goal is selected greedily, the policy explores
### π
684 proximators, and is aligned with AdaGoal and DISCOVER implementation (with built-in exploration
685 noise via Gaussian perturbations on the actor’s output). Note that this does not go against what
686 claimed in Section 3: the noise introduced is minimal, constant, and not tuned (unlike classic methods
687 where it is either learned or carefully decayed).
688 AdaGoal and DISCOVER use an ensemble of 4 critics, each trained with different random batches.
689 For acting, the minimum value of each action across all critics is used. To mitigate overestimation
### π
690 bias, all algorithms clip Q to 1, its maximum true value: when the agent reaches a goal, the reward
691 in Eq. (1) is 1 and the transition is terminal.
692 SUN and the versions of AdaGoal and DISCOVER with multi-step goal relabeling update the critics
693 with TD(λ) using Watkin’s cutting traces [50]. In practice, we relax the strict arg max used by
694 Watkins’s cutting: an action is considered greedy if its Q-value lies within 1 − γ of the maximum.
### π
695 This is motivated by the bounded scale of the SVF (Q ∈ [0, 1]): strict arg max comparisons under
696 neural-network approximation are too sensitive to small estimation errors, leading to traces being cut
697 too aggressively. The 1 − γ tolerance matches the natural scale of one-step Bellman backups.
698 Encoder. Gridworlds do not need an encoder because their observation is already appropriate (one-
699 hot encoding of the agent’s position). For continuous control, the state and state-goal encoders are
700 a fuzzy tile activation (FTA) layer Pan et al. [31]. FTA replaces the standard binning operation by
701 a soft-binning that retains non-negligible gradients while still producing sparse, high-dimensional
702 representations. Concretely, for each scalar input x, FTA places T learnable tile centers μ 1, . . . , μT
### 1
704
## μt)/σt) with learnable bandwidths σt> 0. To prevent vanishing gradients for inputs outside
707 tile dimension, yielding a vector in [0, 1] per input unit. We use T = 20 tiles initialized with
703 uniformly between v
## and v
## min maxand computes Gaussian activations φt(x) = exp −2((x −
### 2
| 705 [v | , v |
|---|---|
| min | max ] |
| 706 distance once |   |
|  |  |
| 708 v |  |
| =   − | 1 ,   v |
## min max= 1. Environment raw observations are standardized using running mean and
709 standard deviation tracked with Welford’s online algorithm.
710 Feature fusion. We combine the state and goal features by concatenating fs, fg, and their element-
711 wise (Hadamard) product fs⊙ fg, following the matching architecture used in InferSent [10]. The
712 element-wise product provides a multiplicative interaction term that makes pairwise alignment
713 between corresponding components of fsand fgdirectly available to the downstream layers, comple-
714 menting the purely additive information carried by the concatenation of fsand fgalone.
715 Maxout. After feature fusion we apply a Maxout unit [19], which computes K parallel linear
716 projections of its input and takes the element-wise maximum across them, followed by a Layer
717 Norm. Maxout acts as a learned, piecewise-linear activation: it is a universal approximator of convex
718 activation functions and has been shown to interact favorably with dropout. We use K = 4 experts.
719 F.2 GCRL Control
### π
720 Actions are continuous, thus we learn V (s, gs) and a goal-conditioned policy π(a|, s, gs). The
721 action is not part of the goal. We use TD3 + HER as in AdaGoal and DISCOVER original imple-
722 mentation. Once the goal is selected, the policy explores with white noise with standard deviation
## 723 0.1. AdaGoal and DISCOVER use an ensemble of 6 critics, each trained with different random
724 batches. For acting, the minimum value of each action across two random critics is used. To mitigate
### π
725 overestimation bias, all algorithms clip V to 1, its maximum true value: when the agent reaches a
726 goal, the reward in Eq. (1) is 1 and the transition is terminal.
727 SUN and the versions of AdaGoal and DISCOVER with multi-step goal relabeling update the critics
728 with TD(λ) using Watkin’s cutting traces [50]. At every update, we sample 8 batches of 10 samples
729 over consecutive environment steps. Vanilla AdaGoal and DISCOVER use 128 batches of 1 sample.
730 All algorithms use the same network architecture, shown in Figure 9.

---

## Page 22

Linear
LayerNorm Maxout
Dropout Leaky ReLU
SiLU Linear h = concat (fs, fg, fs⊙ fg) LayerNorm
Linear Leaky ReLU
LayerNorm gs Encoder ϕgLinear
Dropout
SiLU Qπ(s, a, gs, ga)
Figure 8: Architecture of Qπ. The size of hidden linear layers is 64 on all environments but Lunar
### Lander (128), and all are initialized with weights close to zero (drawn from a normal distribution
### with 0.01 standard deviation). All but the last have no bias. The dropout probability is 0.1.
s Encoder ϕs
Linear 256
LayerNorm
(s, a) Encoder ϕsa Leaky ReLU
Linear 64
Linear 1024
Maxout-4
LayerNorm
Leaky ReLU
h = concat (fsa, fg, Linear 256 fsa⊙ fg)
Leaky ReLU
Linear 1
Linear 256
LayerNorm
Leaky ReLU
Linear 64
Figure 9: Architecture of Vπand π. For π, we further apply the softmax operator at the end.
g s Encoder ϕg

---

## Page 23

# Figure 10: Visitation heatmaps at the end of training of Figure 5 during one run (same default seed).
# Figure 11: GCRL Visitation heatmaps.
731 G Heatmaps
732 Figure 10 provides a qualitative view of the visitation distributions underlying these metrics. The
733 heatmaps show visits at the end of training of the goal space (black regions are unvisited or un-
734 reachable). Across all environments, SUN covers a broader portion of the reachable state space and
735 produces smoother visitation patterns, while AdaGoal and DISCOVER concentrate their visits in
736 narrow regions. This confirms the entropy ranking from Figure 5 visually: methods can cover many
737 states while still over-concentrating their visits.

---

## Page 24

## Figure 12: Per-step vs. episodic exploration.
738 H Ablations
739 Due to limited compute resources, we an ablations only on gridworlds and classic control tasks.
740 H.1 Ablation: Per-Step vs. Episodic Goal Selection
741 Figure 12 shows SUN performance using its proposed per-step goal selection against DISCOVER
742 and AdaGoal episodic one. In the latter, the goal is selected only at the beginning of an episode, and
743 help fixed until reached. If reached, the agent then explores randomly. The per-step selection overall
744 performs better, mostly in MountainCar and Acrobot, with worse performance only in LunarLander.

---

## Page 25

Figure 13: Pseudocounts vs true counts in SUN.

---

## Page 26

## Figure 14: Comparison of different HER variants.
745 H.2 Ablation: HER vs. L-HER(λ)
746 Figure 13 compares SUN, AdaGoal, and DISCOVER using different version of HER, to show
747 that implicit negatives can negatively affect training. We compare HER(λ) with four goal sampled
748 within the same trajectory, a “full” version were all states within a trajectory are used to relabel, and
749 L-HER(λ), where only the last state is for relabeling. Results show a clear trend: L-HER is the best,
750 followed by “full HER” and finally HER. The fact that “Full HER” is not the best shows that adding
751 more relabeling does not help, giving strong evidence for the argument against implicit negatives.
752 H.3 Ablation: Pseudocounts vs. True Counts
753 We evaluate the proposed pseudocounts, comparing them to true counts. Note that for Gridworlds
754 pseudocounts are already the true counts. For continuous control, we approximate true counts by
755 discretizing the goal space with 50 bins per dimension. Figure 14 shows two variants are nearly
756 indistinguishable except for LunarLander, confirming that the pseudocount preserves the behavior of
757 count-based novelty while making per-step goal-selection feasible in continuous goal spaces.
758 H.4 AdaGoal and DISCOVER Novelty Collapse
759 To validate our intuition that AdaGoal and DISCOVER fail to explore due to novelty collapse,
760 we report the critic disagreement (their standard deviation). Figure 15 shows that indeed novelty
761 collapses, as all critics end up almost agreeing at some point. This explains their poor performance,
762 and is another strong argument in favor of SUN and its count-based novely.
## Figure 15: Critics standard deviation in all versions of AdaGoal and DISCOVER.

---

## Page 27

763 I SUN-UCB: PAC and Empirical Analysis
764 The structural results in Appendix A characterize SUN’s selection rule under an oracle but do
765 not provide sample-complexity guarantees. Here, we show that a UCB-like variant (with additive
766 reachability and novelty) admits a PAC bound matching AdaGoal-UCBVI (Upper Confidence Bound
767 Value Iteration [4]) in tabular MDPs. Subsequently, we empirically evaluate its performance in the
768 approximate deep-RL setting, showing that the proposed multiplicative version of Eq. (2) is better
769 (higher coverage and higher entropy in fewer training steps).
770 I.1 Setup
771 For the PAC analysis, we consider tabular MDPs , A, P, s0with finite spaces |S| = S and
### 772 |A| = A, and stochastic transitions P(s′| s, a). We adopt all assumptions in Assumption 1 except
773 (C2): we relax deterministic dynamics to stochastic with almost-sure hitting (as in Section A.6).
### π
774 All counts ngare exact (tabular setting). Under first-hit termination (C4), the SVF V (s, g) =
### 775 Eπ[γτg1{τg<∞}] ∈ [0, 1].
776 We use δ ∈ (0, 1) as the confidence parameter: the PAC bound holds with probability at least 1 − δ.
777 Let GL≜ {g ∈ S : ∃π s.t. Eπ[τg] ≤ L} be the set of L-reachable goals, with |GL| ≤ S. Episodes
778 have length L and reset to s0.
779 I.2 SUN-UCB algorithm
780 At step t, given stand counts nt(·):
### 781 1. Maintain optimistic value estimates Vˆtπ(st, g) via standard UCBVI [4].
### >0
### 782 2. Filter unreachable goals: Gt≜ {g : Vˆtπ(st, g) > 0}.
783 3. Select
# s
# πclog(SAT /δ)
### gt= arg max Vˆt(st, g) + , c = L2. (23)
>0
# g∈Gtnt(g)
# 784 4. Execute the goal-conditioned policy πˆgt; observe transition; update counts and estimates.
785 The filter in step 2 is the algorithmic counterpart of unreachability rejection (Theorem 3): it removes
786 goals whose value estimate is zero, which by optimism includes all truly unreachable goals.
787 I.3 PAC bound
788 Theorem 4 (PAC bound for SUN-UCB). Let ε ∈ (0, 1) and δ ∈ (0, 1). With probability at least
789 1 − δ, SUN-UCB returns goal-conditioned policies πˆ satisfying
### πˆ
### V (s , g) ≥ V⋆(s , g) − ε ∀g ∈ G
| 0 | 0 | L(24) |
|---|---|---|
|  |  |  |
|  |    |  |
|  |  |  |
| T   | =   O ˜ L 3   S A |  |
|  |  | (25) |
|  | ε 2 |  |
|  |  |  |
|  |  | 3 |
|  |  | − 2 |
792 This matches the rate of AdaGoal-UCBVI [46], which is nearly minimax-optimal: an Ω(L SAε )
793 lower bound rules out asymptotically better rates in this setting.
794 I.4 Proof of Theorem 4
# 795 The proof proceeds in five steps: concentration of Vˆ, unreachability rejection at the learning level,
796 bounding the selected goal’s uncertainty, pigeonhole, and conclusion.
797 Step 1: Concentration. We rely on the standard UCBVI machinery [4] to obtain a concentration
798 event.
799 Lemma 1 (Concentration). There exists an event E1of probability at least 1 − δ/3 on which:

---

## Page 28

π 800 1. (Optimism) For every s, g, t: Vˆt(s, g) + bt(s, g) ≥ V⋆(s, g), where bt(s, g) is the standard
801
# UCBVI bonus aggregated along the path to g.
802 2. (Concentration in goal-count form) For every g ∈ GLand every t:
# s
|Vˆtπ(s0, g) − Vπˆt(s0, g)| ≤ L2log(ntSAT /δ(g) ). (26)
803
# Proof sketch. Optimism is the classical UCBVI result [4, Lemma 7]. The goal-count form follows
804 by re-indexing: under first-hit termination, each visit top g provides a fresh sample of γτg∈ [0, 1], and
2
805 the empirical mean concentrates at rate L log /nt(g) by Hoeffding. The L in the bonus accounts 806 for value-range propagation along L-step paths.
807 Step 2: Unreachability rejection at the learning level.
Vˆtπ(s0, g) = 0. (27)
811 goal g is exactly 0. The empirical mean is therefore Vˆtπ(s0, g) = 0.
812 Combining Lemma 2 with the filter in step 2 of the algorithm:
G>0⊆ G
# t Lwith probability 1. (28)
814
# log /nt(g) may be for unreachable ones. This is the formal counterpart of Theorem 3 in the
815 learning setting.
816 Step 3: Selector targets high-uncertainty goals. Define the uncertainty of goal g at time t:
# s
L2log(SAT /δ)
# U (g) ≜
| t | . (29) |
|---|---|
|  | g )   |
|   π ˆ t   ( s 0 , g ) | ≤   U | for all   g   ∈ G L . |
| .   On   |   g t   selected by SUN-UCB satisfies |
|  |  |
| U ( g )   ≥ | g )   −   1 . |
# t t 2 g∈GLt 2(30)
∗ ∗ >0
819 Proof. Let g ≜ arg max
# t g∈GLUt(g). By unreachability rejection (Lemma 2) and the filter, gt∈ Gt
820 once visited at least once. By optimality of gtin Eq. (23):
π π ∗ Vˆt(s0, gt) + Ut(gt) ≥ Vˆt(s0, gt) + Ut(gt∗). (31)
821 Using Vˆtπ∈ [0, 1] (once-and-done bound):
1 + Ut(gt) ≥ Ut(gt∗), (32)
∗ 822 which gives Ut(gt) ≥ Ut(gt)/2 − 1/2 whenever Ut(gt∗) ≥ 2, and is trivial otherwise.
823 Step 4: Pigeonhole. By Lemma 3, each step visits a goal with uncertainty at least half the maximum 824 over GL. Standard UCB pigeonhole (Azar et al. 4, Lemma 12, adapted to goal counts):
XT
# p
# Ut(gt) ≤ 2L S · T · log(SAT /δ). (33)
t=1
825 For all g ∈ GLto satisfy UT(g) ≤ ε, every g must be visited at least
L2log(SAT /δ)
# nT(g) ≥ ε2(34)

---

## Page 29

### 2
### 2
828 goals each needing L log /ε visits, and accounting for the A-action branching factor at each state
826 times, summing to S · L log(SAT /δ)/ε total visits across GL.
827 Step 5: Conclusion. Each episode has length L, contributing L visits to goal states. With |GL| ≤ S
### 2
### 2
829 via standard UCBVI bookkeeping, the total exploration cost is
## 
## 
### 3
### T ≤ L S A log(SAT /δ) = O˜ L3S A . (35)
### 2
ε ε2
830 The failure probability is bounded by δ/3 + δ/3 + δ/3 = δ, accounting for E1, the goal-conditioned
831 policy execution event, and the pigeonhole union bound. □
832 I.5 Discussion
833 Comparison with AdaGoal. Theorem 4 matches AdaGoal-UCBVI’s sample complexity [46] up
834 to constants. Both algorithms exploit the same fundamental structure: optimism on the value,
835 an uncertainty signal that calibrates with the concentration rate, and a pigeonhole argument on
836 the reachable goal set. The two methods filter unreachable goals through different but equivalent
837 mechanisms: AdaGoal imposes an explicit constraint on the estimated hitting time Dk(g) ≤ L, while
### 838 SUN-UCB filters goals with Vˆtπ(st, g) = 0 (Lemma 2). Under first-hit termination, these two filters
### ⋆
### 839 are equivalent up to the change of variables L ↔ log(1/vmin)/log(1/γ), since V (s0, g) = γkg−1.
840 Role of unreachability rejection. Theorem 3 (proved in the oracle setting in Section A.4) is used
841 algorithmically in step 2 of SUN-UCB to filter unreachable goals, and is the load-bearing property inp
842 Lemma 2. Without it, the bonus log(SAT /δ)/nt(g) unreachable goals would be selected (where
843 the bonus is large because nt(g) = 0), and the PAC bound would fail. This is the formal connection
844 between SUN’s structural unreachability rejection and its sample-complexity guarantees.
845 I.6 Additive vs. multiplicative SUN
## p
846 The PAC bound is for the additive score V + log(SAT /δ)/nt(g), which differs from the multi-
847 plicative form V · 1/ngused in our experiments (Section 4). In Section 3 we argued why the latter
848 is better in deep-RL: with neural-network approximations, additive formulations are sensitive to
849 the relative scale of the two terms and typically require a tuning coefficient to balance them. SUN
850 multiplicative form sidesteps this scaling problem entirely: the two signals share a common “zero”
851 (an unreachable or already-saturated goal scores zero on either factor and is rejected regardless of the
852 other) and a common scale (both are non-negative and bounded).
853 Below, we validate this statement by comparing the two forms over all benchmarks of Section 4.p
854 For the sake of simplicity, we use the UCB1 formp 2 log(Σgng)/ng[3] rather than the PAC-style
### 8
855
## log(SAT /δ)/ngused in Theorem 4.
8The two forms are equivalent up to constants once T is fixed: Σgng = T after T exploration steps, so
### log(Σgng) and log(SAT /δ) differ by a constant log(SA/δ). The same proof goes through for the UCB1 form
### with slightly different constants but the same asymptotic rate.