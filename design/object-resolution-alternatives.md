# Object Resolution: Current Approach and Alternatives

## Current Approach: Symmetric Set Difference with Threshold

### Overview

The current object resolution algorithm lives in `roc/object.py` in the `ObjectResolver` and `CandidateObjects` classes. It runs once per frame on the single highest-saliency focus point and uses a simple set-based distance metric to match observations to known Objects.

### Algorithm

1. **Select focus point**: The attention system produces a ranked list of focus points by saliency strength. Only the top-ranked point is used (`focus_points.iloc[0]`).

2. **Extract features**: The visual features at the focus point's (x, y) location are retrieved from the saliency map and converted into a `FeatureGroup` -- a graph node connected to individual `FeatureNode`s (SingleNode, ColorNode, ShapeNode, etc.) via `Detail` edges.

3. **Find candidates**: Starting from the new FeatureGroup's feature nodes, the algorithm walks backwards through the graph (`FeatureNode -> FeatureGroup -> Object`) to find all existing Objects that share at least one feature node.

4. **Compute distance**: For each candidate Object, compute the symmetric difference between the new feature set and the Object's feature set. Only "physical" features are considered: `SingleNode`, `ColorNode`, and `ShapeNode`. Motion, distance, line, and flood features are excluded. The distance is the count of features present in one set but not the other.

5. **Match or create**: Candidates are sorted by distance (ascending). If the best candidate has distance <= 1, it is considered a match and its `resolve_count` is incremented. Otherwise, a new Object is created and linked to the new FeatureGroup.

### Limitations

- Only one object is resolved per frame (the highest-saliency point).
- The distance metric treats all features as equally important.
- The matching threshold (distance > 1) is arbitrary and not learned.
- Candidate search can be expensive since it walks all connected FeatureGroups and Objects.
- Context (other objects in the scene, spatial position, temporal history) is not used.
- Only a subset of feature types contribute to matching.

---

## Alternative 1: Jaccard Similarity

### Description

Replace the symmetric difference count with Jaccard similarity: `|intersection| / |union|`. This normalizes the distance for objects with different numbers of features. Under the current metric, an object with 10 features will always have a larger symmetric difference than one with 2 features, even if the 2-feature object is a worse match. Jaccard produces a value between 0 (no overlap) and 1 (identical), making thresholds more interpretable and stable across objects of varying complexity.

### Why it would help

- Normalizes for feature set size, making the threshold meaningful across different object types.
- Trivial to implement -- same data, different formula.
- Well-understood metric with known properties.

### Implementation notes

Replace `Object.distance()` to return `1 - (|intersection| / |union|)` so that 0 still means a perfect match and the sort order is preserved. Adjust the threshold accordingly (e.g., 0.5 instead of > 1).

### References

- Jaccard index: https://en.wikipedia.org/wiki/Jaccard_index
- Levandowsky, M. & Winter, D. (1971). "Distance between Sets." Nature, 234(5323), 34-35.

---

## Alternative 2: Weighted / TF-IDF Features

### Description

Not all features are equally discriminating. A `ColorNode(type=15)` shared by hundreds of objects is less informative than a `ShapeNode(type=114)` that only appears on three objects. Weight each feature by its inverse document frequency -- how rare it is across all known objects. This is the same intuition behind TF-IDF in information retrieval.

### Why it would help

- Common features (e.g., the most frequent color in the dungeon) would stop dominating the distance calculation.
- Rare, distinctive features would have more influence, improving discrimination between similar-looking objects.
- Can be combined with Jaccard or any other set/vector similarity metric.

### Implementation notes

Maintain a count of how many Objects each FeatureNode is connected to. When computing distance, weight each feature's contribution by `log(N / df)` where N is total objects and df is the number of objects containing that feature. This can be computed from the graph structure directly.

### References

- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management, 24(5), 513-523.

---

## Alternative 3: Spatial Priors

### Description

Use the (x, y) position of objects to constrain matching. An object observed at position (5, 3) last frame is far more likely to be the same object seen at (5, 3) or (6, 3) this frame than one at (20, 15). The current algorithm completely ignores spatial information during resolution, even though `ResolvedObject` already carries x and y coordinates.

### Why it would help

- NetHack objects generally don't teleport. Spatial proximity is a strong signal for identity.
- Reduces the candidate set significantly -- instead of searching all Objects that share a feature, only consider Objects last seen nearby.
- Handles the common case where two visually identical objects (e.g., two jackals) are distinguished purely by position.
- The data is already available: `ResolvedObject` has `x` and `y`, and `ObjectCache` is already keyed by `(XLoc, YLoc)`.

### Implementation notes

For each candidate Object, look up the position where it was last resolved. Compute Manhattan or Euclidean distance from the current focus point. Combine spatial distance with feature distance, either as a weighted sum or as a filter (only consider objects within radius R). The `ObjectCache` could be extended to store the last-seen position per Object.

### References

- Multi-object tracking with spatial assignment: https://en.wikipedia.org/wiki/Multiple_object_tracking
- Kuhn, H.W. (1955). "The Hungarian method for the assignment problem." Naval Research Logistics Quarterly, 2(1-2), 83-97.
- Bewley, A. et al. (2016). "Simple Online and Realtime Tracking." IEEE ICIP. https://arxiv.org/abs/1602.00763 (SORT tracker -- uses Kalman filter + Hungarian algorithm, good practical reference)

---

## Alternative 4: Bipartite Matching (Global Assignment)

### Description

Instead of resolving one object per frame (the highest-saliency point), resolve all visible objects simultaneously using bipartite matching. Build a cost matrix where rows are observed feature groups in the current frame and columns are known Objects. Each cell contains the distance between that observation and that Object. Use the Hungarian algorithm to find the minimum-cost global assignment.

### Why it would help

- Prevents two observations from claiming the same Object (a problem the current greedy approach can't avoid across frames).
- Resolves all objects per frame, not just the top saliency point -- this is the single biggest limitation of the current approach.
- Produces globally optimal assignments rather than greedy local ones.
- Standard approach in multi-object tracking systems.

### Implementation notes

This requires the attention system to emit multiple focus points per frame (or a separate mechanism to enumerate all objects on screen). For each frame, collect all observed feature groups, compute pairwise distances to all known Objects, and solve the assignment. Unmatched observations become new Objects. The `scipy.optimize.linear_sum_assignment` function implements the Hungarian algorithm.

### References

- Hungarian algorithm: https://en.wikipedia.org/wiki/Hungarian_algorithm
- scipy.optimize.linear_sum_assignment: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
- Kuhn, H.W. (1955). "The Hungarian method for the assignment problem." Naval Research Logistics Quarterly, 2(1-2), 83-97.
- Munkres, J. (1957). "Algorithms for the Assignment and Transportation Problems." Journal of the Society for Industrial and Applied Mathematics, 5(1), 32-38.

---

## Alternative 5: Feature Vectors with Cosine Similarity

### Description

Instead of representing features as discrete sets, encode each feature group as a fixed-length numeric vector. Each dimension corresponds to a feature type, and the value encodes presence, count, or a continuous attribute. Compare feature groups using cosine similarity or Euclidean distance. This moves from "exact feature match" to "how similar are these feature profiles."

### Why it would help

- Supports partial similarity -- two objects can be "somewhat similar" rather than just "matching or not."
- Enables use of all feature types (including motion direction, distance size, flood size) as continuous dimensions rather than requiring exact matches.
- Opens the door to learned embeddings where the vector representation is optimized for discrimination.
- More robust to small variations in features between frames.

### Implementation notes

Define a feature vector schema: one dimension per (feature_type, attribute) pair. For example, `[SingleNode_type, ColorNode_type, ShapeNode_type, MotionNode_direction_encoded, DistanceNode_size, ...]`. Populate the vector from FeatureGroup's feature nodes. Compare with cosine similarity. Store vectors on the Object node or in a side index for fast lookup.

### References

- Cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- Feature embedding for visual recognition: Bengio, Y. et al. (2013). "Representation Learning: A Review and New Perspectives." IEEE TPAMI, 35(8), 1798-1828. https://arxiv.org/abs/1206.5538

---

## Alternative 6: Bayesian Object Identity

### Description

Instead of a hard match/no-match threshold, maintain a probability distribution over object identities. For each observation, compute the posterior probability that it corresponds to each known Object (or a new one) given the features, position, and temporal context. Use Bayes' rule to update beliefs as evidence accumulates.

### Why it would help

- Handles uncertainty naturally -- the system can express "70% likely to be Object A, 30% likely to be Object B" rather than committing to a binary decision.
- Deferred commitment: when evidence is ambiguous, the system can wait for more information before deciding.
- The prior can encode spatial and temporal expectations (where objects should be, how likely they are to persist).
- Fits the "reinforcement learning of concepts" philosophy -- beliefs about object identity are refined over time.

### Algorithm detail

The algorithm proceeds in five steps per frame:

**Step 1: Compute priors -- P(identity) before seeing features.**

Before examining features, assign prior probabilities to each known Object and a "new object" hypothesis based on:

- Spatial prior: Objects last seen nearby get higher probability. An object at (5, 3) last frame is far more likely at (5, 4) than at (20, 15).
- Temporal prior: Objects not seen for many frames get lower probability (they may have disappeared).
- New object prior: A base probability (e.g., 0.1) that the observation is something never seen before.

Example with 5 known Objects:

```
P(O1) = 0.05    (far away)
P(O2) = 0.02    (not seen recently)
P(O3) = 0.50    (nearby, seen last frame)
P(O4) = 0.15    (moderately close)
P(O5) = 0.18    (moderately close)
P(new) = 0.10   (base rate for new objects)
```

**Step 2: Compute likelihoods -- P(features | identity).**

For each known Object, compute how likely the observed features are under that Object's feature profile. A simple independent-feature model multiplies per-feature probabilities. A Dirichlet-Categorical model (see below) provides a principled way to estimate these probabilities from observation counts.

**Step 3: Compute posteriors via Bayes' rule.**

Multiply prior by likelihood and normalize:

```
P(Oi | features) = P(features | Oi) * P(Oi) / Z
```

where Z is the sum over all candidates including "new." This produces a probability distribution over identities.

**Step 4: Make a decision.**

Three options for using the posterior:

- Simple MAP: Take the highest-probability identity. If it exceeds a confidence threshold (e.g., 0.7), resolve as that Object. Otherwise create a new Object.
- Deferred commitment: If the posterior is ambiguous (e.g., P(O3) = 0.45, P(O4) = 0.40), don't commit. Carry the distribution forward and let the next frame's observation refine it.
- Soft assignment: Pass the full distribution downstream. The Transformer and Action components weight their processing by identity probability.

**Step 5: Update beliefs across frames.**

The posterior from frame N becomes the prior for frame N+1. This means identity beliefs accumulate evidence over time. If O3 usually has 3 features but one is occasionally missing, the Bayesian approach handles it gracefully -- spatial/temporal priors keep the identity stable through appearance changes.

### The Dirichlet-Categorical model for feature likelihoods

The Dirichlet-Categorical model provides a principled way to learn each Object's feature profile and compute likelihoods for Step 2.

**Categorical distribution**: Each Object has a probability distribution over features -- its "feature profile." When you observe the object, you draw from that distribution.

**The problem**: You don't know an Object's true feature distribution. You've only seen it a few times. The current set-difference approach treats features as all-or-nothing, which is brittle.

**Dirichlet prior**: The Dirichlet distribution is a "distribution over distributions." It expresses uncertainty about what an Object's feature profile actually is, parameterized by a vector of pseudo-counts (alpha values), one per possible feature:

```
alpha = [alpha_SingleNode(399), alpha_ColorNode(10), alpha_ShapeNode(114), ...]
```

- `alpha = 1` for a feature means "no information" (uniform prior)
- `alpha = 5` means "strong evidence this feature belongs to this object"
- `alpha = 0.1` means "would be surprised if this feature appeared"

**Conjugate updating**: Every time you observe an Object and see a feature, add 1 to that feature's alpha. The update rule is just counting:

```
Before:       alpha_SingleNode(399) = 1.0   (no information)
See it once:  alpha = 2.0
See it 10x:   alpha = 11.0
```

**Computing likelihoods**: Use the posterior predictive distribution. For Object Oi with alpha vector a:

```
P(feature_j | Oi) = a_j / sum(a)
```

For a full observation, multiply per-feature probabilities (assuming independence):

```
P(observation | Oi) = product of P(feature_j | Oi) for each observed feature_j
```

**Concrete example**: Two Objects after several frames:

O3 (seen 20 times):
```
alpha_SingleNode(399) = 21    (prior 1 + seen 20 times)
alpha_ColorNode(10)   = 19    (prior 1 + seen 18 times)
alpha_ShapeNode(114)  = 16    (prior 1 + seen 15 times)
alpha_ShapeNode(35)   = 1     (prior 1 + never seen)
```

O4 (seen 5 times):
```
alpha_SingleNode(399) = 6     (prior 1 + seen 5 times)
alpha_ColorNode(10)   = 4     (prior 1 + seen 3 times)
alpha_ShapeNode(114)  = 1     (prior 1 + never seen)
alpha_ShapeNode(35)   = 6     (prior 1 + seen 5 times)
```

New observation: `{SingleNode(399), ColorNode(10), ShapeNode(114)}`

```
P(obs | O3) = (21/sum) * (19/sum) * (16/sum) = high
P(obs | O4) = (6/sum)  * (4/sum)  * (1/sum)  = much lower
```

O3 wins because it has seen `ShapeNode(114)` many times while O4 never has.

**Advantages over set difference**:

- Handles noise: If O3 usually has 3 features but occasionally shows 2, the model captures that. Set difference treats a missing feature as a hard mismatch.
- Learns over time: The more you see an Object, the more confident the model becomes. Early observations have high uncertainty; later observations are precise.
- Graceful with rare features: A feature seen once has low alpha -- the model doesn't over-commit to it. Set difference treats a feature seen once the same as one seen 1000 times.
- Principled "new object" detection: An observation that doesn't fit any existing Object well (low likelihood under all models) naturally gets assigned to "new." The threshold emerges from the math rather than being hand-tuned.

**Limitation**: The independence assumption doesn't capture feature correlations (certain glyph types always co-occur with certain colors). This can be addressed with models over feature pairs, but the independent Dirichlet-Categorical is a solid starting point and very cheap to compute.

### Implementation notes

In the ROC graph:

- Each Object gets a "last seen" position and tick stored on the node (or a separate tracking structure) for spatial/temporal priors.
- Alpha vectors can be stored as a property on the Object node or maintained in a side dictionary.
- The `Features` edge between Object and FeatureGroup could carry a confidence weight from the posterior.
- The `resolve_count` on Object is replaced (or supplemented) by accumulated posterior mass.

The existing architecture supports this well: FeatureNodes are already deduplicated (so likelihood computation is set operations + weighting), and `ObjectCache` keyed by `(XLoc, YLoc)` is already a crude spatial prior that just needs to become probabilistic.

### References

- Bayesian inference: https://en.wikipedia.org/wiki/Bayesian_inference
- Conjugate priors (Dirichlet-Categorical): https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions
- Xu, F. & Tenenbaum, J.B. (2007). "Word Learning as Bayesian Inference." Psychological Review, 114(2), 245-272. (Bayesian concept learning, relevant to learning object categories)
- Murphy, K.P. (2012). "Machine Learning: A Probabilistic Perspective." MIT Press. Section 3.4 covers the Dirichlet-Categorical model; Chapters 3-5 cover Bayesian inference for classification.
- Gelman, A. et al. (2013). "Bayesian Data Analysis." 3rd edition. Chapter 2 for the fundamentals of Bayesian updating.

---

## Alternative 7: Kalman Filter / Particle Filter Tracking

### Description

Model each known Object as a tracked entity with a state (position, velocity, feature profile) and uncertainty. Each frame, predict where each object should be and what it should look like, then match observations to predictions. Update the state estimate based on the observation. This is the standard approach in visual object tracking.

### Why it would help

- Predicts object locations, making matching robust to occlusion and noise.
- Naturally handles motion -- if an object was moving right, the filter predicts it will continue moving right.
- Quantifies uncertainty in object position and identity.
- Can detect when a tracked object has disappeared (prediction with no matching observation) or when a new object has appeared (observation with no matching prediction).

### Implementation notes

For each Object, maintain a Kalman filter state: `[x, y, vx, vy]` with associated covariance. Each frame: (1) predict next state for all tracked objects, (2) compute assignment cost as Mahalanobis distance between predictions and observations, (3) solve assignment with Hungarian algorithm, (4) update matched filters, create new filters for unmatched observations, mark unmatched predictions as potentially lost. `filterpy` is a good Python library for this. For non-linear dynamics or multimodal distributions, particle filters are more flexible.

### References

- Kalman filter: https://en.wikipedia.org/wiki/Kalman_filter
- Particle filter: https://en.wikipedia.org/wiki/Particle_filter
- Welch, G. & Bishop, G. (2006). "An Introduction to the Kalman Filter." UNC Chapel Hill TR 95-041. https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
- Bewley, A. et al. (2016). "Simple Online and Realtime Tracking (SORT)." https://arxiv.org/abs/1602.00763
- Wojke, N. et al. (2017). "Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)." https://arxiv.org/abs/1703.07402
- filterpy library: https://github.com/rlabbe/filterpy

---

## Alternative 8: Learned Distance (Reinforcement-Based)

### Description

Rather than hand-designing the distance metric, learn which features matter for correct resolution through experience. Track which resolutions lead to good downstream outcomes (correct predictions, successful actions) and adjust feature weights accordingly. This could be as simple as a linear model with learned weights, or as complex as a small neural network.

### Why it would help

- Aligns with the project's reinforcement learning theme -- the agent learns what matters for object identity.
- Adapts to the specific game environment rather than relying on hand-tuned heuristics.
- Can discover non-obvious feature combinations that are discriminating.
- The threshold for "same object" can be learned rather than set arbitrarily.

### Implementation notes

Define a reward signal for object resolution quality. One option: if a resolved object's predicted next state (from Transform) matches the actual next state, the resolution was likely correct. Use this signal to update weights on features in the distance function. Start with a logistic regression: `P(same_object) = sigmoid(w . feature_diff_vector)`. Train online as the agent plays.

### References

- Metric learning: https://en.wikipedia.org/wiki/Similarity_learning
- Xing, E. et al. (2002). "Distance Metric Learning with Application to Clustering with Side-Information." NIPS. https://papers.nips.cc/paper/2002/hash/c3e4035af2a1cde9f21e1ae1951ac80b-Abstract.html
- Bromley, J. et al. (1993). "Signature Verification using a Siamese Time Delay Neural Network." NIPS. (Siamese networks for learned similarity)
- Chopra, S. et al. (2005). "Learning a Similarity Metric Discriminatively, with Application to Face Verification." CVPR.

---

## Recommendation

The alternatives are listed roughly in order of implementation complexity. For the highest impact relative to effort:

1. **Spatial priors** (Alternative 3) -- Cheapest win. Position data already exists in the system. Disambiguates visually identical objects (e.g., two monsters of the same type).

2. **Bipartite matching** (Alternative 4) -- Addresses the single biggest limitation: only resolving one object per frame. Unlocks multi-object tracking.

3. **Jaccard + weighted features** (Alternatives 1 and 2) -- Simple improvements to the distance metric that don't require architectural changes. Can be done in a single function.

The more complex alternatives (Bayesian, Kalman, learned distance) are worth considering once the simpler improvements are in place and the system needs to handle more challenging scenarios.
