# Algorithm Updates -- Planned Improvements

Places in the codebase where a placeholder, naive, or arbitrary algorithm is used with intent to revisit later.

---

## roc/significance.py -- `Significance.do_significance`

**Comment:** (implicit -- no TODO, but simple weighted sum)

**Context:** The significance calculation is a basic weighted linear sum of normalized intrinsic values. Weights are static floats from config (`significance_weights`). There is no adaptive weighting, no learned significance, and no consideration of game context or state transitions. This is the module the user specifically called out as needing algorithm work.

---

## roc/attention.py:143-158 -- `SaliencyMap.get_strength`

**Comment:**
```
# TODO: not really sure that the strength should depend on the number of features
# TODO: this is pretty arbitrary and might be biased based on my
# domain knowledge... I suspect I will come back and modify this
# based on object recognition and other factors at some point in the future
```

**Context:** Saliency strength is computed as feature count + hardcoded bonuses per feature type (Single +10, Delta +15, Motion +20). The weights are arbitrary and acknowledged as domain-knowledge-biased. Should eventually incorporate object recognition feedback and learned weights.

---

## roc/attention.py:292-300 -- `CrossModalAttention`

**Comment:**
```
# TODO: other attention classes
# TODO: listen for attention events
# TODO: select and emit a single event
```

**Context:** `CrossModalAttention` is a stub component. It exists as a placeholder for cross-modal attention integration (combining visual + other modalities), but has no implementation.

---

## roc/object.py:88-98 -- `Object.distance`

**Comment:**
```
# TODO: allowed_attrs is physical attributes, not really great but
# NetHack doesn't give us much feature-space to work with. in the future
# we may want to come back and use motion or other features for object recognition
# TODO: line? flood?
```

**Context:** Object distance/similarity uses symmetric set difference on a hardcoded whitelist of physical feature types (`SingleNode`, `ColorNode`, `ShapeNode`). Motion, line, flood, and other features are excluded. The distance metric itself is just the count of non-overlapping features -- no weighting, no learned similarity.

---

## roc/object.py:124-130 -- `CandidateObjects.__init__`

**Comment:**
```
# TODO: this currently only uses features, not context, for resolution
# the other objects in the current context should influence resolution
# TODO: getting all objects for the set of features is going to be a
# huge explosion of objects... need to come back to this an make a
# smarter selection algorithm
```

**Context:** Object resolution candidates are found by walking all feature groups for all feature nodes and collecting every associated object. No pruning, no spatial/temporal context, no indexing. Will not scale as the object graph grows. Contextual objects (nearby objects, recent objects) should influence candidate scoring.

---

## roc/object.py:188-190 -- `ObjectResolver.do_object_resolution`

**Comment:**
```
# TODO: instead of just taking the first focus_point (highest saliency
# strength) we probably want to adjust the strength for known objects /
# novel objects
```

**Context:** Object resolution always processes only the single highest-saliency focus point. Known vs. novel objects should modulate attention priority -- e.g., novel objects should get more attention, known objects less.

---

## roc/object.py:206-208 -- `ObjectResolver.do_object_resolution`

**Comment:**
```
# TODO: "> 1" as a cutoff for matching is pretty arbitrary
# should it be a % of features?
# or the cutoff for matching be determined by how well the prediction works?
```

**Context:** The threshold for deciding "new object" vs "existing object" is a hardcoded `dist > 1`. Should be a dynamic threshold -- possibly a percentage of total features, or feedback-driven from prediction accuracy.

---

## roc/predict.py:64 -- `Predict.do_predict`

**Comment:**
```
# TODO: play forward multiple frames?
```

**Context:** Prediction only looks one frame ahead. The architecture supports multi-step prediction (applying transforms repeatedly), but this is not yet implemented.

---

## roc/predict.py:72 -- `Predict.do_predict`

**Comment:** (implicit -- references `default="naive"`)

**Context:** Prediction confidence scoring uses a "naive" ExpMod, but no concrete implementation of `PredictionConfidenceExpMod` with `name = "naive"` exists in the codebase. The confidence scoring algorithm is either missing or expected to be plugged in. The entire prediction confidence system needs a real implementation.

---

## roc/intrinsic.py:123 -- `IntrinsicTransformable.get_transform`

**Comment:**
```
# TODO: create transform for raw values using IntrinsicOps?
```

**Context:** Intrinsic transforms currently only track normalized change. Raw value transforms (using `IntrinsicOps` like min/max/delta) are not generated, limiting the richness of intrinsic state change representation.

---

## roc/graphdb.py:330-332 -- `Node.to_gml` (label export)

**Comment:**
```
# TODO: this converts labels to a string, but maybe there's a better
# way to preserve the list so that it can be used for filtering in
# external programs
```

**Context:** When exporting to GML, node labels (a set) are joined into a single string. This loses structure and makes filtering by individual label harder in external graph tools.

---

## roc/graphdb.py:969-970 -- `EdgeList.select`

**Comment:**
```
# TODO: Edge.get_many() would be more efficient here if / when it gets implemented
```

**Context:** Edge filtering iterates and fetches edges one-by-one via `Edge.get()`. A batch `Edge.get_many()` would reduce DB round-trips.

---

## roc/component.py:152 -- `Component.load`

**Comment:**
```
# TODO: shutdown previously loaded components
```

**Context:** When loading a new set of components, previously loaded components are not shut down or cleaned up. Could cause resource leaks or stale event listeners.

---

# Magic Numbers -- Candidates for Algorithmic Improvement

These are hardcoded numeric constants that control algorithmic behavior and could benefit from being configurable, learned, or replaced with a proper algorithm.

---

## roc/attention.py:154-158 -- `SaliencyMap.get_strength`

**Magic numbers:** `+10` (Single), `+15` (Delta), `+20` (Motion)

**Context:** Hardcoded saliency bonuses per feature type. These define how much each feature type contributes to attention. There is no principled basis for these values -- they are guesses about relative importance. Could be config-driven weights, or learned from prediction feedback.

---

## roc/object.py:209 -- `ObjectResolver.do_object_resolution`

**Magic number:** `> 1` (distance threshold)

**Context:** Already listed above. The threshold `1` for new-vs-existing object matching has no theoretical basis. Could be a percentage of feature count, a configurable value, or dynamically adjusted.

---

## roc/feature_extractors/line.py:12 -- `MIN_LINE_COUNT`

**Magic number:** `4`

**Context:** Minimum number of consecutive identical-value points to qualify as a "line" feature. This determines the granularity of line detection. Smaller values would detect more/shorter lines (more noise); larger values would miss short lines. Not configurable, not derived from data.

---

## roc/feature_extractors/flood.py:15 -- `MIN_FLOOD_SIZE`

**Magic number:** `5`

**Context:** Minimum number of adjacent identical-value cells to qualify as a "flood" feature. Controls whether small clusters are recognized as coherent regions. Same tradeoff as `MIN_LINE_COUNT` -- too small is noisy, too large misses small features.

---

## roc/config.py:152-158 -- `ConfigMapIntrinsic` (hunger normalization)

**Magic numbers:** `0.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.0` (hunger state mappings)

**Context:** Maps NetHack hunger states (SATIATED through STARVED) to normalized [0,1] values. These define how "significant" each hunger level is. The mapping is hand-designed -- e.g., SATIATED and WEAK are both `0.5`, which may not accurately reflect their gameplay impact. Could be learned from outcome data.

---

## roc/config.py:170-173 -- `significance_weights`

**Magic number:** `10.0` (hp weight)

**Context:** Only HP has a weight (`10.0`); all other intrinsics default to `1.0`. This makes HP 10x more significant than energy or hunger in the significance calculation. The weight has no empirical basis and only one intrinsic is even configured.

---

## roc/feature_extractors/phoneme.py:33,36 -- `PhonemeFeature._create_nodes` / `_dbfetch_nodes`

**Magic number:** `42` (phoneme type)

**Comment:** `# XXX TODO type`

**Context:** Phoneme nodes always use `type=42` as a placeholder. Every phoneme feature is identical from the graph's perspective. The phoneme extractor has no real type differentiation -- it needs a proper scheme for encoding phoneme identity.

---

## roc/event.py:56-57 -- `EventBus` pretty-printing

**Magic numbers:** `max_length=5`, `max_string=60`

**Context:** These control how event data is displayed in debug output. Not algorithmic per se, but they silently truncate data that might be relevant during debugging. Minor concern.
