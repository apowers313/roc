# Inhibition of Return (IOR): Research Survey for ROC Saliency Module

## 1. Overview

Inhibition of Return (IOR) is an attentional phenomenon where responses to stimuli at previously attended locations are slower than responses at novel locations. First described by Posner and Cohen (1984), IOR is understood as a mechanism that biases attention toward novelty and prevents perseverative re-examination of already-inspected locations. This document surveys the major theories, models, and empirical findings relevant to implementing IOR in the ROC saliency module.

---

## 2. Foundational Discovery

### 2.1 Posner & Cohen (1984)

- **Paper:** Posner, M. I., & Cohen, Y. (1984). "Components of visual orienting." In H. Bouma & D. G. Bouwhuis (Eds.), *Attention and Performance X* (pp. 531-556).
- **Paradigm:** Three boxes on screen (center + two peripheral). One peripheral box brightened for 150 ms (the "cue"). Target (small dot) appeared 0-500 ms after cue onset in one of the boxes.
- **Key finding:** At short stimulus onset asynchronies (SOAs < 200 ms), responses were *facilitated* at the cued location. At SOAs > 300 ms, responses were *slower* at the cued location than at uncued locations -- this is IOR.
- **Proposed function:** Posner and Cohen suggested IOR encourages orienting toward novel objects and events.
- **Note:** The term "inhibition of return" was actually coined by Posner, Rafal, Choate, & Vaughan (1985), not by the original 1984 paper.

**Source:** [Scholarpedia: Inhibition of Return](http://www.scholarpedia.org/article/Inhibition_of_return); [Posner & Cohen on ResearchGate](https://www.researchgate.net/publication/203918232_Components_of_visual_orienting)

---

## 3. Major Theoretical Accounts

### 3.1 Foraging Facilitator Hypothesis

- **Key paper:** Klein, R. M. (2000). "Inhibition of return." *Trends in Cognitive Sciences*, 4(4), 138-147.
- **Earlier proposal:** Klein, R. M. (1988). "Inhibitory tagging system facilitates visual search." *Nature*, 334, 430-431.
- **Theory:** IOR facilitates visual search by attaching inhibitory "tags" to already-inspected spatial locations, discouraging attention from returning to them. This makes the visual system function like an efficient forager, avoiding already-sampled patches.
- **Evidence for:** Klein (1988) demonstrated that following a difficult visual search, IOR was found at locations of rejected distractors, suggesting inhibitory tagging during search.
- **Ecological validation:** Klein & MacInnes (1999) tested IOR under ecologically valid conditions using "Where's Wally?" search images with eye tracking. Probes appeared at previous fixation locations (1-back, 2-back) or at new locations equidistant from current fixation. Saccadic RTs were slower to prior fixation locations than to novel locations, supporting the foraging facilitator role.
- **Evidence against:** Klein later acknowledged challenges to this proposal. Horowitz & Wolfe (1998, 2001) argued that visual search behaves as if it has no memory -- randomizing display during search does not impair performance, suggesting random sampling rather than inhibitory tagging. However, subsequent work has continued to find evidence supporting IOR's role in search efficiency.
- **Recent support:** Redden et al. (2023) found in *Attention, Perception, & Psychophysics* that long-term training on visual search tasks showed IOR effects consistent with the foraging facilitator account.

**Source:** [Klein 2000 in Trends in Cognitive Sciences](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(00)01452-2); [Redden et al. 2023](https://link.springer.com/article/10.3758/s13414-022-02605-0)

### 3.2 Habituation Account

- **Key paper:** Dukewich, K. R. (2009). "Reconceptualizing inhibition of return as habituation of the orienting response." *Psychonomic Bulletin & Review*, 16(2), 238-251.
- **Theory:** IOR is not a specialized inhibitory mechanism but rather a consequence of the general phenomenon of habituation. The cue produces a sensory response; when a similar stimulus (the target) appears at the same location shortly after, the orienting response to the target is weakened because the sensory system has already habituated to stimulation at that location.
- **Key prediction:** At short intervals, the cue's residual activation *adds* to target processing (producing facilitation). At longer intervals, this residual activation decays but habituation persists, producing the IOR effect.
- **Neurophysiological support:** Direct support comes from studies showing sensory depression in the superficial layers of the superior colliculus at previously stimulated locations (Fecteau & Munoz, 2006), paralleling classical habituation.
- **Contrast with foraging facilitator:** The habituation account treats IOR as a byproduct of basic neural adaptation rather than a purpose-built search mechanism. It can explain IOR without positing a specialized inhibitory tagging system.

**Source:** [Dukewich 2009 in Psychonomic Bulletin & Review](https://link.springer.com/article/10.3758/PBR.16.2.238)

### 3.3 Two Forms: Input-Based vs. Output-Based IOR

- **Key papers:**
  - Taylor, T. L., & Klein, R. M. (2000). "Visual and motor effects in inhibition of return." *Journal of Experimental Psychology: Human Perception and Performance*, 26, 1639-1656.
  - Redden, R. S., MacInnes, W. J., & Klein, R. M. (2021). "Inhibition of return: An information processing theory of its natures and significance." *Cortex*, 135, 30-48.

- **Input-based (perceptual/attentional) IOR:**
  - Operates on a *salience map* that influences what will capture attention
  - Decreases the sensory salience of inputs at previously attended locations
  - Generated when the reflexive oculomotor system is *suppressed* (e.g., fixation maintained)
  - Only observed with peripheral targets, not central arrow targets
  - Affects early processing/encoding stages
  - Mechanism: attenuates exogenous activity at previously attended locations

- **Output-based (motoric/decisional) IOR:**
  - Operates on a *priority map* that influences response execution
  - Biases responses away from previously attended locations
  - Generated when the reflexive oculomotor system is *active* (saccades permitted)
  - Observed with both peripheral targets and central arrows
  - Affects late processing/response selection stages
  - Mechanism: inhibits oculomotor responses to previously attended locations

- **Empirical support:** Taylor & Klein (2000) tested 24 variants of Posner's cueing paradigm and found the two forms dissociate based on whether eye movements were made. Redden et al. (2021) applied drift diffusion modeling and showed the two forms correlate with different diffusion parameters.

| Property | Input-Based IOR | Output-Based IOR |
|---|---|---|
| Processing stage | Early (encoding) | Late (response selection) |
| Map affected | Salience map | Priority map |
| Peripheral targets | Effect present | Effect present |
| Central arrows | Effect absent | Effect present |
| Oculomotor state | Suppressed | Active |
| Mechanism | Salience reduction | Response bias |

**Source:** [Taylor & Klein 2000](https://pubmed.ncbi.nlm.nih.gov/11039490/); [Redden et al. 2021 in Cortex](https://pubmed.ncbi.nlm.nih.gov/33360759/); [Klein, Redden & Hilchey 2023 in Frontiers in Cognition](https://www.frontiersin.org/journals/cognition/articles/10.3389/fcogn.2023.1146511/full)

### 3.4 Object-Based IOR

- **Key papers:**
  - Tipper, S. P., Driver, J., & Weaver, B. (1991). "Object-centred inhibition of return of visual attention." *Quarterly Journal of Experimental Psychology*, 43A, 289-298.
  - Tipper, S. P., Weaver, B., Jerreat, L. M., & Burak, A. L. (1994). "Object-based and environment-based inhibition of return of visual attention." *Journal of Experimental Psychology: Human Perception and Performance*, 20, 478-499.
  - List, A., & Robertson, L. C. (2007). "Inhibition of return and object-based attentional selection." *Journal of Experimental Psychology: Human Perception and Performance*, 33(6), 1322-1334.

- **Theory:** IOR is not purely location-based -- it can "tag" and follow moving objects. When an object is cued and then moves to a new location, IOR follows the object to its new position. This demonstrates that the inhibitory tag can be attached to object representations, not just spatial coordinates.

- **Space-based vs. object-based:**
  - Space-based IOR: ~28-38 ms inhibition, robust across conditions, sustained over time
  - Object-based IOR: ~14-15 ms inhibition, fragile, sensitive to procedural manipulations
  - Different time courses: object-based IOR has a slow rise time (~600 ms from most recent cue) and fast fall time
  - Space-based IOR is more robust for search efficiency; object-based IOR has questionable functional utility

**Source:** [List & Robertson 2007 in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2559861/)

---

## 4. Temporal Dynamics

### 4.1 Time Course of Facilitation-to-Inhibition Transition

- **Facilitation phase:** 0-200 ms after cue onset, responses at the cued location are faster (exogenous attentional capture)
- **Crossover point:** ~200-300 ms SOA, the transition from facilitation to inhibition
- **IOR onset:** Typically emerges around 300 ms SOA in simple detection tasks
- **IOR peak:** Typically strongest around 400-600 ms SOA
- **IOR duration:** Can persist for several seconds. Castel et al. (2003) found space-based IOR persists for at least 2 seconds. Samuel & Kat (2003) reported IOR lasting up to 3 seconds.

### 4.2 Factors Affecting Time Course

- **Task demands (Lupianez et al., 1997, 2001):** The facilitation-to-inhibition crossover point is NOT fixed. Detection tasks show IOR at ~250-300 ms SOA, while discrimination tasks show IOR at longer SOAs (often >400 ms). More difficult discrimination tasks further lengthen the SOA at which IOR appears. Adding a distractor to the opposite location *shortens* the SOA at which IOR emerges.
- **Three-factor model (Lupianez):** Performance is determined by: (1) spatial selection and (2) spatial orienting, which facilitate performance, and (3) a detection cost, which impairs it. The balance of these three factors determines when IOR appears.
- **Cue-target similarity:** Greater similarity between cue and target can modulate IOR onset
- **Attentional control settings:** Endogenous (voluntary) attention can delay or modulate IOR
- **Response modality:** Manual responses vs. saccadic responses show different time courses
- **EEG evidence:** ~180 ms after cue presentation, there is a significant increase in attention-related amplitude; after ~200 ms this enhancement disappears and turns into a significant inhibitory effect (SSVEP studies)

### 4.3 Neural Decay Estimates

From the dynamic field model (Ibanez-Gijon & Jacobs, 2012):
- Decision field time constant: tau_D = 0.328 s (fastest -- facilitation decays quickly)
- Sensory field time constant: tau_S = 0.048 s
- Habituation field time constant: tau_H = 1.620 s (slowest -- inhibition persists)
- Human manual responses: IOR extends to ~4 seconds
- Monkey saccadic responses: IOR extends to ~1 second

---

## 5. Spatial Properties

### 5.1 Spatial Extent

- IOR is not a point inhibition but has a spatial gradient
- The inhibited region extends around the cued location with a center-surround profile
- Studies have found IOR extends approximately 3-5 degrees of visual angle from the cued location (Bennett & Pratt, 2001)
- The inhibitory gradient decays with distance from the cued location
- **Temporal change in gradient (Samuel & Kat, 2003):** During the first second, inhibition magnitude is inversely related to distance from the original stimulus location. After ~1 second, this spatial gradient disappears and inhibition becomes more uniform across the affected region.

### 5.2 Capacity: Number of Inhibited Locations

- **Snyder & Kingstone (2000):** Demonstrated IOR at a minimum of 5 previously cued locations simultaneously. Inhibition magnitude is largest at the most recently searched location and declines approximately linearly with each earlier location.
- **Danziger, Kingstone, & Snyder (1998):** Found IOR can reside at 3 spatial locations simultaneously.
- **Wang & Klein (2010):** IOR lasts for at least 1000 ms or about 4 previously inspected items/locations during search.
- **Active attention requirement:** Multiple-location IOR is only observed when attention had to be actively directed to the cued locations; without active attention allocation, only the most recently cued location shows IOR.
- **Scene dependency (Takeda & Yagi, 2000):** Inhibitory tags are removed when the search array is removed before probe presentation, suggesting tags are stored with the scene representation, not as purely spatial coordinates.
- **Practical implication for ROC:** The system should maintain a limited-capacity buffer of recently attended locations (~4-5), each with a decaying inhibitory strength that is strongest for the most recent and weakest for the oldest.

---

## 6. Computational Models

### 6.1 Koch & Ullman (1985) -- Original Saliency Map Proposal

- **Paper:** Koch, C., & Ullman, S. (1985). "Shifts in selective visual attention: Towards the underlying neural circuitry." *Human Neurobiology*, 4, 219-227.
- **Proposal:** First proposed the concept of a "saliency map" -- a topographic map where each location's activation represents its conspicuity. A winner-take-all (WTA) network selects the most salient location, and IOR is implemented by suppressing the winning location's activity after selection.
- **IOR mechanism:** After the WTA network selects a location, that location's saliency is reduced, allowing the next-most-salient location to win. This creates a sequence of fixations ordered by decreasing saliency.

### 6.2 Itti, Koch & Niebur (1998) -- Computational Saliency Model

- **Paper:** Itti, L., Koch, C., & Niebur, E. (1998). "A model of saliency-based visual attention for rapid scene analysis." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(11), 1254-1259.
- **Architecture:**
  1. Visual input decomposed into feature maps (color, intensity, orientation) at 8 spatial scales
  2. Center-surround differences compute local contrast (42 feature maps total)
  3. Iterative lateral inhibition within each feature map
  4. Feature maps combined into a single "conspicuity map" per channel
  5. Conspicuity maps normalized and summed into a single **saliency map**
  6. Winner-take-all (WTA) network on the saliency map selects the most salient location

- **IOR implementation:**
  - After WTA selects a location, a **disc of inhibition** is applied to the saliency map at that location
  - This transiently suppresses the winning location's saliency value
  - The inhibition allows the next-most-salient location to win
  - Creates a sequential scan path through the scene in order of decreasing saliency

- **Subsequent refinement:** Itti & Koch (2000, 2001) further developed the model:
  - Itti, L., & Koch, C. (2000). "A saliency-based search mechanism for overt and covert shifts of visual attention." *Vision Research*, 40, 1489-1506.
  - Itti, L., & Koch, C. (2001). "Computational modelling of visual attention." *Nature Reviews Neuroscience*, 2, 194-203.

**Source:** [Itti et al. 1998 on IEEE Xplore](https://ieeexplore.ieee.org/document/730558/); [Itti & Koch 2000](http://ilab.usc.edu/publications/doc/Itti_Koch00vr.pdf); [Itti & Koch 2001](https://www.nature.com/articles/35058500)

### 6.3 Dynamic Neural Field Model (Ibanez-Gijon & Jacobs, 2012)

- **Paper:** Ibanez-Gijon, J., & Jacobs, D. M. (2012). "Decision, sensation, and habituation: A multi-layer dynamic field model for inhibition of return." *PLOS ONE*, 7(3), e33169.

- **Architecture:** Three coupled activation fields:
  1. **Decision Field D(x,t):** Inspired by intermediate superior colliculus; accumulates evidence for motor decisions; response triggered when activation reaches 80% of maximum
  2. **Sensory Field S(x,t):** Reflects early sensory processing; receives exogenous stimulus input; subject to habituation
  3. **Habituation Field H(x,t):** Implements activation-dependent habituation; reciprocally coupled with sensory field

- **Core dynamics:**
  - tau_D * dD/dt = -D + h_D + [lateral interactions] + sensory input + endogenous input
  - tau_S * dS/dt = -S + h_S + [lateral interactions] + exogenous input + endogenous input
  - tau_H * dH/dt = -H + h_H + [activity-dependent term]
  - Lateral interactions: Mexican-hat profile (excitatory center, sigma_a=4; inhibitory surround, sigma_b=7)

- **IOR mechanism:** IOR emerges from differential temporal dynamics across layers:
  - Short SOAs (0-75 ms): cue-induced pre-target activation in the decision field summates with target input, producing facilitation despite sensory habituation
  - Long SOAs (200-500 ms): pre-target decision activation has decayed rapidly, but habituation persists in the slow habituation field, reducing sensory response to the target

- **Fitted parameters:**
  - tau_D = 0.328 s (decision -- fastest decay)
  - tau_S = 0.048 s (sensory)
  - tau_H = 1.620 s (habituation -- slowest decay)
  - Perceptual delays: exogenous 70 ms, endogenous 120 ms, motor output 80 ms
  - Model fit: r = 0.95 with experimental data (p = 0.004), average absolute error = 8.0 ms

- **Neurophysiological grounding:** The decision field mirrors intermediate superior colliculus neuron activity; habituation mirrors sensory depression in superficial SC layers documented by Fecteau & Munoz.

**Source:** [Ibanez-Gijon & Jacobs 2012 in PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0033169)

### 6.4 Zhang et al. (2022) -- Return Fixation Model

- **Paper:** Zhang, M., et al. (2022). "Look twice: A generalist computational model predicts return fixations across tasks and species." *PLOS Computational Biology*, 18(11), e1010654.
- **IOR implementation:** Uses a time-dependent memory map (M_mem,t) updated based on all previous fixation locations, with an approximately exponential memory decay function. The memory map combines linearly with saliency map, target similarity map, and saccade constraint maps to produce a final attention map.
- **Key finding:** Even with finite IOR, return fixations are ubiquitous -- 44,328 out of 217,440 fixations (~20%) were returns, across monkeys and humans in static and natural tasks. This challenges the assumption that IOR should prevent return fixations entirely; finite IOR with exponential decay is the empirically correct model.
- **Relevance to ROC:** Suggests IOR should be implemented with finite, decaying inhibition rather than absolute suppression. Some revisitation is normal and useful.

**Source:** [Zhang et al. 2022 in PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010654)

### 6.5 Satel et al. (2011) -- Single-Layer Habituation Model

- **Reference:** Satel, J., Wang, Z., Hilchey, M. D., & Klein, R. M. (2013). "Adaptive learning of the tonic and phasic components of spatial IOR." Paper presented at OPAM/Psychonomics.
- **Approach:** Applied habituation as an ad-hoc modification of sensory input in a single-layer model
- **Limitation:** Could not explain phenomena arising from interactions between processes at different time scales (addressed by the multi-layer model of Ibanez-Gijon & Jacobs)

### 6.6 SceneWalk Model (Engbert et al., 2015)

- **Paper:** Engbert, R., Trukenbrod, H. A., Barthelme, S., & Wichmann, F. A. (2015). "Spatial statistics and attentional dynamics in scene viewing." *Journal of Vision* / *PLOS Computational Biology*.
- **Architecture:** Two-stream dynamical model with separate attention and inhibition maps:
  - Attention map: `mapAtt = salFixation + exp(-duration * omega_Att) * (prevMapAtt - salFixation)`
  - Inhibition map: `mapInhib = gaussInhib + exp(-duration * omega_Inhib) * (prevMapInhib - gaussInhib)`
  - Combined priority: `u = mapAtt^lambda - inhibStrength * mapInhib^gamma`
  - Final with noise: `u_final = (1 - zeta) * u_star + zeta / mapSize`
- **Key parameters:**
  - omega_Attention = 10 (attention decay rate -- fast)
  - omega_Inhib = 1.97 (inhibition decay rate -- slower, so inhibition outlasts attention)
  - sigma_Attention = 5.9 pixels (spatial width of attention)
  - sigma_Inhib = 4.5 pixels (spatial width of inhibition Gaussian, ~4.88 deg visual angle)
  - inhibStrength = 0.3637 (inhibition weight)
  - lambda = 0.8115 (attention exponent), gamma = 1 (inhibition exponent)
  - zeta = 0.0722 (noise parameter)
- **IOR mechanism:** Inhibition decays slower than attention (omega_Inhib < omega_Att), so after attending a location, the inhibitory trace persists longer than the attentional boost. This is analogous to the Ibanez-Gijon model's differential time constants.

**Source:** [SceneWalk documentation](http://lisaschwetlick.de/SceneWalk_Model/demo/detailed_look_at_sw.html); [PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007880)

### 6.7 Bayesian / Active Inference Models

#### Parr & Friston (2017) -- Epistemic Foraging

- **Paper:** Parr, T., & Friston, K. (2017). "Uncertainty, epistemics and active inference." *Journal of The Royal Society Interface*, 14(136).
- **Key insight:** IOR-like behavior **emerges naturally** from the active inference framework without being explicitly programmed. Recently observed locations have lower epistemic value (information gain) that gradually increases as the probability that the hidden state has changed increases.
- **Volatility modulation:** Greater environmental volatility leads to **shorter IOR** -- in volatile environments, recent observations lose informational value faster, so the system revisits locations sooner. This provides a principled, information-theoretic explanation for IOR as optimal epistemic behavior.

**Source:** [Parr & Friston 2017](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0376)

#### Bayesian Surprise (Itti & Baldi, 2006/2009)

- **Paper:** Itti, L., & Baldi, P. (2009). "Bayesian surprise attracts human attention." *Vision Research*, 49(10), 1295-1306.
- Surprise = KL divergence between posterior and prior beliefs. Among 11 computational metrics, surprise best predicted human gaze. 72% of saccades were directed toward regions more surprising than average.
- IOR emerges implicitly: already-viewed locations have reduced surprise because the posterior has been updated.

**Source:** [Itti & Baldi 2009](http://ilab.usc.edu/publications/doc/Itti_Baldi09vr.pdf)

### 6.8 Adaptive IOR -- Anderson et al. (2010)

- **Paper:** Anderson, A. J., Yadav, H., & Carpenter, R. H. S. (2010). "Influence of environmental statistics on inhibition of saccadic return." *PNAS*, 107(2), 929-934.
- **Key finding:** IOR is **abolished** when return locations are statistically likely. Three conditions tested:
  - Low return probability (1/6): criterion = 1.18, strongest inhibition
  - Equal (1/3): criterion = 1.03, moderate inhibition
  - High (1/2): criterion = 0.88, minimal/no inhibition
- **Dual mechanisms identified via Linear Ballistic Accumulator model:**
  1. A **fixed heuristic:** evidence is always accumulated at a slower rate for return locations (alpha ~0.75, constant across conditions)
  2. A **plastic adaptive system:** evidence threshold varies systematically with statistical context
- **Relevance to ROC:** IOR strength should potentially adapt to game context -- environments with frequent meaningful changes at the same location should have weaker IOR.

**Source:** [Anderson et al. 2010 in PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC2818969/)

### 6.9 Deep Learning Saliency Models

#### DeepGaze III (Kummerer et al., 2022)

- **Paper:** Kummerer, M., et al. (2022). "DeepGaze III: Modeling free-viewing human scanpaths with deep learning." *Journal of Vision*, 22(5).
- **Architecture:** DenseNet-201 spatial priority network + scanpath history network encoding last 4 fixations (Euclidean distance, x/y differences) through 1x1 convolutions.
- **IOR finding:** "Effects like (spatial) inhibition of return or excitation of return are already completely decayed after two fixations" -- contradicting traditional models with longer-lasting IOR. The last 2 fixations are most relevant for scanpath dynamics.
- Performance: 2.442 bit/fix log-likelihood on MIT1003.

#### Deep Convolutional Saccadic Model (DCSM, Bao et al., 2020)

- **Paper:** Bao, Y., Chen, Z., et al. (2020). "Human scanpath prediction based on deep convolutional saccadic model." *Neurocomputing*.
- IOR is content-dependent rather than a simple Gaussian disc: associated with image content from both spatial and temporal aspects. Both foveal saliency and fixation durations are predicted by CNNs.

**Source:** [DeepGaze III in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9055565/); [DCSM on ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231220304331)

### 6.10 Priority Map (Bisley & Goldberg, 2010)

- **Paper:** Bisley, J. W., & Goldberg, M. E. (2010). "Attention, intention, and priority in the parietal lobe." *Annual Review of Neuroscience*, 33, 1-21.
- **Concept:** LIP (lateral intraparietal area) acts as a **priority map** combining bottom-up saliency with top-down goal signals. "Priority" is deliberately chosen over "saliency" because both bottom-up and top-down influences are represented.
- **IOR mechanism:** When potential targets are checked and found non-rewarding, their neural representation is reduced -- functioning as short-term spatial memory. Planning a saccade toward one location enhances its priority while **suppressing** LIP neurons representing other locations.

**Source:** [Bisley & Goldberg 2010 in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3683564/)

### 6.11 Other Computational Approaches

- **NAVIS system:** Uses dynamic neural fields with separate local and global inhibition
- **HMAX model:** Implements object-based (not spatial) inhibition
- **Information-theoretic models (e.g., Schill model):** Use Dempster-Shafer belief theory where IOR emerges implicitly because information gain at previously fixated locations is automatically reduced
- **Wang et al. (2013) collicular model:** Hybrid model combining short-term depression with delayed direct collicular inhibition (arXiv:1307.5684)

---

## 7. Neural Substrates

### 7.1 Superior Colliculus (SC)

The superior colliculus is the primary neural substrate implicated in IOR:
- **Superficial layers:** Show sensory depression (habituation) to repeated stimuli -- directly related to input-based IOR
- **Intermediate layers:** Generate saccade commands; involved in output-based IOR
- **Key studies:** Dorris et al. (2002); Fecteau & Munoz (2006) demonstrated reduced neural responses in SC neurons for targets at previously cued locations in monkey studies
- **Lesion evidence:** Patients with SC damage show reduced or absent IOR (Sapir et al., 1999)

### 7.2 Oculomotor System

- IOR is closely linked to the oculomotor system, even when no eye movements are made
- Preparing a saccade to a location and then canceling it still produces IOR at that location (Rafal et al., 1989)
- The reflexive oculomotor system's activation state determines which form of IOR (input vs. output) is generated

### 7.3 Cortical Involvement

- Posterior parietal cortex: involved in maintaining spatial representations that support IOR
- Frontal eye fields: contribute to voluntary control that modulates IOR
- IOR may involve a network of structures rather than a single neural locus

---

## 8. IOR in Visual Search

### 8.1 Evidence For IOR in Search

- Klein (1988): First demonstrated IOR at rejected distractor locations during search
- Klein & MacInnes (1999): "Inhibition of return is a foraging facilitator in visual search." Showed IOR at previously fixated locations during free-viewing search using "Where's Wally?" images with eye tracking.
- Muller & von Muhlenen (2000): Found IOR at previously searched distractor locations, but only when the search display remained visible. When search objects were removed after the search response, no IOR was found. This implies IOR in search requires a stable scene as a spatial framework.

### 8.2 Wang & Klein (2010) -- Review of ~15 Studies

- **Paper:** Wang, Z. & Klein, R. M. (2010). "Searching for inhibition of return in visual search: A review." *Vision Research*, 50(2), 220-228.
- Reviewed ~15 studies published in the 20 years following Klein (1988)
- IOR is consistently observed when the search display remains visible during probing
- IOR lasts for at least 1000 ms or about 4 previously inspected items/locations during search
- IOR was found in static and slower dynamic search but **not** in faster dynamic search
- Visibility of the search array at probe time is critical

**Source:** [Wang & Klein 2010 in Vision Research](https://www.sciencedirect.com/science/article/pii/S0042698909005252)

### 8.3 Evidence Against / Complications

- Horowitz & Wolfe (1998, 2001): Argued search has "no memory" -- randomizing display during search does not impair performance, challenging the idea that inhibitory tags guide search
- Search in dynamic displays: IOR may not operate the same way in constantly changing environments (relevant to ROC's game environment where the screen updates each frame)
- Zhang et al. (2022): ~20% of all fixations are return fixations across species and tasks, suggesting IOR does not prevent returns but merely biases against them
- The debate remains partially unresolved, with evidence on both sides

### 8.3 Recent Synthesis: Klein, Redden & Hilchey (2023)

- **Paper:** Klein, R. M., Redden, R. S., & Hilchey, M. D. (2023). "Visual search and the inhibitions of return." *Frontiers in Cognition*, 2, 1146511.
- **Key contribution:** Argues both forms of IOR (input and output) can serve the novelty-seeking and search-facilitating function
- **Input-based IOR in search:** Biases attention away from already-inspected locations by reducing their salience
- **Output-based IOR in search:** Biases responses away from already-inspected locations at the decision level
- Concludes: "Both forms of IOR can serve the novelty seeking (and search facilitating) function"

**Source:** [Klein et al. 2023 in Frontiers](https://www.frontiersin.org/journals/cognition/articles/10.3389/fcogn.2023.1146511/full)

---

## 9. Comparative Analysis of Theories

### 9.1 Theory Comparison Matrix

| Dimension | Foraging Facilitator | Habituation Account | Input/Output Theory | Object-Based IOR |
|---|---|---|---|---|
| **Core claim** | IOR is a search optimization mechanism | IOR is a byproduct of sensory habituation | Two distinct mechanisms exist | IOR attaches to object representations |
| **Mechanism** | Inhibitory tags on inspected locations | Reduced orienting response to repeated stimuli | Salience reduction (input) vs. response bias (output) | Inhibition follows object identity |
| **Level** | Functional/computational | Mechanistic/neural | Information processing | Representational |
| **Relation to search** | Primary function | Side effect that happens to aid search | Both forms aid search | Uncertain search utility |
| **Neural basis** | Unspecified | Superior colliculus habituation | SC superficial (input) vs. intermediate (output) layers | Requires object tracking circuitry |
| **Testable prediction** | IOR at rejected distractors in search | IOR should follow habituation laws (stimulus specificity, dishabituation) | Dissociation by oculomotor state | IOR follows moving objects |

### 9.2 Points of Agreement

All major theories agree that:
1. IOR produces slower responses to previously attended locations
2. IOR emerges after an initial facilitation period (~200-300 ms)
3. IOR has a functional consequence of biasing attention toward novelty
4. IOR involves the superior colliculus

### 9.3 Points of Disagreement

1. **Is IOR a purpose-built mechanism or a side effect?** Foraging facilitator says purpose-built; habituation account says side effect.
2. **Is IOR unitary or dual?** Input/output theory says two distinct mechanisms; habituation account and foraging facilitator originally assumed a single mechanism.
3. **Does IOR primarily affect perception or action?** Input-based theory says perception; output-based says action; the current consensus is both, depending on conditions.
4. **How many locations can be simultaneously inhibited?** Estimates range from 3-5 (limited capacity like visual working memory) to potentially more.

---

## 10. Implications for ROC Implementation

### 10.1 Relevant Design Considerations

Given ROC's architecture (SaliencyMap -> VisionAttention -> ObjectResolver), IOR should:

1. **Operate on the saliency map** (input-based IOR): Reduce saliency at recently attended locations so the next attention cycle naturally selects novel locations. This aligns with the Itti-Koch model and the input-based IOR mechanism.

2. **Maintain a limited-capacity inhibition buffer:** Track the last N attended locations (empirically supported range: 3-5 locations) with decaying inhibitory strength.

3. **Implement temporal decay:** Inhibition should decay over time/frames, following an exponential or similar decay curve. The habituation field time constant of ~1.6 seconds from Ibanez-Gijon & Jacobs (2012) provides a neurophysiologically grounded reference.

4. **Use spatial gradients:** Inhibition should have a spatial extent (not just point inhibition), with strongest suppression at the attended location and decreasing inhibition with distance.

5. **Consider frame-based rather than time-based decay:** Since ROC processes discrete game frames rather than continuous time, the decay could be measured in frames/ticks rather than milliseconds.

### 10.2 Simplest Viable Approach (Itti-Koch Style)

The most straightforward implementation follows the Itti-Koch model:
- After selecting a focus point from the saliency map, subtract a Gaussian-shaped inhibitory disc centered on that location
- The disc decays over subsequent frames
- Multiple discs can be maintained simultaneously (one per recently attended location)

### 10.3 More Sophisticated Approach (Dynamic Field)

For a more neurophysiologically grounded implementation:
- Maintain a separate inhibition/habituation map that evolves independently from the saliency map
- The inhibition map builds up through attention and decays slowly
- The effective saliency at each location = raw saliency * (1 - inhibition)
- This naturally produces both facilitation (at short intervals) and inhibition (at longer intervals)

### 10.4 Bayesian/Information-Theoretic Approach

For a principled approach aligned with Parr & Friston (2017) and Itti & Baldi (2009):
- IOR emerges implicitly from tracking information gain at each location
- Locations that were recently observed have lower expected information gain
- Information gain recovers naturally over time as uncertainty about the location's state increases
- In volatile environments (frequent game state changes), IOR would be shorter -- locations regain informational value faster
- This approach requires no explicit inhibition mechanism; IOR is a natural consequence of optimal epistemic behavior

### 10.5 Adaptive IOR (Anderson et al. 2010 Inspired)

IOR strength could adapt to game context:
- Locations where meaningful changes frequently occur should have faster IOR decay
- Static regions (walls, empty floor) should have longer-lasting IOR
- This could be implemented by modulating the decay rate based on historical change frequency at each location

### 10.6 Object-Based Considerations

Since ROC already resolves objects, object-based IOR could be implemented by:
- Tagging resolved objects with inhibitory values (not just locations)
- If an object moves between frames, its inhibition follows it
- However, the empirical evidence suggests object-based IOR is fragile and has smaller effects than space-based IOR, so location-based IOR should be the primary mechanism.

---

## 11. Summary of Quantitative Parameters

| Parameter | Value | Source |
|---|---|---|
| **Behavioral** | | |
| Typical IOR magnitude (simple cueing) | ~20-25 ms RT slowing | Multiple studies |
| Space-based IOR magnitude | ~29-38 ms | List & Robertson 2007 |
| Object-based IOR magnitude | ~6-14 ms (fragile) | List & Robertson 2007 |
| Facilitation-to-IOR transition (detection) | ~250-300 ms SOA | Posner & Cohen 1984 |
| Facilitation-to-IOR transition (discrimination) | >400 ms SOA | Lupianez et al. 1997 |
| IOR stability window | 300-1600 ms SOA | Samuel & Kat 2003 |
| IOR maximum duration | ~3-4 seconds | Samuel & Kat 2003; Ibanez-Gijon & Jacobs 2012 |
| IOR during search | ~1000 ms or ~4 items | Wang & Klein 2010 |
| Return fixation frequency (with IOR) | ~20% of all fixations | Zhang et al. 2022 |
| IOR decay in free viewing | ~2 fixations | DeepGaze III, Kummerer et al. 2022 |
| **Spatial** | | |
| Spatial extent | ~3-5 degrees visual angle | Bennett & Pratt 2001 |
| Spatial gradient slope (single cue) | -0.140 ms/degree | Samuel & Kat 2003 |
| Spatial gradient slope (multiple cues) | -0.087 ms/degree | Samuel & Kat 2003 |
| Spatial gradient duration | ~1 second before becoming uniform | Samuel & Kat 2003 |
| Simultaneous inhibited locations | At least 5, linear decline | Snyder & Kingstone 2000 |
| **Neural** | | |
| EEG facilitation onset | ~170-180 ms post-cue | SSVEP studies |
| EEG inhibition onset | ~200 ms post-cue | SSVEP studies |
| **Model Parameters (Ibanez-Gijon & Jacobs 2012)** | | |
| Decision field time constant | tau_D = 0.328 s | Fastest decay |
| Sensory field time constant | tau_S = 0.048 s | |
| Habituation field time constant | tau_H = 1.620 s | Slowest decay -- IOR persists |
| **Model Parameters (SceneWalk, Engbert et al. 2015)** | | |
| Attention decay rate | omega_Att = 10 | Fast decay |
| Inhibition decay rate | omega_Inhib = 1.97 | Slow decay (~5x slower) |
| Inhibition Gaussian width | sigma = 4.5 px (~4.88 deg) | |
| Inhibition strength weight | 0.3637 | |
| **Adaptive IOR (Anderson et al. 2010)** | | |
| Fixed rate reduction for returns | alpha ~0.75 (constant) | Accumulation rate scaling |
| Threshold at low return prob (1/6) | 1.18 | Strongest IOR |
| Threshold at high return prob (1/2) | 0.88 | IOR abolished |

---

## 12. Computational Models Summary

| Model | Year | IOR Mechanism | Key Innovation |
|---|---|---|---|
| Koch & Ullman | 1985 | Suppression at selected location in saliency map | Original saliency map proposal |
| Itti, Koch, Niebur | 1998 | I&F neuron WTA + transient suppression disk | First computational implementation |
| Itti & Koch | 2000 | Same + DoG kernel lateral interactions | Extended to overt/covert shifts |
| Bayesian Surprise | 2006 | KL divergence reduction at visited locations | Information-theoretic IOR |
| Anderson et al. | 2010 | LBA with adaptive threshold | IOR abolished by environmental statistics |
| Bisley & Goldberg | 2010 | Activity reduction in LIP priority map | Neurophysiological priority map |
| Ibanez-Gijon & Jacobs | 2012 | Three-layer neural field with habituation | Explains biphasic facilitation/IOR |
| SceneWalk | 2015 | Dual-stream ODE (attention + inhibition maps) | Exponential decay, separate time constants |
| Parr & Friston | 2017 | Emergent from epistemic value computation | Volatility modulates IOR duration |
| DCSM | 2020 | Content-dependent CNN IOR | Deep learning foveal saliency |
| Zhang et al. | 2022 | Memory map with exponential decay | ~20% return fixations are normal |
| DeepGaze III | 2022 | Learned scanpath history (4 fixations) | IOR decays after ~2 fixations |

---

## 13. Key References

### Foundational
- Posner, M. I., & Cohen, Y. (1984). Components of visual orienting. *Attention and Performance X*, 531-556.
- Posner, M. I., Rafal, R. D., Choate, L. S., & Vaughan, J. (1985). Inhibition of return: Neural basis and function. *Cognitive Neuropsychology*, 2, 211-228.
- Klein, R. M. (1988). Inhibitory tagging system facilitates visual search. *Nature*, 334, 430-431.

### Major Reviews
- Klein, R. M. (2000). Inhibition of return. *Trends in Cognitive Sciences*, 4(4), 138-147.
- Klein, R. M., Redden, R. S., & Hilchey, M. D. (2023). Visual search and the inhibitions of return. *Frontiers in Cognition*, 2, 1146511.
- Redden, R. S., MacInnes, W. J., & Klein, R. M. (2021). Inhibition of return: An information processing theory of its natures and significance. *Cortex*, 135, 30-48.
- Lupianez, J., Klein, R. M., & Bartolomeo, P. (2006). Inhibition of return: Twenty years after. *Cognitive Neuropsychology*, 23, 1003-1014.

### Input/Output Distinction
- Taylor, T. L., & Klein, R. M. (2000). Visual and motor effects in inhibition of return. *Journal of Experimental Psychology: Human Perception and Performance*, 26, 1639-1656.

### Object-Based IOR
- Tipper, S. P., Driver, J., & Weaver, B. (1991). Object-centred inhibition of return of visual attention. *Quarterly Journal of Experimental Psychology*, 43A, 289-298.
- List, A., & Robertson, L. C. (2007). Inhibition of return and object-based attentional selection. *Journal of Experimental Psychology: Human Perception and Performance*, 33(6), 1322-1334.

### Habituation Account
- Dukewich, K. R. (2009). Reconceptualizing inhibition of return as habituation of the orienting response. *Psychonomic Bulletin & Review*, 16(2), 238-251.

### Computational Models
- Koch, C., & Ullman, S. (1985). Shifts in selective visual attention. *Human Neurobiology*, 4, 219-227.
- Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. *IEEE TPAMI*, 20(11), 1254-1259.
- Itti, L., & Koch, C. (2000). A saliency-based search mechanism for overt and covert shifts of visual attention. *Vision Research*, 40, 1489-1506.
- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. *Nature Reviews Neuroscience*, 2, 194-203.
- Ibanez-Gijon, J., & Jacobs, D. M. (2012). Decision, sensation, and habituation: A multi-layer dynamic field model for inhibition of return. *PLOS ONE*, 7(3), e33169.

### Visual Search Debate
- Horowitz, T. S., & Wolfe, J. M. (1998). Visual search has no memory. *Nature*, 394, 575-577.
- Klein, R. M., & MacInnes, W. J. (1999). Inhibition of return is a foraging facilitator in visual search. *Psychological Science*, 10, 346-352.
- Muller, H. J., & von Muhlenen, A. (2000). Probing distractor inhibition in visual search. *Journal of Experimental Psychology: Human Perception and Performance*, 26, 1591-1605.
- Wang, Z., & Klein, R. M. (2010). Searching for inhibition of return in visual search: A review. *Vision Research*, 50(2), 220-228.

### Neural Substrates
- Dorris, M. C., Klein, R. M., Everling, S., & Munoz, D. P. (2002). Contribution of the primate superior colliculus to inhibition of return. *Journal of Cognitive Neuroscience*, 14, 1256-1263.
- Fecteau, J. H., & Munoz, D. P. (2006). Salience, relevance, and firing: a priority map for target selection. *Trends in Cognitive Sciences*, 10, 382-390.

### Multiple Locations
- Snyder, J. J., & Kingstone, A. (2000). Inhibition of return and visual search. *Perception & Psychophysics*, 62, 1232-1243.
- Danziger, S., Kingstone, A., & Snyder, J. J. (1998). Inhibition of return to successively stimulated locations in a sequential visual search paradigm. *Journal of Experimental Psychology: Human Perception and Performance*, 24, 1467-1475.

### Time Course
- Lupianez, J., Milan, E. G., Tornay, F. J., Madrid, E., & Tudela, P. (1997). Does IOR occur in discrimination tasks? Yes, it does, but later. *Perception & Psychophysics*, 59, 1241-1254.
- Lupianez, J., et al. (2001). On the strategic modulation of the time course of facilitation and inhibition of return. *Quarterly Journal of Experimental Psychology A*, 54(3), 753-773.
- Samuel, A. G., & Kat, D. (2003). Inhibition of return: A graphical meta-analysis of its time course. *Psychonomic Bulletin & Review*, 10, 897-906.

### Return Fixation Models
- Zhang, M., et al. (2022). Look twice: A generalist computational model predicts return fixations across tasks and species. *PLOS Computational Biology*, 18(11), e1010654.

### Bayesian / Active Inference
- Parr, T., & Friston, K. (2017). Uncertainty, epistemics and active inference. *Journal of The Royal Society Interface*, 14(136).
- Itti, L., & Baldi, P. (2009). Bayesian surprise attracts human attention. *Vision Research*, 49(10), 1295-1306.
- Anderson, A. J., Yadav, H., & Carpenter, R. H. S. (2010). Influence of environmental statistics on inhibition of saccadic return. *PNAS*, 107(2), 929-934.

### Scanpath / Deep Learning Models
- Engbert, R., Trukenbrod, H. A., Barthelme, S., & Wichmann, F. A. (2015). Spatial statistics and attentional dynamics in scene viewing. *PLOS Computational Biology*.
- Kummerer, M., et al. (2022). DeepGaze III: Modeling free-viewing human scanpaths with deep learning. *Journal of Vision*, 22(5).
- Bao, Y., Chen, Z., et al. (2020). Human scanpath prediction based on deep convolutional saccadic model. *Neurocomputing*.

### Priority Map
- Bisley, J. W., & Goldberg, M. E. (2010). Attention, intention, and priority in the parietal lobe. *Annual Review of Neuroscience*, 33, 1-21.

### Salience Model Reviews
- Krasovskaya, S., & MacInnes, W. J. (2019). Salience models: A computational cognitive neuroscience review. *Vision*, 3(4), 56.
