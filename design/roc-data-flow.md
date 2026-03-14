# ROC Data Flow: Perception through Deltas and Keyframes

```mermaid
flowchart TD
    subgraph Gymnasium["Gymnasium (Main Loop)"]
        ENV["NetHack Environment"]
        ENV -->|"obs = env.step(action)"| SEND["send_obs()"]
    end

    subgraph PerceptionBus["Perception Bus"]
        SEND -->|VisionData| PB(("EventBus\n[PerceptionData]"))
        SEND -->|AuditoryData| PB
        SEND -.->|"ProprioceptiveData\n(currently unused)"| PB

        subgraph VisualExtractors["Visual Feature Extractors"]
            FE_SINGLE["single\n+10 saliency"]
            FE_DELTA["delta\n+15 saliency"]
            FE_MOTION["motion\n+20 saliency"]
            FE_COLOR["color"]
            FE_SHAPE["shape"]
            FE_FLOOD["flood"]
            FE_LINE["line"]
            FE_DIST["distance"]
        end

        subgraph AuditoryExtractors["Auditory Feature Extractors"]
            FE_PHON["phoneme\n(gruut text-to-phoneme)"]
        end

        PB -->|VisionData| FE_SINGLE
        PB -->|VisionData| FE_DELTA
        PB -->|VisionData| FE_MOTION
        PB -->|VisionData| FE_SHAPE
        PB -->|VisionData| FE_FLOOD
        PB -->|VisionData| FE_LINE
        PB -->|VisionData| FE_DIST
        PB -->|"VisionData +\nSingleFeature"| FE_COLOR
        PB -->|AuditoryData| FE_PHON

        FE_SINGLE -->|"VisualFeature + Settled"| PB
        FE_SINGLE -.->|"SingleFeature"| FE_COLOR
        FE_DELTA -->|"VisualFeature + Settled"| PB
        FE_MOTION -->|"VisualFeature + Settled"| PB
        FE_COLOR -->|"VisualFeature + Settled"| PB
        FE_SHAPE -->|"VisualFeature + Settled"| PB
        FE_FLOOD -->|"VisualFeature + Settled"| PB
        FE_LINE -->|"VisualFeature + Settled"| PB
        FE_DIST -->|"VisualFeature + Settled"| PB
        FE_PHON -->|"PhonemeFeature + Settled"| PB
    end

    subgraph IntrinsicPath["Intrinsic Path"]
        ENV -->|"bottom-line stats"| IB(("EventBus\n[IntrinsicData]"))
        IB --> INTR["Intrinsic Component"]
        INTR -->|"normalize via IntrinsicOps\n(Percent, Int, Map, Bool)"| INODES["IntrinsicNodes\n(Transformable)"]
    end

    subgraph Significance["Significance"]
        IB -->|"IntrinsicData"| SIG["Significance Component"]
        SIG --> SIGB(("EventBus\n[Significance]"))
    end

    subgraph Attention["Attention"]
        PB -->|"VisionData +\nVisualFeatures +\nSettled"| VA["VisionAttention"]
        VA -->|"accumulate into"| SM["SaliencyMap\n(grid of features)"]
        SM -->|"peak detection"| FP["Focus Points\n(x, y, strength)"]
        FP --> AB(("EventBus\n[AttentionData]"))
    end

    subgraph ObjectResolution["Object Resolution"]
        AB --> OR["ObjectResolver"]
        OR -->|"1. features at focus point\n(highest saliency only)"| FG["FeatureGroup"]
        OR -->|"2. search DB for candidates"| CAND["Candidate Objects\n(distance metric)"]
        CAND -->|"3. match or create"| OBJ["Object"]
        OBJ --> OB(("EventBus\n[ObjectData]"))
    end

    subgraph ActionBus["Action Bus"]
        ACTB(("EventBus\n[Action]\n(cache_depth=10)"))
        ACTION["Action Component"]
        ACTB -->|"ActionRequest"| ACTION
        ACTION -->|"TakeAction"| ACTB
        GYM_ACT["Gymnasium"] -->|"ActionRequest"| ACTB
        ACTB -->|"TakeAction\n(from cache)"| GYM_ACT
    end

    subgraph Sequencer["Sequencer (Keyframes)"]
        OB -->|"ResolvedObject"| SEQ["Sequencer"]
        INODES -->|"IntrinsicNodes"| SEQ
        ACTB -->|"TakeAction"| SEQ
        SEQ -->|"FrameAttribute edges"| FRAME["Current Frame\n(tick=N)"]

        SEQ -->|"frame complete\n(triggered by TakeAction)"| EMIT_FRAME["Completed Frame"]
        EMIT_FRAME -->|"NextFrame edge"| NEW_FRAME["New Frame\n(tick=N+1)"]
        EMIT_FRAME --> SB(("EventBus\n[Frame]"))
    end

    subgraph Transformer["Transformer (Deltas)"]
        SB --> TX["Transformer"]
        TX -->|"get prev frame\nvia NextFrame edge"| COMPARE["Compare Transformables\nsame_transform_type()"]
        COMPARE -->|"create_transform()"| DELTA["Transform Container"]
        DELTA --> IT["IntrinsicTransform\n(e.g. hp: -0.29)"]
        DELTA --> TB(("EventBus\n[TransformResult]"))
    end

    subgraph Predict["Predict"]
        TB -->|"TransformResult"| PRED["Predict Component"]
        PRED -->|"merge_transforms()\npredicted frame"| PREDB(("EventBus\n[Predict]"))
    end

    subgraph GraphDB["Graph DB (Memgraph)"]
        direction LR
        G_OBJ["Object"] -->|Features| G_FG["FeatureGroup"]
        G_FG -->|Detail| G_FN["FeatureNode"]
        G_F1["Frame N"] -->|NextFrame| G_F2["Frame N+1"]
        G_F1 -->|FrameAttribute| G_ATTR["Objects, Intrinsics, Action"]
        G_F1 -->|Change| G_TX["Transform"]
        G_TX -->|Change| G_F2
    end

    style PB fill:#4a9eff,color:#fff
    style IB fill:#4a9eff,color:#fff
    style AB fill:#4a9eff,color:#fff
    style OB fill:#4a9eff,color:#fff
    style SB fill:#4a9eff,color:#fff
    style TB fill:#4a9eff,color:#fff
    style ACTB fill:#4a9eff,color:#fff
    style SIGB fill:#4a9eff,color:#fff
    style PREDB fill:#4a9eff,color:#fff
    style FRAME fill:#2ecc71,color:#fff
    style EMIT_FRAME fill:#2ecc71,color:#fff
    style NEW_FRAME fill:#2ecc71,color:#fff
    style DELTA fill:#e74c3c,color:#fff
    style IT fill:#e74c3c,color:#fff
    style VisualExtractors fill:#f0f0f0,stroke:#999
    style AuditoryExtractors fill:#fff3e0,stroke:#e65100
```

**Color Key:**
- Blue circles -- EventBus channels (typed reactive streams)
- Green -- Keyframes (complete state snapshots)
- Red -- Deltas/Transforms (changes between frames)
- Gray box -- Visual feature extractors (process VisionData)
- Orange box -- Auditory feature extractors (process AuditoryData)

**Notes:**
- ProprioceptiveData is sent on the Perception bus but no component currently consumes it (dashed line)
- Color extractor depends on Single extractor output in addition to VisionData (dashed dependency)
- Action bus uses cache_depth=10 so Gymnasium can retrieve TakeAction responses
- CrossModalAttention is auto-loaded but incomplete (not shown)
- ObjectResolver currently only processes the single highest-saliency focus point
