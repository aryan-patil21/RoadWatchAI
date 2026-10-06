# RoadWatch AI - Dataset & Annotation Specification

## 1. Overview
Road safety risk models require rigorous, objective training datasets.
In subjective manual labelling, human annotators often disagree on what looks "dangerous" or "aggressive".
To eliminate subjectivity and prevent demographic or vehicular stereotypes, RoadWatch AI establishes an **objective kinematic taxonomy**.

Every sample in the RoadWatch dataset is tagged with:
1. **Kinematic Safety Level**: `NORMAL` (LOW), `CAUTION` (MEDIUM), or `HIGH_RISK` (HIGH).
2. **Specific Behavioural Event**: The primary observable physical manoeuvre.

---

## 2. Objective Event Taxonomy & Physical Criteria

| Event Label | Physical Kinematic Definition | Target Risk Level |
| :--- | :--- | :--- |
| **`normal_driving`** | Approach rate $\le 0.05$/s, proximity $< 0.40$, lateral speed $|v_{\text{lat}}| < 25$ px/s, no lane cutting. | `NORMAL` |
| **`close_following`** (Tailgate) | Proximity $\ge 0.55$, stable approach rate ($-0.05 \le \text{rate} \le 0.05$), duration $\ge 10$ consecutive frames. | `CAUTION` |
| **`rapid_approach`** | Approach rate $> 0.12$/s, closing speed $v_{\text{long}} > 35$ px/s, $TTC \le 4.0$s. | `HIGH_RISK` |
| **`unsafe_lane_change`** (Cutting) | Lateral velocity $|v_{\text{lat}}| > 35$ px/s shifting across the ego-corridor boundary with proximity $\ge 0.40$. | `HIGH_RISK` |
| **`sudden_braking`** | Longitudinal acceleration proxy $a_{\text{long}} < -100$ px/s$^2$ with proximity $\ge 0.35$. | `HIGH_RISK` |
| **`parallel_cruising`** | Vehicle in adjacent lane (`LEFT_ZONE` or `RIGHT_ZONE`) maintaining speed without lateral drift towards rider. | `NORMAL` |

---

## 3. Data Leakage Prevention (Crucial ML Principle)

### What is Data Leakage in Video?
If consecutive video frames from the same vehicle (e.g. Vehicle #1 at frame 40 and frame 41) are split randomly between the training set and the test set:
- Frame 40 and frame 41 are $99\%$ identical.
- The test set is contaminated with exact copies of training examples.
- The machine learning model appears to achieve $100\%$ accuracy during evaluation, but completely fails in real life!

### How RoadWatch AI Prevents Data Leakage:
1. **Entity-Group Splitting (`GroupShuffleSplit` on `vehicle_id`)**:
   All observations belonging to a specific tracked vehicle must reside **exclusively in the Training set or exclusively in the Test set**.
2. **Temporal Segment Partitioning**:
   Continuous video events are blocked into time chunks rather than random frame shuffle.

---

## 4. Dataset Splits
- **Training Set (70%)**: Used to fit model weights.
- **Validation Set (15%)**: Used to tune decision thresholds and hyperparameters.
- **Test Set (15%)**: Strictly held out to evaluate true generalization on unseen vehicles.
