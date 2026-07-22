# Road Intelligence — Pedestrian Accident Prediction and Prevention Using LLM

**System Name:** Pedestrian Road Intelligence & Safety Simulation Platform  
**Document Version:** 1.0.0  
**Target Domain:** Urban Safety Engineering, Predictive Risk Modeling, Prevention Policy Simulation  
**Case Study Scope:** Intersections & Road Networks in Chicago, IL  

---

## 1. Executive Summary & Architectural Overview

The **Chicago Pedestrian Road Intelligence & Safety Simulation Platform** is an end-to-end machine learning and policy simulation architecture designed to proactively mitigate pedestrian crash risk. Traditional crash prediction architectures rely predominantly on historical incident reports, which record *where* and *when* accidents occurred but fail to model the *built infrastructure context* that exacerbates risk.

This platform bridges that critical gap by coupling **1M+ historical crash records** from the City of Chicago with **street-level computer vision features** extracted from Mapillary open-source imagery. Furthermore, the architecture extends beyond passive prediction to enable **active, counterfactual policy simulation**, allowing urban planners and municipal decision-makers to evaluate safety interventions (e.g., adding stop signs, marking crosswalks, installing traffic signals) and calculate the associated economic and health benefits using **Maximum Abbreviated Injury Scale (MAIS)** and **Value of a Statistical Life (VSL)** frameworks.

---

## 2. End-to-End System Architecture

The overall system architecture is organized into four interconnected functional layers, transitioning raw spatial and visual data into actionable municipal policy guidelines.

```
+--------------------------------------------------------------------------------------------------+
|                                  1. DATA INTEGRATION & INGESTION                                  |
|                                                                                                  |
|  +-----------------------------+                  +--------------------------------------------+  |
|  | City of Chicago Crash Data  |                  |    Open Source Mapillary Imagery (Street)  |  |
|  |  - 1M+ Incident Records     |                  |    - Street-Level Visual Context           |  |
|  |  - Spatial Coordinates      |                  |    - GIS / Geospatial Intersection Buffer  |  |
|  +--------------+--------------+                  +---------------------+----------------------+  |
|                 |                                                       |                         |
|                 v                                                       v                         |
|    [ Data Cleansing & Filter ]                            [ Visual Feature Extraction ]          |
|    - Remove Invalid/Missing Spatial                     - Detect Traffic Control Devices        |
|    - Geofence Chicago Intersections                     - Detect Stop Signs & Crosswalks       |
|                 |                                                       |                         |
|                 +--------------------------+----------------------------+                         |
|                                            |                                                      |
|                                            v                                                      |
|                             [ Spatial Joint & Merging Engine ]                                    |
|                             - GeoPandas Intersection Aggregation                                  |
|                                            |                                                      |
+--------------------------------------------|------------------------------------------------------+
                                             v
+--------------------------------------------------------------------------------------------------+
|                            2. FEATURE ENGINEERING & RISK REPRESENTATION                           |
|                                                                                                  |
|  +-------------------------------------+        +----------------------------------------------+  |
|  |  Feature Transformation Pipeline    |        |  Severity & Economic Cost Framework          |  |
|  |  - Categorical: One-Hot Encoding   |        |  - MAIS (0.0 None -> 1.0 Fatality)           |  |
|  |    (Weather, Light, Surface, Cause) |        |  - VSL Mapping ($ Valuation per Incident)    |  |
|  |  - Numeric: Standard Scaling       |        |  - Expected Harm Metric Formulation          |  |
|  |    (Prior Crash Counts, Density)    |        |                                              |  |
|  +------------------+------------------+        +----------------------+-----------------------+  |
|                     |                                                  |                          |
|                     +--------------------------+-----------------------+                          |
|                                                |                                                  |
+------------------------------------------------|--------------------------------------------------+
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                3. PREDICTIVE MODELING ENGINE                                     |
|                                                                                                  |
|  +--------------------------------------------------------------------------------------------+  |
|  | Train / Dev / Test Splitting & Class Imbalance Resampling (SMOTE / Class Weighting)        |  |
|  +---------------------------------------------+----------------------------------------------+  |
|                                                |                                                  |
|         +--------------------------------------+-----------------------------------+              |
|         |                                      |                                   |              |
|         v                                      v                                   v              |
|  +---------------+                    +------------------+                +--------------------+  |
|  | Baseline: LR  |                    | Baseline: RF     |                | Core DNN Model     |  |
|  | Interpretability                   | Non-linear Inter. |                | Deep Neural Net    |  |
|  +---------------+                    +------------------+                +---------+----------+  |
|                                                                                     |             |
|                                                                                     v             |
|                                                                           [ LLM Qualitative Spec ]|
|                                                                           - Crash Narrative Context|
+-------------------------------------------------------------------------------------|-------------+
                                                                                      v
+--------------------------------------------------------------------------------------------------+
|                            4. POLICY SIMULATION & DECISION SUPPORT                               |
|                                                                                                  |
|  +--------------------------------------------------------------------------------------------+  |
|  | Interactive Counterfactual Engine                                                          |  |
|  | - User Intervention Input (Toggle Crosswalk, Add Stop Sign, Upgrade Signal)               |  |
|  | - Synthetic Feature Vector Generation                                                       |  |
|  +---------------------------------------------+----------------------------------------------+  |
|                                                |                                                  |
|                                                v                                                  |
|                             [ Trained DNN Inference Execution ]                                   |
|                                                |                                                  |
|                                                v                                                  |
|  +--------------------------------------------------------------------------------------------+  |
|  | Risk & Benefit Quantification Output Module                                                |  |
|  | - Delta Crash Risk (P_before -> P_after)                                                   |  |
|  | - Estimated Prevented Injuries (MAIS Reduction)                                             |  |
|  | - Estimated Municipal Cost Benefit ($ VSL Savings)                                         |  |
|  +---------------------------------------------+----------------------------------------------+  |
|                                                |                                                  |
|                                                v                                                  |
|                             [ Actionable Safety Prioritization Dashboard ]                        |  |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Architectural Components

### 3.1 Component 1: Data Integration & Ingestion Pipeline

#### 3.1.1 Chicago Crash Dataset Processing
* **Input Dataset:** City of Chicago Traffic Crashes (1M+ records).
* **Key Attributes Filtered:**
  * Spatial Attributes: `LATITUDE`, `LONGITUDE`, `CRASH_DATE`, `STREET_NAME`, `INTERSECTION_INDICATOR`.
  * Contextual Attributes: `WEATHER_CONDITION`, `LIGHTING_CONDITION`, `ROAD_DEFECT`, `FIRST_CRASH_TYPE`, `PRIM_CONTRIBUTORY_CAUSE`.
  * Impact Attributes: `INJURIES_TOTAL`, `INJURIES_FATAL`, `INJURIES_INCAPACITATING`, `INJURIES_NON_INCAPACITATING`.
* **Preprocessing Pipeline:**
  1. Spatial Validation: Filter out records missing latitude/longitude or with coordinates outside Chicago municipal bounding box `[41.644, -87.940, 42.023, -87.524]`.
  2. Noise Reduction: Deduplicate crash records and standardize date/time formats.
  3. Intersection Spatial Buffering: Cluster crash points within a 20-meter radius buffer around known intersection nodes from Chicago GIS street centerline maps.

#### 3.1.2 Mapillary Street-Level Roadway Context Ingestion
* **Source:** Open-source Mapillary imagery API.
* **Computer Vision & Feature Extraction Layer:**
  * Extracts street-level imagery bounded within each intersection buffer.
  * Semantic Object Detection / Class Segmentation generates **Boolean Road-Scene Indicators**:
    * `has_traffic_signal` (Boolean: 0/1)
    * `has_stop_sign` (Boolean: 0/1)
    * `has_marked_crosswalk` (Boolean: 0/1)
    * `has_pedestrian_signal` (Boolean: 0/1)
    * `has_speed_hump` (Boolean: 0/1)
* **Spatial Merging Engine:**
  * Merges historical crash aggregates (e.g., prior 3-year crash count, severe injury count) with Mapillary feature vectors using spatial intersection IDs as the primary key ($K_{spatial}$).

---

### 3.2 Component 2: Feature Engineering & Risk Representation

#### 3.2.1 Feature Transformations
To ensure optimal training performance across neural models and linear baselines, features are preprocessed into standardized vector spaces:

| Feature Category | Original Variable | Transformation Method | Output Dimension / Format |
| :--- | :--- | :--- | :--- |
| **Infrastructure (Mapillary)** | `stop_sign`, `crosswalk`, `signal` | Pass-through (Binary) | Dense Vector $[0, 1]^K$ |
| **Environmental Context** | `weather`, `lighting`, `surface` | One-Hot Encoding (OHE) | Sparse Binary Vectors |
| **Accident Characteristics** | `crash_type`, `contributing_cause` | One-Hot Encoding (OHE) | Sparse Binary Vectors |
| **Historical Risk Metrics** | `prior_crash_count_3yr` | Standard Scaler ($\mu=0, \sigma=1$) | Continuous Value $\mathbb{R}$ |
| **Traffic Exposure** | `estimated_aadt` (Avg Traffic) | Log Transformation + Z-Score | Continuous Value $\mathbb{R}$ |

#### 3.2.2 MAIS & VSL Severity and Cost-Aware Extension
To move beyond generic binary classification (Crash vs. No Crash), the platform formulates an **Expected Harm Index ($E_H$)**:

1. **Maximum Abbreviated Injury Scale (MAIS) Weighting:**
   Injury categories reported in Chicago crash data are mapped to MAIS severity coefficients $S \in [0.0, 1.0]$:
   * **No Injury / Property Damage Only:** $S_{	ext{MAIS}} = 0.00$
   * **Reported / Non-Incapacitating Injury:** $S_{	ext{MAIS}} = 0.15$
   * **Incapacitating Injury:** $S_{	ext{MAIS}} = 0.55$
   * **Fatality:** $S_{	ext{MAIS}} = 1.00$

2. **Value of a Statistical Life (VSL) Monetary Valuation:**
   Using US DOT guidelines, a baseline VSL of $\$12.5	ext{M}$ is applied, with fractional valuations scaled according to MAIS severity $V(S_{	ext{MAIS}})$:
   * Fatal ($S=1.0$): $\$12,500,000$
   * Incapacitating ($S=0.55$): $\$2,625,000$
   * Non-Incapacitating ($S=0.15$): $\$237,500$

3. **Composite Location Risk Formulation:**
   $$	ext{Expected Harm}(i) = P(	ext{Crash}_i) 	imes \sum_{k} P(	ext{Injury}_k | 	ext{Crash}_i) 	imes V(S_k)$$
   This transforms predictions into monetized social burden estimates, directly actionable for capital budgeting.

---

### 3.3 Component 3: Predictive Modeling Engine

#### 3.3.1 Model Architectures & Hierarchy

```
                                +---------------------------+
                                | Input Feature Vector (X)  |
                                +-------------+-------------+
                                              |
                     +------------------------+------------------------+
                     |                        |                        |
                     v                        v                        v
        +-------------------------+ +-------------------+ +-------------------------+
        |   Logistic Regression   | |   Random Forest   | |   Deep Neural Network   |
        |  (Interpretability)     | | (Non-linear Trees)| |    (Primary Model)      |
        +------------+------------+ +---------+---------+ +------------+------------+
                     |                        |                        |
                     +------------------------+------------------------+
                                              |
                                              v
                                +---------------------------+
                                |  Predicted Risk Score P(y)|
                                +---------------------------+
```

1. **Primary Model: Deep Neural Network (DNN)**
   * **Architecture:** Multi-Layer Perceptron (MLP) with Residual Connections.
   * **Layers:**
     * Input Layer: Combined vector size ($N_{features} pprox 64$).
     * Dense Layer 1: 128 units, ReLU activation, Batch Normalization, Dropout ($p=0.3$).
     * Dense Layer 2: 64 units, ReLU activation, Batch Normalization, Dropout ($p=0.2$).
     * Dense Layer 3: 32 units, ReLU activation.
     * Output Layer: 1 unit, Sigmoid activation (outputs pedestrian crash risk probability $P(y \in [0, 1])$).
   * **Loss Function:** Focal Loss / Weighted Binary Cross-Entropy to handle high class imbalance.

2. **Baseline Models:**
   * **Logistic Regression:** Provides baseline odds-ratios for feature coefficient interpretability.
   * **Random Forest Classifier:** Captures feature interaction effects and non-linear boundaries across tree ensembles (100–300 trees).

3. **Exploratory LLM Integration:**
   * Uses Large Language Models (e.g., via fine-tuned Llama/GPT architectures) to parse unstructured crash narrative logs, extracting soft environmental attributes (e.g., "sun glare", "driver line-of-sight obstruction by parked delivery truck") to augment structured tabular features.

#### 3.3.2 Training, Validation, and Evaluation Metrics
* **Dataset Partitioning:** 70% Train, 15% Validation (Dev), 15% Hold-out Test set (Spatially stratified to prevent data leakage across adjacent intersections).
* **Imbalance Management:** Combination of Synthetic Minority Over-sampling Technique (SMOTE) on training folds and class-weight balancing ($w_1 = rac{N_{neg}}{N_{pos}}$).
* **Evaluation Metrics:**
  * **Area Under Precision-Recall Curve (PR-AUC):** Primary evaluation metric for imbalanced pedestrian crash events.
  * **F1-Score (at optimal decision threshold):** Balanced measure of Precision and Recall.
  * **ROC-AUC & Accuracy:** Secondary reporting benchmarks.

---

### 3.4 Component 4: Policy Simulation & Decision Support System

The policy simulation engine converts the static predictive model into an interactive decision-support workflow for counterfactual analysis.

#### 3.4.1 Counterfactual Simulation Workflow
1. **Base State Retrieval:** The user selects target intersection $i$ with feature vector $\mathbf{x}_i = [x_{infra}, x_{env}, x_{hist}]$.
2. **Intervention Modification:** The user specifies a hypothetical infrastructure alteration $\Delta x_{infra}$ (e.g., flipping `has_stop_sign` from 0 to 1, or adding `has_marked_crosswalk`).
3. **Synthetic Vector Construction:** Form modified feature vector $\mathbf{x}_i' = [x_{infra} + \Delta x_{infra}, x_{env}, x_{hist}]$.
4. **Model Inference:** Evaluate $\mathbf{x}_i'$ through the trained Deep Neural Network:
   $$\hat{P}_{baseline} = f_{DNN}(\mathbf{x}_i), \quad \hat{P}_{simulated} = f_{DNN}(\mathbf{x}_i')$$
5. **Delta Assessment:**
   $$\Delta 	ext{Risk} = \hat{P}_{simulated} - \hat{P}_{baseline}$$
   $$\Delta 	ext{Expected Harm (\$) } = \Delta 	ext{Risk} 	imes 	ext{MAIS/VSL Valuation Factor}$$

#### 3.4.2 Decision Support Output Interface
* **Safety Benefit Metrics:**
  * Estimated Annual Crashes Prevented per Intersection.
  * Estimated Injury Reduction (Count of MAIS 1-5 prevented).
  * Net Social Cost Savings (\$).
  * Benefit-Cost Ratio (BCR) comparing installation costs vs. VSL safety savings.

---

## 4. Key Architectural Differentiators & Novelty

| Feature / Dimension | Conventional Crash Models | Proposed Road Intelligence System |
| :--- | :--- | :--- |
| **Data Scope** | Tabular accident records only | Tabular city crash data + Mapillary visual road features |
| **Target Focus** | Generic motor vehicle collisions | Pedestrian-centric crash risk & injury mitigation |
| **Analytical Scope** | Retrospective (What happened?) | Counterfactual & Preventive (What if we intervene?) |
| **Severity Integration** | Simple crash count aggregation | Severity-weighted (MAIS) & Monetized (VSL) Expected Harm |

---

## 5. Technology Stack Summary

```
+-----------------------------------------------------------------------------------+
| LAYER                 | TECHNOLOGIES / LIBRARIES                                  |
+-----------------------+-----------------------------------------------------------+
| Data Ingestion        | Python 3.10+, Pandas, GeoPandas, Shapely, PyProj          |
| Visual Features       | Mapillary API v4, OpenCV, PyTorch / torchvision           |
| Machine Learning      | PyTorch / TensorFlow, Scikit-learn, XGBoost, LightGBM     |
| Policy Engine         | NumPy, SciPy, Custom Counterfactual Pipeline              |
| Storage & Spatial DB  | PostgreSQL + PostGIS, Parquet File Formats                |
| Dashboard / UI        | Streamlit / React + Leaflet.js Map Visualizations          |
+-----------------------------------------------------------------------------------+
```

---

## 6. Conclusion & Deployment Roadmap

The proposed architecture provides a scalable framework for modern traffic safety engineering. By linking spatial computer vision with historical records and economic valuation models, the platform shifts urban planning from reactive accident investigation to proactive, data-driven disaster prevention.
