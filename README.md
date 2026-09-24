# Practical Application II: What Drives the Price of a Car?

An end-to-end predictive modeling and econometric analysis of used vehicle resale prices using the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

---

## 🔗 Project Notebook
* Direct link to executed notebook: [notebooks/prompt_II.ipynb](notebooks/cars.ipynb)

---

## 1. Executive Summary & Business Understanding
Used vehicle dealerships operate in a dynamic, capital-intensive environment where inventory profitability is governed by precise valuation and rapid lot turnover. Overpaying at regional auto auctions or accepting over-valued trade-ins directly erodes dealership margins, while overpriced retail lot inventory increases holding costs and leads to steep forced markdowns.

### Business Objective
Deliver an interpretable, data-driven pricing intelligence system to:
1. **Identify Top Value Drivers:** Quantify positive price premiums commanded by specific powertrains, drivetrains, and body styles.
2. **Mitigate Depreciation Exposure:** Empirically establish risk thresholds (e.g., mileage and age inflection points) to avoid toxic inventory.
3. **Deploy Operational Valuation:** Provide transparent regression benchmarks for lot appraisal teams.

### Mathematical Framing
We model used car valuation as a continuous supervised regression problem:
62714\hat{y} = f(X) + \epsilon, \quad y \in \mathbb{R}^+62714
where $ represents the transaction sale price ($USD), $ is the multidimensional feature matrix (continuous and categorical attributes), and $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is the residual error.

---

## 2. Dataset & Exploratory Data Analysis (EDA)
The study utilizes the Kaggle Used Cars Dataset comprising **426,880 listings** across 18 initial features:
* **Target:** `price`
* **Continuous Features:** `odometer`, `year` (engineered into `vehicle_age = 2024 - year`)
* **Categorical Specifications:** `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `type`, and `manufacturer`

### Key Empirical Findings:
* **Extreme Artifacts & Skew:** The raw price distribution contained severe anomalies ranging from placeholder $0 listings to invalid extreme values ($3.7B).
* **Fuel & Powertrain Premiums:** Diesel vehicles exhibit significantly higher median resale prices compared to conventional gasoline platforms.
* **Drivetrain Elasticity:** Four-wheel drive (4WD) and all-wheel drive (AWD) configurations command steady price premiums across all vehicle age segments.
* **Condition Monotonicity:** Resale value scales directly with declared condition, with salvage and fair-condition titles experiencing steep discount cliffs.

---

## 3. Data Preparation & Preprocessing Pipeline
To prevent data contamination and prepare the feature space for linear modeling:
* **Outlier Filtering:**
  * Price bounded to **$1,500 – $80,000** (eliminates salvage scrap auctions and hyper-luxury collector outliers).
  * Odometer bounded to **500 – 280,000 miles** (eliminates rollbacks, test entries, and non-drivable wrecks).
  * Year restricted to **1995 – 2022** (focuses on standard retail financing inventory).
  * Cleaned analytic subset: **340,497 verified records**.
* **Leakage Prevention & Feature Pipeline:**
  * Sampled an 80,000-record stratum for cross-validated modeling.
  * Executed an **80/20 Train-Test split** prior to calculating transformation parameters.
  * Packaged preprocessors via Scikit-Learn `ColumnTransformer`:
    * Continuous numeric features (`odometer`, `vehicle_age`): Standardized via `StandardScaler`.
    * Nominal categoricals (`condition`, `fuel`, `drive`, etc.): One-hot encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')` to eliminate multicollinearity.

---

## 4. Modeling & Hyperparameter Optimization
Three linear regression architectures were evaluated using **5-Fold Cross-Validation (`cv=5`)** with negative MSE scoring:
1. **Ordinary Least Squares (OLS) Linear Regression:** Unpenalized baseline model.
2. **Ridge Regression ($L_2$ Regularization):** Shrinks coefficient magnitudes to counter multicollinearity.
   * *Optimal $\alpha = 1.0$*
3. **Lasso Regression ($L_1$ Regularization):** Enforces sparsity by driving uninformative feature weights strictly to zero.
   * *Optimal $\alpha = 0.1$*

---

## 5. Model Evaluation & Benchmark Results
Models were benchmarked against the held-out 20% test partition (16,000 unseen vehicles):

| Model | MAE ($) | RMSE ($) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | $4,414.82 | $6,164.73 | 0.7642 |
| **Ridge Regression (Tuned)** | **$4,414.37** | **$6,164.16** | **0.7642** |
| **Lasso Regression (Tuned)** | $4,414.69 | $6,164.47 | 0.7642 |

### Performance Rationale
* **Mean Absolute Error (MAE):** Selected as the core operational metric (~**$4,414**), reflecting the expected dollar variance for dealership trade-in appraisal tolerances.
* **Root Mean Squared Error (RMSE):** Monitored (~**$6,164**) to penalize large mispricing errors on higher-tier inventory.
* **$R^2$ Score (~0.764):** Demonstrates that the regularized linear model successfully explains **>76.4%** of total price variance in the test partition.

---

## 6. Key Value Drivers & Depreciation Factors
Extracting coefficients from the tuned Lasso model highlights the primary factors influencing vehicle price:

### Top Positive Value Drivers (+ Price)
1. **Diesel Powertrains:** Strongest positive powertrain coefficient, commanding premium retention across truck and commercial lines.
2. **4WD / AWD Drivetrains:** Significant price premium over FWD and RWD configurations.
3. **Pickup Trucks & Commercial Body Types:** Retain higher utility value than standard passenger sedans.
4. **8-Cylinder Engines:** Command higher pricing elasticity within truck and performance segments.

### Top Depreciation Drivers (- Price)
1. **Odometer Mileage:** Largest continuous downward pressure on vehicle price.
2. **Vehicle Age:** Linear depreciation penalty compounding with mechanical wear.
3. **Salvage / Rebuilt Titles:** Substantial price penalty that completely overrides low mileage or physical condition.
4. **Front-Wheel Drive (FWD):** Strong downward valuation trend relative to 4WD platforms.

---

## 7. Actionable Recommendations for Dealerships

### 1. Procurement & Sourcing Strategy
* **Overweight Trucks & Diesels:** Focus auction bidding on well-maintained diesel pickups and 4WD SUVs; these configurations hold residual value best and rotate quickly off dealership lots.
* **Seasonality Adjustments:** Stock 4WD inventory ahead of regional winter demand cycles to capture local price elasticity.

### 2. Risk Mitigation & Trade-in Appraisal
* **The 100k-Mile Discount Buffer:** Enforce steeper automated discount haircuts for vehicles approaching or exceeding 100,000 miles to safeguard margin buffers.
* **Strict Branded Title Exclusions:** Restrict trade-in allowances on branded or salvage titles to wholesale liquidation rates, avoiding retail lot placement.

### 3. Operational Deployment
* Integrate the regularized Lasso pricing equation into dealership CRM / lot appraisal tools to standardize appraisal baselines across lot managers.
