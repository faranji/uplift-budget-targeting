# Uplift Modeling + Linear Programming for Budget-Constrained Customer Targeting

> Who should a marketing campaign contact when the budget is limited? This project estimates each customer's *incremental* response to a campaign (uplift modeling) and turns those estimates into a budget-constrained targeting decision with linear programming. It then studies how that decision changes when the budget, contact costs or customer value change (parametric sensitivity analysis).

**Status:** planning phase (October 2026). There are no results yet. The roadmap below shows what is done and what is next.

Senior design project, Mathematical Engineering, Yıldız Technical University (YTÜ).

---

## 1. Motivation

A response model answers "who is likely to buy?". A campaign decision needs a different answer: "who buys *because of* the campaign?". Some customers buy anyway (sure things), some never buy (lost causes), and some react negatively to contact (sleeping dogs). Contacting the first two groups wastes budget, and contacting the third group destroys value.

Uplift modeling estimates the individual treatment effect

```
uplift(x) = P(Y = 1 | X = x, treated) - P(Y = 1 | X = x, control)
```

from data of a randomized campaign. Given those estimates, a marketer still has to decide whom to contact under a limited budget, possibly with several treatments that have different costs. That decision is an optimization problem, and its answer depends on parameters (budget, cost, value per conversion) that are uncertain in practice. This project connects the two steps.

## 2. Research questions

- **RQ1 (modeling).** How do the main uplift approaches (two-model, class transformation, S-learner, X-learner, and a tree-based direct model) compare when hyperparameters are tuned with uplift-specific metrics and the models are evaluated with repeated, nested, stratified cross-validation instead of a single train/test split?
- **RQ2 (decision).** Given estimated uplifts, how much does an LP-based assignment of customers and treatments improve expected incremental profit over simple policies (random, treat-all, response-propensity ranking, top-k by uplift)? When does the LP reduce to plain sort-and-cut, and when does it not?
- **RQ3 (sensitivity).** How do the optimal profit, the targeted set and the shadow price of the budget change as the budget, the contact cost and the value per conversion vary?

## 3. Positioning against related work

| Work | What it does | What this project adds |
|---|---|---|
| Gutierrez & Gérardy (2016) | Literature review of causal inference and uplift modeling; compares uplift approaches on a simulated dataset with a single 80/20 split, no hyperparameter tuning and no costs. | Real randomized marketing data, tuned models, repeated cross-validation, and a decision layer with costs and a budget. |
| Verbeke et al. (2012) | Profit-driven evaluation of churn models (maximum profit criterion). Targets customers by churn probability with a fixed retention rate and customer lifetime value, and names uplift modeling as a priority for future research. | Targets by *incremental* effect instead of churn probability, and optimizes the whole targeted set under a budget instead of choosing a single cut-off. |
| scikit-uplift | Provides uplift models, Qini/AUUC metrics and a maximum profit uplift measure. | This project does **not** claim a new metric. It uses the library as a toolbox and contributes the comparison design, the LP decision layer and the sensitivity analysis. |
| [causal-crm-uplift-modeling](https://github.com/faranji/causal-crm-uplift-modeling) (my earlier project) | Three-arm two-model pipeline on Hillstrom: CatBoost, LightGBM and logistic regression blended per arm, single 70/30 split, Qini curves, `visit` as the outcome. | Profit-relevant outcome (conversion/spend), tuning against uplift metrics, repeated nested CV with reported Qini AUC/AUUC, several model families compared, and the LP decision layer with sensitivity analysis, none of which exist in the earlier project. |

### Relation to my earlier project

The earlier project is a separate, earlier piece of work. This thesis starts from its data preparation and exploratory analysis and then changes the research question: from "can we rank customers by uplift?" to "which customers and treatments should a limited budget buy, and how stable is that decision?".

## 4. Data

All datasets come from randomized experiments (treatment is assigned at random), which is required for a valid uplift estimate. Scraped or observational data is not suitable.

- **Primary: Hillstrom e-mail dataset.** 64,000 customers, randomly assigned to a men's e-mail, a women's e-mail or no e-mail. Outcomes: `visit`, `conversion`, `spend`. Having two treatment arms plus a control group makes a multi-treatment decision possible.
- **Planned second dataset (optional): X5 RetailHero.** About 200,000 clients with raw purchase histories, which allows RFM feature engineering. Criteo and Lenta are alternatives.
- **Costs and margins are not in any of these datasets.** Contact cost and value per conversion are explicit assumptions, and the sensitivity analysis shows how the results depend on them.

Datasets are loaded with `sklift.datasets` and stored under `data/` (git-ignored).

## 5. Methods

**Uplift models**

1. Two-model approach (separate models for treated and control, T-learner baseline)
2. Class transformation (Jaskowski & Jaroszewicz)
3. S-learner (single model with the treatment indicator as a feature)
4. X-learner (Künzel et al., 2019), implemented by hand
5. Tree-based direct uplift model (implementation to be decided)

Base learners are gradient boosting and logistic regression. Hyperparameters are tuned with Optuna against an uplift metric.

**Evaluation**

- Qini curve and Qini coefficient, uplift curve and AUUC, uplift@k
- Repeated stratified k-fold cross-validation, stratified on treatment and outcome
- Nested cross-validation: tuning happens only in the inner loop
- Policy value: expected incremental profit of a targeting policy, estimated on held-out randomized data

**Decision layer (linear programming)**

For customer `i` and treatment `k` (including "no contact"), let `x_ik` be the share of the assignment, `u_ik` the estimated uplift, `v` the value of one conversion and `c_k` the contact cost:

```
maximize    sum_i sum_k (u_ik * v - c_k) * x_ik
subject to  sum_k x_ik            <= 1      for every customer i
            sum_i sum_k c_k x_ik  <= B      (budget)
            sum_i x_ik            <= N_k    (optional capacity of treatment k)
            0 <= x_ik <= 1
```

With a single treatment and equal cost, this problem reduces to sorting customers by uplift and cutting at the budget. Multiple treatments, different costs and capacity limits are what make the LP more than a sort.

**Parametric sensitivity analysis**

The budget `B` is a right-hand-side parameter, the profit coefficients `u_ik * v - c_k` are objective parameters, and the contact costs enter the left-hand side. The optimal value is piecewise linear in `B`. The analysis reports breakpoints and shadow prices (duals). The first version solves the LP on a grid of parameter values with a solver (SciPy/HiGHS). A simplex-based parametric algorithm is a possible extension, not a commitment.


## 6. Assumptions and limitations

- Contact costs and value per conversion are assumptions. No dataset provides them, so conclusions are conditional on the tested ranges.
- The LP treats the estimated uplifts as known. In reality they are noisy estimates, so the optimal policy inherits that noise.
- Each dataset covers a single campaign. Results may not carry over to other channels or periods.
- Customers are assumed not to influence each other (no interference between units).

## 7. References

- Gutierrez, P., & Gérardy, J.-Y. (2016). Causal Inference and Uplift Modeling: A Review of the Literature. *JMLR: Workshop and Conference Proceedings*, 67, 1–13.
- Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*, 218(1), 211–229.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156–4165.
- Jaskowski, M., & Jaroszewicz, S. (2012). Uplift modeling for clinical trial data. *ICML Workshop on Clinical Data Analysis*.
- Radcliffe, N. J. (2007). Using control groups to target on predicted lift. *Direct Marketing Analytics Journal*.
- Ascarza, E. (2018). Retention futility: Targeting high-risk customers might be ineffective. *Journal of Marketing Research*, 55(1), 80–98.
- Temelcan Ergenecosar, G., Kocken, H. G., Albayrak, I., & Sivri, M. (2026). Solution to the single parametric linear programming problems via simplex-based algorithms: handling the uncertainties in costs, left or right-hand sides. *Croatian Operational Research Review*, 17(1), 29–40.
- Hillstrom, K. (2008). The MineThatData E-Mail Analytics and Data Mining Challenge.

## 8. Author and advisor

- Author: Simanur Hattapoğlu
- Advisor: İnci Albayrak

## License

To be decided.