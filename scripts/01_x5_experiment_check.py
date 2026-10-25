"""
Step 01: First look at the X5 RetailHero experiment.

WHAT THIS DATA IS
  X5 Retail Group (a large food retailer) sent a communication to about half of a group of
  loyalty-card customers and nothing to the other half, then recorded who bought afterwards.
  Two files are used here (a third file, purchases, is looked at in step 02):

  data/uplift_train.csv.gz   one row per customer in the experiment (200,039 rows)
      client_id      customer identifier
      treatment_flg  1 = received the communication, 0 = did not (the "control" group)
      target         1 = the customer bought something afterwards, 0 = did not

  data/clients.csv.gz        one row per loyalty-card customer (400,162 rows)
      first_issue_date   date the loyalty card was issued
      first_redeem_date  date the customer first redeemed loyalty points (empty = never)
      age, gender        age in years (some values are invalid), gender F / M / U (unknown)

WHAT THIS SCRIPT ANSWERS
  1. How big is the effect of the communication on the purchase rate?
  2. Is that effect the same for everybody, or does it depend on age / gender?
  3. Were the two groups really comparable (a fair experiment)?
  4. A figure of the age result that can be shown to the advisor.

HOW TO RUN (from the repository root, after downloading the two files into data/):
    python scripts/01_x5_experiment_check.py
"""

import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt

TRAIN_FILE = "data/uplift_train.csv.gz"
CLIENTS_FILE = "data/clients.csv.gz"
FIGURE_FILE = "reports/figures/x5_age_band_effect.png"


# ----------------------------------------------------------------------------------------
# Small helper functions (used in sections 1-2)
# ----------------------------------------------------------------------------------------
def effect(sub):
    """Effect of the communication in one group of customers.

    target is 0/1, so the mean of target is the share of customers who bought (purchase rate).
    effect = purchase rate of treated customers - purchase rate of control customers.
    se     = standard error of that difference (how much it would wobble with a new sample).
    """
    treated = sub.loc[sub["treatment_flg"] == 1, "target"]
    control = sub.loc[sub["treatment_flg"] == 0, "target"]
    diff = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    return len(sub), control.mean(), treated.mean(), diff, se


def effect_table(data, group_col):
    """Effect of the communication inside every group of one column (e.g. every age band)."""
    rows = []
    for group, sub in data.groupby(group_col, observed=True):
        n, control, treated, diff, se = effect(sub)
        rows.append({
            "group": group,
            "n": n,
            "control_%": control * 100,
            "treated_%": treated * 100,
            "effect_pp": diff * 100,  # pp = percentage points
            "ci_low": (diff - 1.96 * se) * 100,  # 95% confidence interval = effect +- 1.96 * se
            "ci_high": (diff + 1.96 * se) * 100,
            "se": se,
        })
    return pd.DataFrame(rows)


def cochran_q(table):
    """Test: is the effect the same in all groups?  (Cochran's Q test)

    Each group's effect is weighted by 1 / se^2 (precise groups count more). Q measures how far the
    group effects are from their weighted average. If all groups had the same true effect, Q would be
    small and follow a chi-square distribution with (groups - 1) degrees of freedom.
    A small p-value (below 0.05) means the effects really differ between groups.
    """
    known = table[table["group"] != "unknown"]  # the "unknown" group is not a real group
    d = known["effect_pp"].to_numpy() / 100
    w = 1 / known["se"].to_numpy() ** 2
    q = float((w * (d - (w * d).sum() / w.sum()) ** 2).sum())
    dof = len(known) - 1
    return q, dof, chi2.sf(q, dof)


# ----------------------------------------------------------------------------------------
# 0. Load the two files, join them and clean the age column
# ----------------------------------------------------------------------------------------
train = pd.read_csv(TRAIN_FILE)
clients = pd.read_csv(CLIENTS_FILE, parse_dates=["first_issue_date", "first_redeem_date"])

# Attach age, gender and card dates to every experiment row (both files share client_id).
df = train.merge(clients, on="client_id", how="left")

# Ages outside 18-90 are data errors (for example -5 or 1900): treat them as missing.
df.loc[~df["age"].between(18, 90), "age"] = np.nan

# Age bands: pd.cut(bins=[17, 29, ...]) means 18-29, 30-39, ... (the lower edge is excluded).
df["age_band"] = pd.cut(df["age"], bins=[17, 29, 39, 49, 59, 90], labels=["18-29", "30-39", "40-49", "50-59", "60+"])
df["age_band"] = df["age_band"].cat.add_categories("unknown").fillna("unknown")
df["gender"] = df["gender"].fillna("unknown")

# Dates as "days since 2017-01-01" so that they can be used as numbers later.
reference = pd.Timestamp("2017-01-01")
df["card_issue_days"] = (df["first_issue_date"] - reference).dt.days
df["card_redeem_days"] = (df["first_redeem_date"] - reference).dt.days

print(f"Rows: {len(df):,}  (customers with a missing or invalid age: {(df['age_band'] == 'unknown').sum():,})")
print(df.head())

# ----------------------------------------------------------------------------------------
# 1. Overall effect
# ----------------------------------------------------------------------------------------
# HOW TO READ: "control %" is the share of customers who bought although they received nothing.
# "treated %" is the same share among customers who received the communication. Because customers
# are assumed to be assigned like a coin flip, the difference between the two is the effect CAUSED
# by the communication. Result we obtained: 60.33% vs 63.65%, i.e. +3.32 percentage points.
# Meaning: out of 100 contacted customers about 64 buy, but only about 3 of those purchases happen
# because of the message; the other 60 or so would have bought anyway.
print("\n=== 1. Overall effect ===")
print(f"Share of customers who received the communication: {df['treatment_flg'].mean():.3f}")
n, control, treated, diff, se = effect(df)
print(f"Purchase rate  control: {control * 100:.2f}%   treated: {treated * 100:.2f}%")
print(f"Effect: {diff * 100:+.2f} percentage points  (95% interval {(diff - 1.96 * se) * 100:+.2f} to {(diff + 1.96 * se) * 100:+.2f})")

# ----------------------------------------------------------------------------------------
# 2. Does the effect depend on age or gender?
# ----------------------------------------------------------------------------------------
# HOW TO READ: every row repeats section 1 inside one group. If the effect were the same for
# everybody, all rows would show about the same "effect_pp". Result we obtained for age: the effect
# rises from +1.3 points (18-29) to +5.3 points (60+), and Cochran's Q test gives p < 0.001, so
# the difference between age bands is real and not noise. Gender differs only weakly (p = 0.044).
# This is the reason to model the effect per customer (uplift modeling) instead of contacting everybody.
print("\n=== 2. Effect by age band ===")
age_table = effect_table(df, "age_band")
print(age_table.drop(columns="se").round(2).to_string(index=False))
q, dof, p = cochran_q(age_table)
print(f"Same effect in all age bands?  Cochran Q = {q:.1f}, degrees of freedom = {dof}, p = {p:.4f}")

print("\n=== 2b. Effect by gender ===")
gender_table = effect_table(df, "gender")
print(gender_table.drop(columns="se").round(2).to_string(index=False))
q, dof, p = cochran_q(gender_table)
print(f"Same effect for all genders?  Cochran Q = {q:.1f}, degrees of freedom = {dof}, p = {p:.4f}")

# ----------------------------------------------------------------------------------------
# 3. Were the two groups comparable?  (balance check)
# ----------------------------------------------------------------------------------------
# WHY: the effect in section 1 only means "caused by the message" if the two groups looked the same
# BEFORE the message. The documentation does not say how customers were assigned, so we check it.
#
# 3a. Standardized mean difference (SMD) = (mean of treated - mean of control) / typical spread.
#     Rule of thumb: |SMD| below 0.1 means the groups are balanced on that variable.
#     Result we obtained: every |SMD| is below 0.01, so the groups are very well balanced.
print("\n=== 3a. Balance: standardized mean difference (treated vs control) ===")
check = pd.DataFrame({
    "age": df["age"],
    "card_issue_days": df["card_issue_days"],
    "card_redeem_days": df["card_redeem_days"],
    "gender_F": (df["gender"] == "F").astype(float),
    "gender_M": (df["gender"] == "M").astype(float),
    "gender_U": (df["gender"] == "U").astype(float),
})
smd = {}
for name in check.columns:
    t = check.loc[df["treatment_flg"] == 1, name].dropna()
    c = check.loc[df["treatment_flg"] == 0, name].dropna()
    smd[name] = (t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)
print(pd.Series(smd).round(3).to_string())

# 3b. Can a model guess who received the communication from the customer variables?
#     If assignment was random, nothing can predict it: AUC = 0.5 (a coin flip).
#     AUC is computed on predictions for customers the model has NOT seen (3-fold cross-validation).
#     Result we obtained: AUC = 0.518 with all variables. A single-variable check shows where this
#     small signal comes from: first_redeem_date alone. Because of this we do NOT use first_redeem_date
#     as a model input later (whether and when a customer redeemed points may itself depend on the
#     communication, so it is not a clean "before the campaign" variable).
print("\n=== 3b. Can customer variables predict who got the communication? (0.5 = no) ===")


def treatment_auc(columns):
    """AUC of a gradient boosting model that predicts treatment_flg from the given columns."""
    features = pd.DataFrame({c: check[c] for c in columns})  # missing values are handled by the model
    model = HistGradientBoostingClassifier(max_iter=60, random_state=0)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    proba = cross_val_predict(model, features, df["treatment_flg"], cv=folds, method="predict_proba")[:, 1]
    return roc_auc_score(df["treatment_flg"], proba)


all_columns = list(check.columns)
without_redeem = [c for c in all_columns if c != "card_redeem_days"]
print(f"All variables:                     AUC = {treatment_auc(all_columns):.3f}")
print(f"All except card_redeem_days:       AUC = {treatment_auc(without_redeem):.3f}")
for single in ["age", "card_issue_days", "card_redeem_days", "gender_F"]:
    print(f"Only {single:<28s}  AUC = {treatment_auc([single]):.3f}")

# ----------------------------------------------------------------------------------------
# 4. Figure for the advisor
# ----------------------------------------------------------------------------------------
# Left: purchase rates (bars start at zero, so you see that most customers buy anyway).
# Right: the effect of the communication per age band, with its 95% interval.
bands = age_table[age_table["group"] != "unknown"].reset_index(drop=True)
x = np.arange(len(bands))
labels = [f"{g}\n(n = {n:,})" for g, n in zip(bands["group"], bands["n"])]
blue, orange, grey = "#2a78d6", "#eb6834", "#52514e"

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))

left.bar(x - 0.19, bands["control_%"], 0.36, color=blue, label="No communication (control)")
left.bar(x + 0.19, bands["treated_%"], 0.36, color=orange, label="Communication (treated)")
left.set_ylim(0, 90)
left.set_ylabel("Purchase rate (%)")
left.set_title("Most customers buy even without a message", loc="left", fontweight="bold")
left.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)

error = [bands["effect_pp"] - bands["ci_low"], bands["ci_high"] - bands["effect_pp"]]
right.axhline(diff * 100, color=grey, linestyle="--", linewidth=1)
right.text(-0.55, diff * 100 + 0.15, f"overall effect {diff * 100:.1f} pp", color=grey, fontsize=9)
right.errorbar(x, bands["effect_pp"], yerr=error, fmt="o", color=orange, ecolor=grey, capsize=4, markersize=8)
right.set_xlim(-0.6, len(bands) - 0.4)
right.set_ylim(min(0, bands["ci_low"].min()) - 0.5, bands["ci_high"].max() + 1)
right.set_ylabel("Increase in purchase rate (percentage points)")
right.set_title("...but the effect of the message grows with age", loc="left", fontweight="bold")

for axis in (left, right):
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=9)
    axis.yaxis.grid(True, color="#e6e5e1")
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
os.makedirs(os.path.dirname(FIGURE_FILE), exist_ok=True)
fig.savefig(FIGURE_FILE, dpi=200)
print(f"\nFigure saved to {FIGURE_FILE}")
