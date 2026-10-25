"""
Uc sey uretir:
    1. Gruplara gore visit / conversion / spend oranlari ve kontrolden farklari
    2. Randomizasyon kontrolu (ozellikler gruplar arasinda benzer mi?)
    3. Two-model baseline: Qini AUC sayilari ve qini_baseline.png
"""

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklift.metrics import qini_auc_score, qini_curve
from sklift.models import TwoModels

warnings.filterwarnings("ignore")

CONTROL = "No E-Mail"
ARMS = ["Mens E-Mail", "Womens E-Mail"]
OUTCOMES = ["visit", "conversion"]
SEED = 17


def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        from sklift.datasets import fetch_hillstrom

        data, target, treatment = fetch_hillstrom(
            target_col="all", return_X_y_t=True, data_home="data"
        )
        df = pd.concat([data, target, treatment], axis=1)
    df.columns = df.columns.str.lower()
    return df



def group_summary(df):
    g = df.groupby("segment").agg(
        n=("visit", "size"),
        visit=("visit", "mean"),
        conversion=("conversion", "mean"),
        spend=("spend", "mean"),
    )
    print("\n=== 1. Gruplara gore ortalamalar ===")
    print((g.assign(visit=g.visit * 100, conversion=g.conversion * 100)).round(3).to_string())
    print("(visit ve conversion yuzde, spend kisi basi dolar)")

    print("\n=== Kontrole gore fark (yuzde puan) ve %95 guven araligi ===")
    c = df[df.segment == CONTROL]
    for arm in ARMS:
        a = df[df.segment == arm]
        for out in OUTCOMES:
            p1, p0 = a[out].mean(), c[out].mean()
            se = np.sqrt(p1 * (1 - p1) / len(a) + p0 * (1 - p0) / len(c))
            d = (p1 - p0) * 100
            print(f"{arm:14s} {out:10s} fark = {d:+.2f} puan  [{d - 196 * se:+.2f}, {d + 196 * se:+.2f}]")


def balance_check(df):
    cols = [c for c in ["recency", "history", "newbie", "mens", "womens"] if c in df.columns]
    print("\n=== 2. Randomizasyon kontrolu (ozellik ortalamalari gruplara gore) ===")
    print(df.groupby("segment")[cols].mean().round(3).to_string())
    print("Satirlar birbirine yakinsa gruplar rastgele atanmis demektir.")


def make_features(df):
    X = df[["recency", "history", "mens", "womens", "newbie"]].copy()
    dummies = pd.get_dummies(df[["zip_code", "channel"]], dtype=int)
    return pd.concat([X, dummies], axis=1)


def two_model_baseline(df, X_all, outcome, arm):
    mask = df.segment.isin([CONTROL, arm])
    X = X_all[mask]
    y = df.loc[mask, outcome].values
    w = (df.loc[mask, "segment"] == arm).astype(int).values
    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
        X, y, w, test_size=0.3, random_state=SEED, stratify=y * 2 + w
    )

    def learner():
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=SEED
        )

    model = TwoModels(learner(), learner(), method="vanilla")
    model.fit(Xtr, ytr, wtr)
    uplift = model.predict(Xte)
    return qini_auc_score(yte, uplift, wte), qini_curve(yte, uplift, wte)


def baseline_section(df):
    print("\n=== 3. Two-model baseline (tek bolme, ayar yok) ===")
    X_all = make_features(df)
    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(13, 5))
    rows = []
    for ax, out in zip(axes, OUTCOMES):
        for arm in ARMS:
            auc, (x, y) = two_model_baseline(df, X_all, out, arm)
            rows.append((out, arm, auc))
            (line,) = ax.plot(x, y, label=f"{arm} (Qini AUC {auc:.4f})")
            # Her kolun kendi rastgele hedefleme cizgisi (ayni renk, noktali)
            ax.plot([0, x[-1]], [0, y[-1]], ":", color=line.get_color(), alpha=0.7)
        ax.set_title(f"Qini egrisi, sonuc = {out}\n(noktali cizgi = rastgele hedefleme)")
        ax.set_xlabel("Hedeflenen musteri sayisi")
        ax.set_ylabel("Kumulatif net artis")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("qini_baseline.png", dpi=150)
    print(pd.DataFrame(rows, columns=["sonuc", "kol", "qini_auc"]).round(4).to_string(index=False))
    print("Grafik kaydedildi: qini_baseline.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Hillstrom CSV dosyasi (opsiyonel)")
    args = parser.parse_args()

    df = load_data(args.csv)
    print(f"Veri: {df.shape[0]} satir, sutunlar: {list(df.columns)}")
    group_summary(df)
    balance_check(df)
    baseline_section(df)