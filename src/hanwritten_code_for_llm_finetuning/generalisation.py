import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.stats import chi2, binomtest
import emoji

# Pre-compiled regular expressions
URL_REGEX = re.compile(r'https?://\S+|www\.\S+|t\.co/\S+')
MENTION_REGEX = re.compile(r'@\w+')
LAUGH_REGEX = re.compile(r'[هـ]{2,}|[hH][aA]{2,}|[lL][oO][lL]')
REPEAT_PUNCT_REGEX = re.compile(r'([!؟?\.])\1+')

def compute_sarcasm_prior(df: pd.DataFrame, col: str = "sarcasm", name: str = "Split") -> float:
    """Computes and prints the sarcasm rate for a given dataset."""
    prior = df[col].astype(int).mean()
    print(f"[{name}] Taux de sarcasme : {prior*100:.2f}% ({df[col].astype(int).sum()}/{len(df)} tweets)")
    return float(prior)

def evaluate_at_threshold(y_true, probs, threshold: float = 0.5) -> dict:
    """Evaluates binary predictions at a specific probability threshold."""
    y_true_arr = np.asarray(y_true, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    preds = (probs_arr >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1_sarc": float(f1_score(y_true_arr, preds, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true_arr, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true_arr, preds, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, preds))
    }

def compute_threshold_curve(y_true, probs, num_points: int = 91) -> tuple:
    """Computes F1 score across a range of thresholds [0.05, 0.95]."""
    y_true_arr = np.asarray(y_true, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    thresholds = np.linspace(0.05, 0.95, num_points)
    f1_scores = np.array([
        f1_score(y_true_arr, (probs_arr >= t).astype(int), pos_label=1, zero_division=0)
        for t in thresholds
    ])
    return thresholds, f1_scores

def find_optimal_threshold(y_true, probs, num_points: int = 91) -> tuple:
    """Finds the optimal decision threshold maximizing F1 on the positive class."""
    thresholds, f1_scores = compute_threshold_curve(y_true, probs, num_points)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

def plot_f1_vs_threshold_curves(models_summary: dict, y_test, output_path: str = None):
    """
    Plots F1 vs threshold curves for all models on the test set,
    marking the val threshold and peak a posteriori test threshold.
    """
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    y_test_arr = np.asarray(y_test, dtype=int)

    for i, (name, m_info) in enumerate(models_summary.items()):
        color = colors[i % len(colors)]
        probs = np.asarray(m_info["probs"], dtype=float)
        v_th = float(m_info["val_thresh"])

        ths, f1_curve = compute_threshold_curve(y_test_arr, probs)
        opt_idx = int(np.argmax(f1_curve))
        opt_th = ths[opt_idx]
        opt_f1 = f1_curve[opt_idx]

        f1_at_val = f1_score(y_test_arr, (probs >= v_th).astype(int), pos_label=1, zero_division=0)

        plt.plot(ths, f1_curve, label=f"{name} (Peak={opt_f1:.3f} @ {opt_th:.2f})", color=color, lw=2)
        plt.scatter([v_th], [f1_at_val], color=color, s=80, zorder=5, marker="o")
        plt.axvline(v_th, color=color, linestyle="--", alpha=0.4)

    plt.axvline(0.5, color="gray", linestyle=":", alpha=0.7, label="Seuil par défaut (0.50)")
    plt.xlabel("Seuil de décision P(sarcastique)", fontsize=11)
    plt.ylabel("F1-score (classe sarcastique)", fontsize=11)
    plt.title("Partie F1 : Courbe F1 = f(seuil) sur le test officiel descellé", fontsize=12, fontweight="bold")
    plt.legend(loc="lower center", fontsize=9, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Graphique sauvegardé avec succès dans : {output_path}")
    plt.show()

def remove_urls_and_mentions(text: str) -> str:
    """Removes URLs and user mentions from text."""
    if not isinstance(text, str):
        return ""
    text = URL_REGEX.sub(" ", text)
    text = MENTION_REGEX.sub(" ", text)
    return " ".join(text.split())

def perturb_sarcasm_cues(text: str) -> str:
    """Ablation of surface sarcasm markers: emojis, laughter (هههه), and repeated punctuation."""
    if not isinstance(text, str):
        return ""
    # 1. Remove emojis
    text = emoji.replace_emoji(text, replace="")
    # 2. Remove laughter
    text = LAUGH_REGEX.sub(" ", text)
    # 3. Collapse repeated punctuation (e.g. "!!" -> "!")
    text = REPEAT_PUNCT_REGEX.sub(r"\1", text)
    return " ".join(text.split())

def bootstrap_f1_ci(y_true, y_pred, n_bootstraps: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple:
    """Calculates non-parametric 95% bootstrap confidence interval for F1-sarcastic."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    n = len(y_true_arr)

    boot_scores = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        score = f1_score(y_true_arr[idx], y_pred_arr[idx], pos_label=1, zero_division=0)
        boot_scores.append(score)

    boot_scores = np.array(boot_scores)
    ci_low = float(np.percentile(boot_scores, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return float(np.mean(boot_scores)), ci_low, ci_high, boot_scores

def mcnemar_test(y_true, y_pred1, y_pred2) -> dict:
    """
    Performs McNemar's paired statistical test with Edwards continuity correction.
    b: Model 1 correct, Model 2 incorrect
    c: Model 1 incorrect, Model 2 correct
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    p1 = np.asarray(y_pred1, dtype=int)
    p2 = np.asarray(y_pred2, dtype=int)

    correct1 = (p1 == y_true_arr)
    correct2 = (p2 == y_true_arr)

    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    total_discordant = b + c

    if total_discordant == 0:
        return {
            "b (M1 ok, M2 err)": 0,
            "c (M1 err, M2 ok)": 0,
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "significant": False
        }

    # Edwards continuity correction
    chi2_stat = float((abs(b - c) - 1.0) ** 2 / total_discordant)
    p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))

    # Exact binomial test if discordant pairs are small
    if total_discordant < 25:
        b_res = binomtest(b, total_discordant, p=0.5, alternative="two-sided")
        p_val = float(b_res.pvalue)

    return {
        "b (M1 ok, M2 err)": b,
        "c (M1 err, M2 ok)": c,
        "chi2_stat": chi2_stat,
        "p_value": p_val,
        "significant": bool(p_val < 0.05)
    }

def extract_30_errors(test_df: pd.DataFrame, y_true, y_pred, probs, contra_tweets=None) -> pd.DataFrame:
    """Extracts 15 False Negatives and 15 False Positives for qualitative analysis."""
    df = test_df.copy().reset_index(drop=True)
    df["y_true"] = np.asarray(y_true, dtype=int)
    df["y_pred"] = np.asarray(y_pred, dtype=int)
    df["prob_sarc"] = np.asarray(probs, dtype=float)

    if contra_tweets is None:
        contra_tweets = set()

    df["is_contradictory_duplicate"] = df["tweet"].astype(str).isin(contra_tweets)

    # False Negatives (FN): True = 1, Pred = 0 (sorted by lowest predicted probability)
    fn_df = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].sort_values(by="prob_sarc", ascending=True)
    fn_15 = fn_df.head(15).copy()
    fn_15["error_type"] = "Faux Négatif (FN)"

    # False Positives (FP): True = 0, Pred = 1 (sorted by highest predicted probability)
    fp_df = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].sort_values(by="prob_sarc", ascending=False)
    fp_15 = fp_df.head(15).copy()
    fp_15["error_type"] = "Faux Positif (FP)"

    combined = pd.concat([fn_15, fp_15], ignore_index=True)
    return combined
