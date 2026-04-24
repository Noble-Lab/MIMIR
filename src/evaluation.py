import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional, Literal
import pickle
from scipy.stats import gaussian_kde


def evaluate_imputations(
    pred_dict: Dict[Tuple[Tuple[str, ...], str], pd.DataFrame],
    multi_omic_data: Dict[str, pd.DataFrame],
    plot_scatter: bool = True,
    max_points_plot: int = 5000,
) -> Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]]:
    """
    Evaluate imputed modalities against ground truth.

    Args:
        pred_dict: dict with keys ((modalities_present_tuple), target_mod)
                   and values DataFrames of imputed values
                   (index = samples, columns = features).
        multi_omic_data: dict {mod: DataFrame} with true values.
        plot_scatter: if True, show scatter plots (flattened true vs imputed).
        max_points_plot: subsample points for plotting if there are more than this.

    Returns:
        metrics: dict keyed by the same keys as pred_dict, with values:
            {
              'mse': float,
              'pearson': float,
              'spearman': float,
              'n_points': int,
            }
    """
    metrics: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = {}

    for key, df_pred in pred_dict.items():
        modalities_present, target_mod = key

        if target_mod not in multi_omic_data:
            print(f"[WARN] Target modality '{target_mod}' not in multi_omic_data; skipping.")
            continue

        df_true_full = multi_omic_data[target_mod]

        # Align on samples and features
        common_samples = df_pred.index.intersection(df_true_full.index)
        common_features = df_pred.columns.intersection(df_true_full.columns)

        if len(common_samples) == 0 or len(common_features) == 0:
            print(f"[WARN] No overlap in samples/features for key {key}; skipping.")
            continue

        df_pred_aligned = df_pred.loc[common_samples, common_features]
        df_true_aligned = df_true_full.loc[common_samples, common_features]

        # Flatten to 1D
        y_true = df_true_aligned.values.reshape(-1)
        y_pred = df_pred_aligned.values.reshape(-1)

        # Mask non-finite values on either side
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        n_points = y_true.shape[0]
        if n_points == 0:
            print(f"[WARN] No finite points for key {key}; skipping.")
            continue

        # MSE
        mse = float(np.mean((y_true - y_pred) ** 2))

        # Use pandas Series for Pearson & Spearman (no SciPy dependency)
        s_true = pd.Series(y_true)
        s_pred = pd.Series(y_pred)

        pearson = float(s_true.corr(s_pred, method="pearson"))
        spearman = float(s_true.corr(s_pred, method="spearman"))

        metrics[key] = {
            "mse": mse,
            "pearson": pearson,
            "spearman": spearman,
            "n_points": int(n_points),
        }

        # Optional scatter plot with KDE coloring
        if plot_scatter:
            # subsample if too many points
            if n_points > max_points_plot:
                idx = np.random.choice(n_points, size=max_points_plot, replace=False)
                x_plot = y_true[idx]
                y_plot = y_pred[idx]
            else:
                x_plot = y_true
                y_plot = y_pred

            # Compute 2D KDE at each point (for coloring)
            xy = np.vstack([x_plot, y_plot])
            kde = gaussian_kde(xy)
            density = np.log(kde(xy))

            # Sort by density so densest points are drawn last (on top)
            order = density.argsort()
            x_plot = x_plot[order]
            y_plot = y_plot[order]
            density = density[order]

            plt.figure(figsize=(6, 6))
            plt.scatter(
                x_plot,
                y_plot,
                c=density,
                s=4,
                cmap="OrRd",  # orange -> red
                alpha=0.8,
                linewidths=0,
            )
            
            # --- make x/y scales identical + add y=x line ---
            lo = np.nanmin([x_plot.min(), y_plot.min()])
            hi = np.nanmax([x_plot.max(), y_plot.max()])
            plt.xlim(lo, hi)
            plt.ylim(lo, hi)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color='black')
            
            plt.xlabel("Observed values", fontsize=18)
            plt.ylabel("Imputed values", fontsize=18)
            plt.title(
                f"present={modalities_present}, target={target_mod}\n"
                #f"MSE={mse:.4f}, r={pearson:.3f}, ρ={spearman:.3f}",
                f"r={pearson:.3f}",
                fontsize=15,
            )
           # plt.colorbar(label="Point density (KDE)")

            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            
            plt.tight_layout()
            plt.show()

    return metrics

def per_feature_corr(true_df, pred_df):
    """
    Compute per-feature Pearson correlations ignoring NaNs.
    Returns a Series indexed by feature.
    """
    corrs = {}
    for col in true_df.columns:
        t = true_df[col].values
        p = pred_df[col].values
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() < 3:
            corrs[col] = np.nan
        else:
            corrs[col] = np.corrcoef(t[mask], p[mask])[0, 1]
    return pd.Series(corrs)

def compare_methods_per_feature(
    method1: str or Dict[Tuple[Tuple[str, ...], str], pd.DataFrame],
    method2: str or Dict[Tuple[Tuple[str, ...], str], pd.DataFrame],
    multi_omic_data: Dict[str, pd.DataFrame],
    m1_name: str = 'method 1',
    m2_name: str = 'method 2',
    plot_scatter: bool = True,
):
    """
    For each scenario (modalities_present, target_mod):

        1. Load predictions for that scenario.
        2. Align (samples, features) between methods and true data.
        3. Compute per-feature Pearson correlations:
               corr_m1, corr_2
        4. Make scatter plot of corr_m2 vs corr_m1.
        5. Count features where M1 > M2.

    Returns:
        metrics[(present_mods_tuple, target_mod)] = {
            'corr_m1': Series,
            'corr_m2': Series,
            'n_features': int,
            'n_better_mae': int
        }
    """

    if type(method1)==str:
        # Load the method 1 predictions
        with open(method1, "rb") as f:
            m1_preds = pickle.load(f)
    else:
        m1_preds = method1

    if type(method2)==str:
        # Load the method2 predictions
        with open(method2, "rb") as f:
            m2_preds = pickle.load(f)
    else:
        m2_preds = method2

    results = {}

    for key, df_m2 in m2_preds.items():
        present_mods, target_mod = key

        if key not in m1_preds:
            print(f"[WARN] Method 1 predictions missing scenario {key}; skipping.")
            continue

        df_m1 = m1_preds[key]

        # Ground truth
        df_true_full = multi_omic_data[target_mod]

        # Align samples and features jointly across all three
        samples = df_m2.index.intersection(df_m1.index).intersection(df_true_full.index)
        features = (
            df_m2.columns
            .intersection(df_m1.columns)
            .intersection(df_true_full.columns)
        )

        if len(samples) == 0 or len(features) == 0:
            print(f"[WARN] No overlap for scenario {key}; skipping.")
            continue

        df_true = df_true_full.loc[samples, features]
        df_m1 = df_m1.loc[samples, features]
        df_m2 = df_m2.loc[samples, features]

        # Compute per-feature correlations
        corr_m1 = per_feature_corr(df_true, df_m1)
        corr_m2 = per_feature_corr(df_true, df_m2)

        # Count features where MAE beats KNN
        both = pd.concat([corr_m1.rename(m1_name), corr_m2.rename(m2_name)], axis=1)
        both["better_method1"] = both[m1_name] > both[m2_name]
        n_better_m1 = int(both["better_method1"].sum())/len(both)

        # Save results
        results[key] = {
            "corr_"+m1_name: corr_m1,
            "corr_"+m2_name: corr_m2,
            "n_features": len(features),
            "n_better_"+m1_name: n_better_m1,
        }

        # Plot scatter
        if plot_scatter:
            plt.figure(figsize=(10, 6))
            plt.scatter(
                both[m2_name], both[m1_name],
                s=12, alpha=0.5
            )
            lims = [
                min(both.min().min(), -1),
                max(both.max().max(), 1),
            ]
            plt.plot(lims, lims, "k--", linewidth=1)
            plt.xlim(lims)
            plt.ylim(lims)
            plt.xlabel(m2_name+" Per-Feature Correlation")
            plt.ylabel(m1_name+" Per-Feature Correlation")
            plt.title(
                f"Scenario: present={present_mods}, target={target_mod}\n"
                f"Features better in {m1_name}: {n_better_m1}"
            )
            plt.tight_layout()
            plt.show()

    return results

    ###################################################################
    # For missing values evaluation
    ###################################################################

def evaluate_values_imputation(
    pred_dfs: Dict[str, pd.DataFrame],
    mask_dfs: Dict[str, pd.DataFrame],
    multi_omic_data: Dict[str, pd.DataFrame],
    corrupt_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    evaluate_on: Literal["masked", "observed", "all"] = "masked",
    plot_scatter: bool = True,
    max_points_plot: int = 5000,
    seed: int = 0,
    use_kde_if_available: bool = True,
    title: str = "MCAR"
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate reconstructions from mask_and_predict using the saved pickles.

    By default evaluates ONLY the artificially masked entries (evaluate_on="masked").

    Args:
        pred_dfs: {mod: DataFrame} reconstructed values (full matrix)
        mask_dfs: {mod: DataFrame(bool)} True where artificially masked
        multi_omic_data: {mod: DataFrame} ground truth full data (may include natural NaNs)
        corrupt_dfs: optional {mod: DataFrame} corrupted data (NaNs at masked entries)
                    If provided, we can more robustly define "observed" entries.
        evaluate_on:
            - "masked": evaluate only entries where mask==True
            - "observed": evaluate only entries where mask==False AND (optionally corrupt is finite)
            - "all": evaluate all entries (finite on both sides), ignoring mask
        plot_scatter: if True, scatter plot true vs imputed for each modality
        max_points_plot: subsample for plotting
        seed: RNG seed for plotting subsample
        use_kde_if_available: if True and SciPy available, color points by KDE density

    Returns:
        metrics_by_mod: {mod: {"mse","pearson","spearman","n_points"}}
    """
    rng = np.random.default_rng(seed)
    metrics_by_mod: Dict[str, Dict[str, Any]] = {}

    for mod, df_pred in pred_dfs.items():
        if mod not in multi_omic_data:
            print(f"[WARN] Modality '{mod}' not in multi_omic_data; skipping.")
            continue
        if mod not in mask_dfs:
            print(f"[WARN] Modality '{mod}' not in mask_dfs; skipping.")
            continue

        df_true_full = multi_omic_data[mod]
        df_mask_full = mask_dfs[mod]

        # Align indices/columns across pred, true, mask (and corrupt if provided)
        common_samples = df_pred.index.intersection(df_true_full.index).intersection(df_mask_full.index)
        common_features = df_pred.columns.intersection(df_true_full.columns).intersection(df_mask_full.columns)

        if len(common_samples) == 0 or len(common_features) == 0:
            print(f"[WARN] No overlap in samples/features for mod '{mod}'; skipping.")
            continue

        df_pred_aligned = df_pred.loc[common_samples, common_features]
        df_true_aligned = df_true_full.loc[common_samples, common_features]
        df_mask_aligned = df_mask_full.loc[common_samples, common_features].astype(bool)

        df_cor_aligned = None
        if corrupt_dfs is not None and mod in corrupt_dfs:
            df_cor = corrupt_dfs[mod]
            common_samples2 = common_samples.intersection(df_cor.index)
            common_features2 = common_features.intersection(df_cor.columns)
            if len(common_samples2) > 0 and len(common_features2) > 0:
                # re-align everything to corrupt’s overlap as well
                common_samples = common_samples2
                common_features = common_features2
                df_pred_aligned = df_pred.loc[common_samples, common_features]
                df_true_aligned = df_true_full.loc[common_samples, common_features]
                df_mask_aligned = df_mask_full.loc[common_samples, common_features].astype(bool)
                df_cor_aligned = df_cor.loc[common_samples, common_features]

        y_true_mat = df_true_aligned.values
        y_pred_mat = df_pred_aligned.values
        m_mat = df_mask_aligned.values

        # Decide which entries to evaluate
        if evaluate_on == "masked":
            eval_mask = m_mat
        elif evaluate_on == "observed":
            # If corrupted is available, observed means not masked AND corrupt is finite
            # (natural NaNs are excluded later anyway)
            if df_cor_aligned is not None:
                cor_mat = df_cor_aligned.values
                eval_mask = (~m_mat) & np.isfinite(cor_mat)
            else:
                eval_mask = ~m_mat
        elif evaluate_on == "all":
            eval_mask = np.ones_like(m_mat, dtype=bool)
        else:
            raise ValueError(f"Unknown evaluate_on: {evaluate_on}")

        # Also require finite on both true and pred
        finite_mask = np.isfinite(y_true_mat) & np.isfinite(y_pred_mat)
        eval_mask = eval_mask & finite_mask

        y_true = y_true_mat[eval_mask].reshape(-1)
        y_pred = y_pred_mat[eval_mask].reshape(-1)

        n_points = int(y_true.shape[0])
        if n_points == 0:
            print(f"[WARN] No finite eval points for mod '{mod}' (evaluate_on={evaluate_on}); skipping.")
            continue

        mse = float(np.mean((y_true - y_pred) ** 2))
        s_true = pd.Series(y_true)
        s_pred = pd.Series(y_pred)
        pearson = float(s_true.corr(s_pred, method="pearson"))
        spearman = float(s_true.corr(s_pred, method="spearman"))

        metrics_by_mod[mod] = {
            "mse": mse,
            "pearson": pearson,
            "spearman": spearman,
            "n_points": n_points,
        }

        # Plot
        if plot_scatter:
            if n_points > max_points_plot:
                idx = rng.choice(n_points, size=max_points_plot, replace=False)
                x_plot = y_true[idx]
                y_plot = y_pred[idx]
            else:
                x_plot = y_true
                y_plot = y_pred

            plt.figure(figsize=(6, 6))

            if use_kde_if_available:
                xy = np.vstack([x_plot, y_plot])
                kde = gaussian_kde(xy)
                density = np.log(kde(xy))
                order = density.argsort()
                x_plot = x_plot[order]
                y_plot = y_plot[order]
                density = density[order]
                plt.scatter(x_plot, y_plot, c=density, s=4, alpha=0.8, linewidths=0, cmap="OrRd")
            else:
                plt.scatter(x_plot, y_plot, s=4, alpha=0.35, linewidths=0)


            # --- make x/y scales identical + add y=x line ---
            lo = np.nanmin([x_plot.min(), y_plot.min()])
            hi = np.nanmax([x_plot.max(), y_plot.max()])
            plt.xlim(lo, hi)
            plt.ylim(lo, hi)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color='black')

            plt.xlabel("Observed values", fontsize=18)
            plt.ylabel("Imputed values", fontsize=18)
            plt.title(
                f"{mod} ({title})\n"
                f"r={pearson:.3f}",
                fontsize=14,
            )
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.show()

    return metrics_by_mod

