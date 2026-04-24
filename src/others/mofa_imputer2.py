"""
pred_dfs = impute_values_from_corrupt(
    corrupt_pickle_path="corrupt.pkl",
    multi_omic_data=multi_omic_data,
    mofa_hdf5_path="global_mofa.hdf5",
    views_order=["rna", "atac", "protein"],
    use_multi_view_projection=True,
    only_na=True,
)

imputer = MOFAGlobalImputer(
    hdf5_path="global_mofa.hdf5",
    multi_omic_data=multi_omic_data,
    views_order=["rna", "atac", "protein"],
    use_multi_view_projection=True,
)

imputed_df = imputer.impute_for_scenario(scenario_payload)
"""
import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from mofapy2.run.entry_point import entry_point
import mofax as mofa


# ----------------------------------------------------------------------
# 1) TRAINING: global MOFA model on training samples only
# ----------------------------------------------------------------------


def train_global_mofa(
    multi_omic_data: Dict[str, pd.DataFrame],
    train_samples: List[str],
    out_hdf5_path: str,
    views_order: Optional[List[str]] = None,
    n_factors: Optional[int] = None,
    train_iter: Optional[int] = None,
    seed: Optional[int] = None,
    use_float32: bool = True,
    verbose: bool = True,
) -> str:
    """
    Train a global MOFA model using mofapy2 on training samples only,
    and save it as an HDF5 file.

    Args
    ----
    multi_omic_data
        dict(modality -> DataFrame [samples x features]) with full dataset.
    train_samples
        Sample IDs to use for MOFA training.
    out_hdf5_path
        Path to write the MOFA model HDF5 file.
    views_order
        Optional explicit order of modalities (views). If None, uses
        sorted(multi_omic_data.keys()).
    n_factors
        If provided, overrides the default number of factors.
    train_iter
        If provided, overrides the default number of iterations.
    seed
        Optional random seed for training.
    use_float32
        Cast data to float32 before sending to MOFA.
    verbose
        Print progress messages.

    Returns
    -------
    out_hdf5_path
        The path to the saved MOFA model.
    """
    if views_order is None:
        views_order = sorted(multi_omic_data.keys())

    for mod, df in multi_omic_data.items():
        missing = set(train_samples) - set(df.index)
        if missing:
            raise ValueError(
                f"Some train_samples are missing in modality '{mod}': "
                f"{sorted(list(missing))[:5]} ..."
            )

    if verbose:
        print("[MOFA train] Views:", views_order)
        print("[MOFA train] N_train:", len(train_samples))

    M = len(views_order)
    G = 1
    data_mat = [[None for _ in range(G)] for _ in range(M)]

    for m, mod in enumerate(views_order):
        df = multi_omic_data[mod].loc[train_samples]
        X = df.to_numpy()
        if use_float32:
            X = X.astype(np.float32)
        data_mat[m][0] = X

    ent = entry_point()

    ent.set_data_options(
        scale_views=False,
    )

    likelihoods = ["gaussian"] * M

    ent.set_data_matrix(
        data_mat,
        likelihoods=likelihoods,
    )

    if n_factors is not None:
        ent.set_model_options(factors=n_factors)
    else:
        ent.set_model_options()

    train_kwargs = {}
    if train_iter is not None:
        train_kwargs["iter"] = train_iter
    if seed is not None:
        train_kwargs["seed"] = seed

    if train_kwargs:
        ent.set_train_options(**train_kwargs)
    else:
        ent.set_train_options()

    if verbose:
        print("[MOFA train] Building model...")
    ent.build()

    if verbose:
        print("[MOFA train] Running model...")
    ent.run()

    out_dir = os.path.dirname(out_hdf5_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ent.save(out_hdf5_path, save_data=True)

    if verbose:
        print(f"[MOFA train] Model saved to {out_hdf5_path}")

    return out_hdf5_path


# ----------------------------------------------------------------------
# 2) IMPUTATION: one class for
#    - missing modality imputation
#    - missing value imputation from corrupted data
# ----------------------------------------------------------------------


class MOFAGlobalImputer:
    """
    MOFA-based imputer using a single global pretrained model.

    Supports two related tasks:

    1) Missing modality imputation:
       - infer latent factors from present modalities
       - reconstruct one completely missing target modality

    2) Missing value imputation:
       - infer latent factors from corrupted input matrices
       - reconstruct requested modalities
       - fill only NaN entries (or optionally return full reconstructions)
    """

    def __init__(
        self,
        hdf5_path: str,
        multi_omic_data: Dict[str, pd.DataFrame],
        views_order: Optional[List[str]] = None,
        projection_view: Optional[str] = None,
        use_multi_view_projection: bool = False,
        verbose: bool = False,
    ):
        """
        Args
        ----
        hdf5_path
            Path to the global MOFA model HDF5.
        multi_omic_data
            Full dataset as dict(modality -> DataFrame [samples x features]).
            Used for scenario-based missing modality imputation and for
            feature-name / ordering reference.
        views_order
            Optional explicit order of modalities (views). If None, uses
            sorted(multi_omic_data.keys()).
        projection_view
            Optional modality to use for single-view projection.
        use_multi_view_projection
            If False: project with one view using mofax.project_data().
            If True: combine present views using least-squares projection.
        verbose
            Print progress information.
        """
        self.hdf5_path = hdf5_path
        self.multi_omic_data = multi_omic_data
        self.views_order = (
            list(views_order) if views_order is not None else sorted(multi_omic_data.keys())
        )
        self.projection_view = projection_view
        self.use_multi_view_projection = use_multi_view_projection
        self.verbose = verbose

        self.model = mofa.mofa_model(hdf5_path)

        # Must match training view order
        self.mod_to_view_index = {
            mod: i for i, mod in enumerate(self.views_order)
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _choose_projection_view(self, modalities_present: List[str]) -> str:
        if self.projection_view is not None:
            if self.projection_view not in modalities_present:
                raise ValueError(
                    f"projection_view='{self.projection_view}' not in "
                    f"modalities_present={modalities_present}"
                )
            return self.projection_view

        for mod in self.views_order:
            if mod in modalities_present:
                return mod

        raise ValueError(
            f"No projection view available: modalities_present={modalities_present}, "
            f"views_order={self.views_order}"
        )

    def _project_single_view_from_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        samples: List[str],
        proj_view: str,
    ) -> pd.DataFrame:
        """
        Project samples onto MOFA latent space using a single modality from
        an arbitrary data dict.

        Missing entries are filled with 0.0 before projection, which is
        reasonable if features are centered / z-scored.
        """
        df_view = data_dict[proj_view].loc[samples].copy()
        df_view = df_view.fillna(0.0)

        view_idx = self.mod_to_view_index[proj_view]

        Z_new = self.model.project_data(
            data=df_view,
            view=view_idx,
            df=True,
            feature_intersection=False,
        )
        return Z_new

    def _project_multi_view_from_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        samples: List[str],
        modalities_present: List[str],
    ) -> pd.DataFrame:
        """
        Multi-view least-squares projection:

          Z_new = (sum_v X_new^(v) W^(v)) (sum_v W^(v)T W^(v))^{-1}

        Assumes feature order in each data_dict[view].columns matches the order
        used when training MOFA and when retrieving weights.
        """
        XW_sum = None
        WtW_sum = None
        Z_index = None
        Z_columns = None

        for mod in modalities_present:
            df_view = data_dict[mod].loc[samples].copy()
            df_view = df_view.fillna(0.0)

            view_idx = self.mod_to_view_index[mod]
            W_v_df = self.model.get_weights(views=view_idx, df=True)

            X_v = df_view.to_numpy()
            W_v = W_v_df.to_numpy()

            if X_v.shape[1] != W_v.shape[0]:
                raise ValueError(
                    f"Shape mismatch for view '{mod}': "
                    f"X_v has {X_v.shape[1]} features, W_v has {W_v.shape[0]} rows."
                )

            XW = X_v @ W_v
            WtW = W_v.T @ W_v

            if XW_sum is None:
                XW_sum = XW
                WtW_sum = WtW
                Z_index = df_view.index
                Z_columns = list(W_v_df.columns)
            else:
                XW_sum += XW
                WtW_sum += WtW

        if XW_sum is None or WtW_sum is None:
            raise ValueError(
                "Multi-view projection failed: no usable views "
                "with matching shapes for the present modalities."
            )

        eps = 1e-6
        WtW_reg = WtW_sum + eps * np.eye(WtW_sum.shape[0])
        Z_new = XW_sum @ np.linalg.inv(WtW_reg)

        return pd.DataFrame(Z_new, index=Z_index, columns=Z_columns)

    def _get_Z_from_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        samples: List[str],
        modalities_present: List[str],
    ) -> pd.DataFrame:
        """
        Infer latent factors for samples from an arbitrary data dict.
        """
        if self.use_multi_view_projection:
            return self._project_multi_view_from_data(
                data_dict=data_dict,
                samples=samples,
                modalities_present=modalities_present,
            )

        proj_view = self._choose_projection_view(modalities_present)
        if self.verbose:
            print(f"[MOFA] Using projection view: {proj_view}")

        return self._project_single_view_from_data(
            data_dict=data_dict,
            samples=samples,
            proj_view=proj_view,
        )

    def _reconstruct_view(
        self,
        Z_new: pd.DataFrame,
        target_mod: str,
        samples: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Reconstruct one modality from latent factors:

          Y_hat = Z_new @ W_target^T

        Factors are aligned by column name.
        Features are assumed to be aligned by position.
        """
        view_idx = self.mod_to_view_index[target_mod]
        W_target_df = self.model.get_weights(views=view_idx, df=True)

        common_factors = Z_new.columns.intersection(W_target_df.columns)
        if len(common_factors) == 0:
            raise ValueError(
                f"No overlapping factors between Z_new and weights for '{target_mod}'."
            )

        Z_use = Z_new[common_factors].to_numpy()
        W_use = W_target_df[common_factors].to_numpy()

        if feature_names is None:
            feature_names = list(self.multi_omic_data[target_mod].columns)

        if W_use.shape[0] != len(feature_names):
            raise ValueError(
                f"Feature count mismatch for target '{target_mod}': "
                f"W has {W_use.shape[0]} rows, but data has {len(feature_names)} columns."
            )

        Y_hat = Z_use @ W_use.T

        return pd.DataFrame(
            Y_hat,
            index=samples,
            columns=feature_names,
        )

    # ------------------------------------------------------------------
    # Public API 1: missing modality imputation
    # ------------------------------------------------------------------

    def impute_for_scenario(self, scenario_payload: dict) -> pd.DataFrame:
        """
        Impute the missing modality for a single scenario.

        scenario_payload structure:
          - "modalities_present": List[str]
          - "missing_modality": str
          - "samples": List[str]
          - "data": Dict[str, DataFrame] (optional; not required here)

        Returns
        -------
        imputed_df : DataFrame [scenario_samples x features_target_mod]
        """
        modalities_present: List[str] = scenario_payload["modalities_present"]
        target_mod: str = scenario_payload["missing_modality"]
        scenario_samples: List[str] = scenario_payload["samples"]

        if target_mod not in self.multi_omic_data:
            raise ValueError(f"Target modality '{target_mod}' not found in multi_omic_data.")

        if self.verbose:
            print(
                f"[MOFA impute scenario] present={modalities_present}, "
                f"target={target_mod}, n_scenario={len(scenario_samples)}"
            )

        Z_new = self._get_Z_from_data(
            data_dict=self.multi_omic_data,
            samples=scenario_samples,
            modalities_present=modalities_present,
        )

        imputed_df = self._reconstruct_view(
            Z_new=Z_new,
            target_mod=target_mod,
            samples=scenario_samples,
            feature_names=list(self.multi_omic_data[target_mod].columns),
        )
        return imputed_df

    # ------------------------------------------------------------------
    # Public API 2: missing value imputation
    # ------------------------------------------------------------------

    def transform(
        self,
        data_corrupted: Dict[str, pd.DataFrame],
        samples: Optional[List[str]] = None,
        use_modalities: Optional[List[str]] = None,
        only_na: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Impute missing values in corrupted data using the pretrained MOFA model.

        Args
        ----
        data_corrupted
            dict(modality -> DataFrame) with NaNs marking missing/corrupted values.
        samples
            Optional explicit sample order. If None, uses the index of the first
            modality in use_modalities.
        use_modalities
            Which modalities to use. If None, uses the intersection of
            views_order and data_corrupted keys.
        only_na
            If True, preserve observed values and fill only NaNs.
            If False, return full MOFA reconstructions.

        Returns
        -------
        pred_dfs
            dict(modality -> DataFrame), same shape/layout as input for each used modality.
        """
        if use_modalities is None:
            use_modalities = [m for m in self.views_order if m in data_corrupted]

        if len(use_modalities) == 0:
            raise ValueError("No usable modalities found in data_corrupted.")

        missing_mods = [m for m in use_modalities if m not in data_corrupted]
        if missing_mods:
            raise ValueError(f"Missing modalities in data_corrupted: {missing_mods}")

        if samples is None:
            first_mod = use_modalities[0]
            samples = list(data_corrupted[first_mod].index)

        if self.verbose:
            print(
                f"[MOFA transform] modalities={use_modalities}, "
                f"n_samples={len(samples)}, only_na={only_na}"
            )

        # Important: infer Z from the corrupted input, not from the original data
        Z_new = self._get_Z_from_data(
            data_dict=data_corrupted,
            samples=samples,
            modalities_present=use_modalities,
        )

        pred_dfs: Dict[str, pd.DataFrame] = {}

        for mod in use_modalities:
            original = data_corrupted[mod].loc[samples].copy()

            recon = self._reconstruct_view(
                Z_new=Z_new,
                target_mod=mod,
                samples=samples,
                feature_names=list(original.columns),
            )

            if only_na:
                mask = original.isna()
                filled = original.copy()
                filled[mask] = recon[mask]
            else:
                filled = recon

            pred_dfs[mod] = filled

        return pred_dfs


# ----------------------------------------------------------------------
# 3) Convenience wrapper: loop over scenarios (missing modality)
# ----------------------------------------------------------------------


def translate_from_scenario_dir(
    scenarios_dir: str,
    mofa_hdf5_path: str,
    multi_omic_data: Dict[str, pd.DataFrame],
    views_order: Optional[List[str]] = None,
    projection_view: Optional[str] = None,
    use_multi_view_projection: bool = True,
    verbose: bool = False,
    save_pred_pickle_path: Optional[str] = None,
) -> Dict[Tuple[Tuple[str, ...], str], pd.DataFrame]:
    """
    Loop over all scenario pickles in a directory and impute the missing modality
    for each scenario using a global MOFA model.

    Returns
    -------
    predictions
        dict:
          keys   = (tuple(sorted_present_mods), target_mod)
          values = DataFrame [scenario_samples x features_target_mod]
    """
    imputer = MOFAGlobalImputer(
        hdf5_path=mofa_hdf5_path,
        multi_omic_data=multi_omic_data,
        views_order=views_order,
        projection_view=projection_view,
        use_multi_view_projection=use_multi_view_projection,
        verbose=verbose,
    )

    predictions: Dict[Tuple[Tuple[str, ...], str], pd.DataFrame] = {}

    for fname in sorted(os.listdir(scenarios_dir)):
        if not fname.endswith(".pkl"):
            continue

        path = os.path.join(scenarios_dir, fname)
        with open(path, "rb") as f:
            scenario_payload = pickle.load(f)

        modalities_present: List[str] = scenario_payload["modalities_present"]
        target_mod: str = scenario_payload["missing_modality"]

        if verbose:
            print(f"\n[MOFA scenario] File: {fname}")

        imputed_df = imputer.impute_for_scenario(scenario_payload)

        key = (tuple(sorted(modalities_present)), target_mod)
        predictions[key] = imputed_df

    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(predictions, f)
        if verbose:
            print(f"[MOFA scenario] Saved predictions to {save_pred_pickle_path}")

    return predictions


# ----------------------------------------------------------------------
# 4) Convenience wrapper: corrupted value imputation
# ----------------------------------------------------------------------


def impute_values_from_corrupt(
    corrupt_pickle_path: str,
    multi_omic_data: Dict[str, pd.DataFrame],
    mofa_imputer: Optional[MOFAGlobalImputer] = None,
    mofa_hdf5_path: Optional[str] = None,
    views_order: Optional[List[str]] = None,
    projection_view: Optional[str] = None,
    use_multi_view_projection: bool = True,
    use_modalities: Optional[List[str]] = None,
    samples: Optional[List[str]] = None,
    only_na: bool = True,
    save_pred_pickle_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Load a corrupted-data pickle and impute its NaNs using a pretrained global MOFA model.

    Args
    ----
    corrupt_pickle_path
        Path to pickle containing {modality: DataFrame} with NaNs.
    multi_omic_data
        Full reference dataset used for feature ordering / metadata.
    mofa_imputer
        Optional preconstructed MOFAGlobalImputer.
    mofa_hdf5_path
        Path to pretrained MOFA model. Used only if mofa_imputer is None.
    views_order, projection_view, use_multi_view_projection, verbose
        Used to construct MOFAGlobalImputer if needed.
    use_modalities
        Which modalities to impute. If None, uses the intersection of
        corrupt_dfs keys and multi_omic_data keys.
    samples
        Optional sample order.
    only_na
        If True, fill only NaNs and preserve observed values.
        If False, return full reconstructions.
    save_pred_pickle_path
        Optional path to save predictions dict.

    Returns
    -------
    pred_dfs
        dict(modality -> DataFrame) of imputed values.
    """
    with open(corrupt_pickle_path, "rb") as f:
        corrupt_dfs: Dict[str, pd.DataFrame] = pickle.load(f)

    if use_modalities is None:
        use_modalities = [m for m in corrupt_dfs.keys() if m in multi_omic_data]

    if mofa_imputer is None:
        if mofa_hdf5_path is None:
            raise ValueError("Provide either mofa_imputer or mofa_hdf5_path.")

        mofa_imputer = MOFAGlobalImputer(
            hdf5_path=mofa_hdf5_path,
            multi_omic_data=multi_omic_data,
            views_order=views_order,
            projection_view=projection_view,
            use_multi_view_projection=use_multi_view_projection,
            verbose=verbose,
        )

    pred_dfs = mofa_imputer.transform(
        data_corrupted=corrupt_dfs,
        samples=samples,
        use_modalities=use_modalities,
        only_na=only_na,
    )

    if save_pred_pickle_path is not None:
        with open(save_pred_pickle_path, "wb") as f:
            pickle.dump(pred_dfs, f)
        if verbose:
            print(f"[Saved MOFA predictions] {save_pred_pickle_path}")

    return pred_dfs