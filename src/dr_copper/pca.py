import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PCA_FEATURE_COLS: list[str] = [
    "copper_ret1d",
    "copper_ret5d",
    "copper_vol21d",
    "gold_ret1d",
    "gold_ret5d",
    "crude_ret1d",
    "crude_ret5d",
    "sp500_ret1d",
    "sp500_ret5d",
    "dxy_ret1d",
    "dxy_ret5d",
    "cli_china",
    "cli_usa",
    "DFII10",
    "USDCNY",
]


def build_pca_pipeline(n_components=0.8) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )


def fit_pca(
    features: pd.DataFrame,
    n_components=0.8,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, np.ndarray]:
    X = features[PCA_FEATURE_COLS]

    pipe = build_pca_pipeline(n_components=n_components)
    scores_arr = pipe.fit_transform(X)

    pca: PCA = pipe.named_steps["pca"]

    pc_labels = [f"PC{i + 1}" for i in range(len(scores_arr[0]))]

    loadings = pd.DataFrame(
        pca.components_.T,  # shape: (n_features, n_components)
        index=X.columns,
        columns=pc_labels,
    )

    scores = pd.DataFrame(
        scores_arr,
        index=X.index,
        columns=pc_labels,
    )

    evr = pca.explained_variance_ratio_

    return pipe, loadings, scores, evr


def transform_pca(features: pd.DataFrame, pipe: Pipeline):
    X = features[PCA_FEATURE_COLS]

    scores_arr = pipe.transform(X)

    pc_labels = [f"PC{i + 1}" for i in range(len(scores_arr[0]))]

    scores = pd.DataFrame(
        scores_arr,
        index=X.index,
        columns=pc_labels,
    )

    return scores
