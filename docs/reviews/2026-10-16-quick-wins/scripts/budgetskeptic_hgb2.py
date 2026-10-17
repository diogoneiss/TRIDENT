"""Scoring-only: induced impute_score of the hgb recipe at a few booster settings, on a
variant's own folds and induced cells. No MLflow, no baseline cache."""
import contextlib, io, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR

sys.path.insert(0, r"C:\Users\DIOGON~1\AppData\Local\Temp\claude\C--Users-Diogo-Neiss-Documents-Mestrado-Wiki\f402d390-da5f-4f09-a187-282102e58959\scratchpad")
import budgetskeptic_knn as base  # sets ROOT on sys.path and chdir

from src.training.imputation_baselines import BOOSTING_SEED, GradientBoostingImputer, MeanModeImputer, _Constant, score_baselines
from src.training.imputation_metrics import mean_mode_baselines
from src.training.data import build_folds, in_variant_dtypes, load_complete_sibling, prepare_dataset
from src.training.types import DatasetSpec

SETTINGS = {
    "hgb_default": {},
    "hgb_slow": {"learning_rate": 0.05, "max_iter": 300},
    "hgb_reg": {"max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0},
    "hgb_shallow": {"learning_rate": 0.05, "max_iter": 200, "max_depth": 3},
}


def make(name, params):
    def _fit_one(numerical, features, target, categorical):
        if len(target) == 0:
            return _Constant(0.0)
        values, counts = np.unique(target, return_counts=True)
        if len(values) == 1:
            return _Constant(float(values[0]))
        if features.shape[1] == 0:
            return _Constant(float(target.mean()) if numerical else float(values[counts.argmax()]))
        declared = categorical if categorical.any() else None
        if numerical:
            return HR(categorical_features=declared, random_state=BOOSTING_SEED, **params).fit(features, target)
        return HC(categorical_features=declared, random_state=BOOSTING_SEED, **params).fit(features, target.astype(int))
    return type(name, (GradientBoostingImputer,), {"name": name, "_fit_one": staticmethod(_fit_one)})


def main(variant, seed, names):
    with contextlib.redirect_stdout(io.StringIO()):
        spec = DatasetSpec.from_name(variant, None)
        dataset = prepare_dataset(spec)
        sibling = load_complete_sibling(spec)
    features = dataset.frame.drop(columns=[dataset.label_column])
    numerical, categorical = list(dataset.numerical_columns), list(dataset.categorical_columns)
    per = {}
    t0 = time.time()
    for ordinal, fold in enumerate(build_folds(dataset.frame, dataset.label_column, 5, seed), start=1):
        train = features.iloc[np.asarray(fold.train_indices)].reset_index(drop=True)
        test = features.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)
        naive = mean_mode_baselines(train, numerical, categorical)
        imputers = [MeanModeImputer(numerical, categorical)] + [make(n, SETTINGS[n])(numerical, categorical) for n in names]
        for imp in imputers:
            imp.fit(train)
        columns = list(test.columns)
        truth = in_variant_dtypes(sibling.iloc[np.asarray(fold.test_indices)].reset_index(drop=True)[columns].copy(), test, categorical)
        if dataset.scaler is not None and numerical:
            truth[numerical] = dataset.scaler.transform(truth[numerical])
        gaps = (test.isna() & truth.notna()).to_numpy()
        tn, tc = base.frame_values(truth, dataset)
        cells = base.cells_table(gaps, columns, tn, tc, "induced")
        metrics = score_baselines(imputers, test, cells, naive).metrics
        for imp in imputers:
            per.setdefault(imp.name, []).append(metrics[f"baseline/{imp.name}/impute_score"])
        print(f"  fold {ordinal} done {time.time() - t0:.0f}s", flush=True)
    print(variant, "seed", seed)
    for k, v in per.items():
        print(f"  {k:12s} mean {np.mean(v):.4f}  folds {[round(x, 4) for x in v]}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","))
