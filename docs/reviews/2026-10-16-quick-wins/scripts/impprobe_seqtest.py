# Correctness test: ice_seq must reproduce sklearn IterativeImputer(BayesianRidge).transform on numeric-only tables.
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, sys.argv[1])
import warnings; warnings.filterwarnings("ignore")
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import impprobe_arms as A
from scripts.backfill_baseline_imputers import recompute_folds

def factory(train, num, cat):
    ice = A.MiceSeqImputer(num, cat); ice.fit(train)
    Xtr = train[num].to_numpy(float)
    sk = IterativeImputer(estimator=BayesianRidge(), max_iter=int(sys.argv[2]), random_state=0).fit(Xtr)
    print("  template kept rounds", ice.rounds, "fitted rounds", len(ice._seq), "sklearn n_iter_", sk.n_iter_)
    def score(test, cells, naive):
        rows = cells["row"].to_numpy(int); cols = test.columns.get_indexer(cells["column"])
        hidden = np.zeros(test.shape, bool); hidden[rows, cols] = True
        X = test[num].to_numpy(float).copy(); X[hidden | np.isnan(X)] = np.nan
        a = ice.impute_seq(test, hidden)[num].to_numpy(float); b = sk.transform(X)
        print("  ", cells["population"].iloc[0], "max |ice_seq - sklearn| over all cells", float(np.abs(a - b).max()))
        return {}
    return score

for v in sys.argv[3:]:
    print(v)
    recompute_folds(v, 42, 5, 0.2, [], fold_scorer=factory, done={2: {}, 3: {}, 4: {}, 5: {}})
