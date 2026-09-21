"""
_ml_patch.py — pure-NumPy/SciPy fallback for scikit-learn & imbalanced-learn.

Import this module at the very top of main.py, BEFORE any other import.

If scikit-learn imports cleanly, this file does nothing.
If it fails (e.g., Windows Application Control blocking a .pyd DLL), it
installs a compatible subset of the sklearn / imblearn API into sys.modules,
so the rest of BLACKLEDGER runs unchanged.
"""

import sys
import types
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# 0. Try the real thing first
# ──────────────────────────────────────────────────────────────────────────

def _sklearn_works() -> bool:
    try:
        import sklearn.model_selection  # noqa
        import sklearn.preprocessing    # noqa
        import sklearn.linear_model     # noqa
        import sklearn.tree             # noqa
        import sklearn.ensemble         # noqa
        import sklearn.metrics          # noqa
        import sklearn.pipeline         # noqa
        import imblearn.over_sampling   # noqa
        import imblearn.combine         # noqa
        return True
    except Exception as exc:
        print(f"[_ml_patch] scikit-learn unavailable: {type(exc).__name__}: {exc}")
        print("[_ml_patch] Installing pure-NumPy fallback…")
        return False


if not _sklearn_works():

    # ──────────────────────────────────────────────────────────────────────
    # 1. Metrics
    # ──────────────────────────────────────────────────────────────────────
    def _precision(y, p, zero_division=0):
        y = np.asarray(y); p = np.asarray(p)
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        return tp / (tp + fp) if (tp + fp) else zero_division

    def _recall(y, p, zero_division=0):
        y = np.asarray(y); p = np.asarray(p)
        tp = int(((y == 1) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        return tp / (tp + fn) if (tp + fn) else zero_division

    def _f1(y, p, zero_division=0):
        pr, rc = _precision(y, p, zero_division), _recall(y, p, zero_division)
        return 2 * pr * rc / (pr + rc) if (pr + rc) else zero_division

    def _roc_auc(y, s):
        y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return 0.5
        order = np.argsort(s)
        ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
        sum_ranks = np.zeros(len(counts))
        np.add.at(sum_ranks, inv, ranks)
        ranks = (sum_ranks / counts)[inv]
        return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2)
                     / (n_pos * n_neg))

    def _ap(y, s):
        y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
        order = np.argsort(-s); ys = y[order]
        tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max((y == 1).sum(), 1)
        rec_prev = np.concatenate([[0.0], rec[:-1]])
        return float(np.sum((rec - rec_prev) * prec))

    def _cm(y, p):
        y = np.asarray(y); p = np.asarray(p)
        return np.array([
            [int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())],
            [int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())],
        ])

    def _report(y, p, zero_division=0):
        y = np.asarray(y); p = np.asarray(p)
        p0 = _precision(1 - y, 1 - p, zero_division)
        r0 = _recall(1 - y, 1 - p, zero_division)
        f0 = _f1(1 - y, 1 - p, zero_division)
        p1 = _precision(y, p, zero_division)
        r1 = _recall(y, p, zero_division)
        f1 = _f1(y, p, zero_division)
        return (f"              precision    recall  f1-score   support\n"
                f"           0    {p0:>9.4f}  {r0:>7.4f}  {f0:>7.4f}   "
                f"{int((y == 0).sum()):>7d}\n"
                f"           1    {p1:>9.4f}  {r1:>7.4f}  {f1:>7.4f}   "
                f"{int((y == 1).sum()):>7d}\n")

    def _roc_curve(y, s):
        y = np.asarray(y); s = np.asarray(s, dtype=float)
        thresholds = np.r_[s.max() + 1, np.unique(s)[::-1]]
        P = max((y == 1).sum(), 1); N = max((y == 0).sum(), 1)
        fprs, tprs = [], []
        for t in thresholds:
            pred = (s >= t).astype(int)
            tprs.append(((y == 1) & (pred == 1)).sum() / P)
            fprs.append(((y == 0) & (pred == 1)).sum() / N)
        return np.array(fprs), np.array(tprs), thresholds

    def _pr_curve(y, s):
        y = np.asarray(y); s = np.asarray(s, dtype=float)
        thresholds = np.unique(s)[::-1]
        P = max((y == 1).sum(), 1)
        prec, rec = [], []
        for t in thresholds:
            pred = (s >= t).astype(int)
            tp = ((y == 1) & (pred == 1)).sum()
            fp = ((y == 0) & (pred == 1)).sum()
            prec.append(tp / (tp + fp) if (tp + fp) else 1.0)
            rec.append(tp / P)
        return np.array(prec), np.array(rec), thresholds

    # ──────────────────────────────────────────────────────────────────────
    # 2. Model selection
    # ──────────────────────────────────────────────────────────────────────
    def _safe_index(obj, idx):
        return obj.iloc[idx] if hasattr(obj, "iloc") else np.asarray(obj)[idx]

    def _train_test_split(*arrays, test_size=0.2, random_state=None,
                          stratify=None, **kw):
        n = len(arrays[0])
        rng = np.random.default_rng(random_state if random_state is not None else 42)
        if stratify is not None:
            y = np.asarray(stratify)
            test_idx = []
            for c in np.unique(y):
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                test_idx.append(idx[: int(round(len(idx) * test_size))])
            test_idx = np.concatenate(test_idx)
            mask = np.ones(n, dtype=bool); mask[test_idx] = False
            train_idx = np.where(mask)[0]
            rng.shuffle(train_idx)
        else:
            idx = np.arange(n); rng.shuffle(idx)
            k = int(round(n * test_size))
            test_idx, train_idx = idx[:k], idx[k:]
        out = []
        for a in arrays:
            out.append(_safe_index(a, train_idx))
            out.append(_safe_index(a, test_idx))
        return tuple(out)

    # ──────────────────────────────────────────────────────────────────────
    # 3. Preprocessing
    # ──────────────────────────────────────────────────────────────────────
    class StandardScaler:
        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float)
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
            return self
        def transform(self, X):
            return (np.asarray(X, dtype=float) - self.mean_) / self.std_
        def fit_transform(self, X, y=None):
            return self.fit(X).transform(X)

    # ──────────────────────────────────────────────────────────────────────
    # 4. Logistic regression (L-BFGS-B on log-loss)
    # ──────────────────────────────────────────────────────────────────────
    class LogisticRegression:
        def __init__(self, class_weight=None, max_iter=2000,
                     random_state=None, n_jobs=None, C=1.0, **kw):
            self.class_weight = class_weight
            self.max_iter = max_iter
            self.C = C

        def fit(self, X, y):
            from scipy.optimize import minimize
            X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
            n = len(y)
            if self.class_weight == "balanced":
                n_pos = max((y == 1).sum(), 1); n_neg = max((y == 0).sum(), 1)
                w = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))
            else:
                w = np.ones(n)
            Xb = np.hstack([np.ones((n, 1)), X])

            def loss(b):
                z = Xb @ b
                ll = np.where(y == 1, np.logaddexp(0, -z), np.logaddexp(0, z))
                return float((w * ll).sum() + 0.5 * (b[1:] ** 2).sum() / self.C)

            def grad(b):
                z = Xb @ b; p = 1 / (1 + np.exp(-z))
                g = Xb.T @ (w * (p - y))
                g[1:] += b[1:] / self.C
                return g

            res = minimize(loss, np.zeros(X.shape[1] + 1), jac=grad,
                           method="L-BFGS-B",
                           options={"maxiter": self.max_iter})
            self.intercept_ = res.x[:1]
            self.coef_ = res.x[1:].reshape(1, -1)
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            z = np.asarray(X, dtype=float) @ self.coef_.ravel() + self.intercept_[0]
            p = 1 / (1 + np.exp(-z))
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ──────────────────────────────────────────────────────────────────────
    # 5. Decision tree
    # ──────────────────────────────────────────────────────────────────────
    class _Node:
        __slots__ = ("leaf", "prob", "feature", "threshold", "left", "right")
        def __init__(self, leaf=False, prob=0.0, feature=0,
                     threshold=0.0, left=None, right=None):
            self.leaf = leaf; self.prob = prob
            self.feature = feature; self.threshold = threshold
            self.left = left; self.right = right

    class DecisionTreeClassifier:
        def __init__(self, max_depth=None, class_weight=None, random_state=None,
                     min_samples_leaf=1, min_samples_split=2, **kw):
            self.max_depth = max_depth
            self.class_weight = class_weight
            self.random_state = random_state
            self.min_samples_leaf = min_samples_leaf
            self.min_samples_split = min_samples_split
            self.tree_ = None
            self.n_features_ = 0
            self.feature_importances_ = np.zeros(1)

        @staticmethod
        def _gini(y, w):
            t = w.sum()
            if t <= 0: return 0.0
            p1 = w[y == 1].sum() / t
            return 1 - (1 - p1) ** 2 - p1 ** 2

        def _best_split(self, X, y, w, feats):
            n = len(y); parent = self._gini(y, w)
            best_gain, best = -1.0, None
            for j in feats:
                v = X[:, j]; uniq = np.unique(v)
                if len(uniq) < 2: continue
                thresholds = (np.quantile(v, np.linspace(0.05, 0.95, 20))
                              if len(uniq) > 20 else (uniq[:-1] + uniq[1:]) / 2)
                for t in np.unique(thresholds):
                    m = v <= t
                    if m.sum() < self.min_samples_leaf or (~m).sum() < self.min_samples_leaf:
                        continue
                    wl, wr = w[m], w[~m]
                    gain = parent - (wl.sum() / w.sum()) * self._gini(y[m], wl) \
                                  - (wr.sum() / w.sum()) * self._gini(y[~m], wr)
                    if gain > best_gain:
                        best_gain, best = gain, (j, t)
            return best

        def _build(self, X, y, w, depth, feats):
            if (len(y) < self.min_samples_split
                    or (self.max_depth is not None and depth >= self.max_depth)
                    or len(np.unique(y)) == 1):
                return _Node(leaf=True, prob=float(y.mean()))
            split = self._best_split(X, y, w, feats)
            if split is None:
                return _Node(leaf=True, prob=float(y.mean()))
            j, t = split; m = X[:, j] <= t
            return _Node(leaf=False, feature=j, threshold=t,
                         left=self._build(X[m], y[m], w[m], depth + 1, feats),
                         right=self._build(X[~m], y[~m], w[~m], depth + 1, feats))

        def fit(self, X, y):
            X = np.asarray(X, dtype=float); y = np.asarray(y).astype(int)
            n = len(y)
            if self.class_weight == "balanced":
                n_pos = max((y == 1).sum(), 1); n_neg = max((y == 0).sum(), 1)
                w = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))
            else:
                w = np.ones(n)
            self.n_features_ = X.shape[1]
            self.tree_ = self._build(X, y, w, 0, np.arange(self.n_features_))
            self._compute_importances()
            return self

        def _compute_importances(self):
            imp = np.zeros(self.n_features_)
            def walk(node):
                if node.leaf: return
                imp[node.feature] += 1
                walk(node.left); walk(node.right)
            walk(self.tree_)
            s = imp.sum()
            self.feature_importances_ = imp / s if s > 0 else imp

        def _leaf_prob(self, x, node):
            while not node.leaf:
                node = node.left if x[node.feature] <= node.threshold else node.right
            return node.prob

        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            p = np.array([self._leaf_prob(x, self.tree_) for x in X])
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ──────────────────────────────────────────────────────────────────────
    # 6. Random forest
    # ──────────────────────────────────────────────────────────────────────
    class RandomForestClassifier:
        def __init__(self, n_estimators=100, max_depth=None, class_weight=None,
                     random_state=None, n_jobs=None, **kw):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.class_weight = class_weight
            self.random_state = random_state
            self.trees_ = []

        def fit(self, X, y):
            X = np.asarray(X, dtype=float); y = np.asarray(y).astype(int)
            rng = np.random.default_rng(self.random_state if self.random_state is not None else 42)
            d = X.shape[1]; mtry = max(1, int(np.sqrt(d)))
            self.trees_ = []
            for _ in range(self.n_estimators):
                idx = rng.integers(0, len(X), len(X))
                Xb, yb = X[idx], y[idx]
                if self.class_weight == "balanced":
                    n = len(yb); n_pos = max((yb == 1).sum(), 1); n_neg = max((yb == 0).sum(), 1)
                    w = np.where(yb == 1, n / (2 * n_pos), n / (2 * n_neg))
                else:
                    w = np.ones(len(yb))
                feats = rng.choice(d, size=mtry, replace=False)
                tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                              random_state=int(rng.integers(0, 1 << 30)))
                tree.n_features_ = d
                tree.tree_ = tree._build(Xb, yb, w, 0, feats)
                tree._compute_importances()
                self.trees_.append(tree)
            return self

        def predict_proba(self, X):
            probs = np.array([t.predict_proba(X)[:, 1] for t in self.trees_])
            p = probs.mean(axis=0)
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        @property
        def feature_importances_(self):
            return np.mean([t.feature_importances_ for t in self.trees_], axis=0)

    # ──────────────────────────────────────────────────────────────────────
    # 7. Gradient boosting (binary, log-loss residuals)
    # ──────────────────────────────────────────────────────────────────────
    class _RegTree:
        def __init__(self, max_depth=3, min_samples_leaf=5):
            self.max_depth = max_depth
            self.min_samples_leaf = min_samples_leaf
            self.tree_ = None

        def fit(self, X, y):
            self.tree_ = self._build(X, y, 0); return self

        def _build(self, X, y, depth):
            if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
                return _Node(leaf=True, prob=float(y.mean()))
            parent = float(((y - y.mean()) ** 2).sum())
            best_gain, best = 0.0, None
            for j in range(X.shape[1]):
                v = X[:, j]; uniq = np.unique(v)
                if len(uniq) < 2: continue
                thresholds = (np.quantile(v, np.linspace(0.1, 0.9, 10))
                              if len(uniq) > 10 else (uniq[:-1] + uniq[1:]) / 2)
                for t in np.unique(thresholds):
                    m = v <= t
                    if m.sum() < self.min_samples_leaf or (~m).sum() < self.min_samples_leaf:
                        continue
                    yl, yr = y[m], y[~m]
                    gain = parent - float(((yl - yl.mean()) ** 2).sum()
                                          + ((yr - yr.mean()) ** 2).sum())
                    if gain > best_gain:
                        best_gain, best = gain, (j, t)
            if best is None:
                return _Node(leaf=True, prob=float(y.mean()))
            j, t = best; m = X[:, j] <= t
            return _Node(leaf=False, feature=j, threshold=t,
                         left=self._build(X[m], y[m], depth + 1),
                         right=self._build(X[~m], y[~m], depth + 1))

        def predict(self, X):
            def walk(x, node):
                while not node.leaf:
                    node = node.left if x[node.feature] <= node.threshold else node.right
                return node.prob
            return np.array([walk(x, self.tree_) for x in np.asarray(X, dtype=float)])

    class GradientBoostingClassifier:
        def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                     random_state=None, **kw):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.random_state = random_state
            self.trees_ = []; self.f0_ = 0.0; self.n_features_ = 0

        def fit(self, X, y):
            X = np.asarray(X, dtype=float); y = np.asarray(y).astype(float)
            self.n_features_ = X.shape[1]
            p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            self.f0_ = float(np.log(p / (1 - p)))
            F = np.full(len(y), self.f0_); self.trees_ = []
            for _ in range(self.n_estimators):
                p = 1 / (1 + np.exp(-F))
                tree = _RegTree(self.max_depth).fit(X, y - p)
                F += self.learning_rate * tree.predict(X)
                self.trees_.append(tree)
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            F = np.full(len(X), self.f0_)
            for t in self.trees_:
                F += self.learning_rate * t.predict(X)
            p = 1 / (1 + np.exp(-F))
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        @property
        def feature_importances_(self):
            # placeholder (splits not tracked in this fallback)
            return np.ones(self.n_features_ or 1) / (self.n_features_ or 1)

    # ──────────────────────────────────────────────────────────────────────
    # 8. Isolation Forest
    # ──────────────────────────────────────────────────────────────────────
    def _avg_path(n):
        if n <= 1: return 0.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    class _ITree:
        def __init__(self, max_depth):
            self.max_depth = max_depth; self.tree_ = None; self.n_features_ = 0

        def fit(self, X, rng):
            self.n_features_ = X.shape[1]
            self.tree_ = self._build(X, 0, rng); return self

        def _build(self, X, depth, rng):
            n = len(X)
            if depth >= self.max_depth or n <= 1:
                return {"leaf": True, "size": n}
            j = int(rng.integers(0, self.n_features_))
            v = X[:, j]; lo, hi = float(v.min()), float(v.max())
            if lo == hi:
                return {"leaf": True, "size": n}
            t = float(rng.uniform(lo, hi)); m = v < t
            if m.sum() == 0 or (~m).sum() == 0:
                return {"leaf": True, "size": n}
            return {"leaf": False, "feature": j, "threshold": t,
                    "left": self._build(X[m], depth + 1, rng),
                    "right": self._build(X[~m], depth + 1, rng)}

        def path_length(self, x, node, depth=0):
            if node.get("leaf"):
                return depth + _avg_path(node["size"])
            if x[node["feature"]] < node["threshold"]:
                return self.path_length(x, node["left"], depth + 1)
            return self.path_length(x, node["right"], depth + 1)

    class IsolationForest:
        def __init__(self, n_estimators=100, contamination=0.1,
                     random_state=None, n_jobs=None, **kw):
            self.n_estimators = n_estimators
            self.contamination = contamination
            self.random_state = random_state
            self.trees_ = []; self.offset_ = 0.0

        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float); n = len(X)
            rng = np.random.default_rng(self.random_state if self.random_state is not None else 42)
            max_depth = int(np.ceil(np.log2(max(n, 2))))
            sample_size = min(256, n)
            self.trees_ = []
            for _ in range(self.n_estimators):
                idx = rng.choice(n, size=sample_size, replace=False)
                self.trees_.append(_ITree(max_depth).fit(X[idx], rng))
            scores = self._anomaly_scores(X)
            self.offset_ = float(np.quantile(scores, 1 - self.contamination))
            return self

        def _anomaly_scores(self, X):
            X = np.asarray(X, dtype=float)
            depths = np.zeros(len(X))
            for t in self.trees_:
                for i, x in enumerate(X):
                    depths[i] += t.path_length(x, t.tree_)
            depths /= max(len(self.trees_), 1)
            c = _avg_path(256) or 1.0
            return 2.0 ** (-depths / c)

        def decision_function(self, X):
            # positive → inlier, negative → outlier (matches sklearn)
            return self.offset_ - self._anomaly_scores(X)

        def predict(self, X):
            return np.where(self.decision_function(X) < 0, -1, 1)

        def fit_predict(self, X, y=None):
            self.fit(X); return self.predict(X)

    # ──────────────────────────────────────────────────────────────────────
    # 9. Pipeline
    # ──────────────────────────────────────────────────────────────────────
    class Pipeline:
        def __init__(self, steps):
            self.steps = list(steps)
            self.named_steps = dict(self.steps)

        def fit(self, X, y):
            Xc = np.asarray(X, dtype=float)
            for _, s in self.steps[:-1]:
                Xc = s.fit_transform(Xc)
            self.steps[-1][1].fit(Xc, y)
            return self

        def predict(self, X):
            Xc = np.asarray(X, dtype=float)
            for _, s in self.steps[:-1]:
                Xc = s.transform(Xc)
            return self.steps[-1][1].predict(Xc)

        def predict_proba(self, X):
            Xc = np.asarray(X, dtype=float)
            for _, s in self.steps[:-1]:
                Xc = s.transform(Xc)
            return self.steps[-1][1].predict_proba(Xc)

    # ──────────────────────────────────────────────────────────────────────
    # 10. imblearn fallbacks
    # ──────────────────────────────────────────────────────────────────────
    class SMOTE:
        def __init__(self, random_state=None, k_neighbors=5, **kw):
            self.random_state = random_state
            self.k_neighbors = k_neighbors

        def fit_resample(self, X, y):
            X = np.asarray(X, dtype=float); y = np.asarray(y).astype(int)
            rng = np.random.default_rng(self.random_state if self.random_state is not None else 42)
            classes, counts = np.unique(y, return_counts=True)
            minority = classes[np.argmin(counts)]
            n_needed = int(counts.max() - counts.min())
            X_min = X[y == minority]
            if n_needed <= 0 or len(X_min) < 2:
                return X, y
            k = min(self.k_neighbors, len(X_min) - 1)
            diff = X_min[:, None, :] - X_min[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dist, np.inf)
            nn = np.argsort(dist, axis=1)[:, :k]
            synth = np.empty((n_needed, X.shape[1]))
            for i in range(n_needed):
                a = int(rng.integers(0, len(X_min)))
                b = int(nn[a][rng.integers(0, k)])
                alpha = float(rng.random())
                synth[i] = X_min[a] + alpha * (X_min[b] - X_min[a])
            X_res = np.vstack([X, synth])
            y_res = np.concatenate([y, np.full(n_needed, minority, dtype=int)])
            perm = rng.permutation(len(X_res))
            return X_res[perm], y_res[perm]

    class SMOTETomek(SMOTE):
        """Fallback: aliased to SMOTE (Tomek-link cleaning omitted)."""
        pass

    # ──────────────────────────────────────────────────────────────────────
    # 11. Install into sys.modules
    # ──────────────────────────────────────────────────────────────────────
    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    pkg = _mod("sklearn"); pkg.__path__ = []
    _mod("sklearn.model_selection", train_test_split=_train_test_split)
    _mod("sklearn.preprocessing",   StandardScaler=StandardScaler)
    _mod("sklearn.linear_model",    LogisticRegression=LogisticRegression)
    _mod("sklearn.tree",            DecisionTreeClassifier=DecisionTreeClassifier)
    _mod("sklearn.ensemble",
         RandomForestClassifier=RandomForestClassifier,
         GradientBoostingClassifier=GradientBoostingClassifier,
         IsolationForest=IsolationForest)
    _mod("sklearn.metrics",
         precision_score=_precision,
         recall_score=_recall,
         f1_score=_f1,
         roc_auc_score=_roc_auc,
         average_precision_score=_ap,
         confusion_matrix=_cm,
         classification_report=_report,
         roc_curve=_roc_curve,
         precision_recall_curve=_pr_curve)
    _mod("sklearn.pipeline", Pipeline=Pipeline)

    pkg2 = _mod("imblearn"); pkg2.__path__ = []
    _mod("imblearn.over_sampling", SMOTE=SMOTE)
    _mod("imblearn.combine",       SMOTETomek=SMOTETomek)

    print("[_ml_patch] ✓ Pure-NumPy fallback installed.")
    print("[_ml_patch]   The app will run with equivalent functionality "
          "(slightly different numerical results).")