import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from ..utilities.vectorop import split_dataset, drop_lowNs
from ..utilities.highlevel import RandomClassifierMock


class SupervisedSuite:
    modelnames = ["baseline", "lda", "logistic", "svm",
                  "qda", "gnb", "forest", "boosting", "knn",
                  "rbf-svm", "poly2-svm", "poly3-svm", "mlp"]
    _models = [RandomClassifierMock, LDA, LogisticRegression, lambda: SVC(kernel="linear"),
               QDA, GaussianNB, RandomForestClassifier, GradientBoostingClassifier,
               KNeighborsClassifier, lambda: SVC(kernel="rbf"), lambda: SVC(kernel="poly", degree=2),
               lambda: SVC(kernel="poly", degree=3), lambda: MLPClassifier(solver="lbfgs")]
    models = dict(zip(modelnames, _models))
    modeltypes = dict(
        linear=["lda", "logistic", "svm"],
        quadratic=["qda", "gnb"],
        nonparametric=["forest", "boosting", "knn"],
        kernel=["rbf-svm", "poly2-svm", "poly3-svm"],
        ann=["mlp"]
    )

    def __init__(self, include_models=(), exclude_models=()):
        if isinstance(include_models, str):
            include_models = [include_models]
            if include_models in self.modeltypes:
                include_models = self.modeltypes[include_models]
            elif include_models in self.modelnames:
                include_models = [include_models]
            else:
                raise ValueError(f"Include not understood: {include_models}")
        if isinstance(exclude_models, str):
            if exclude_models in self.modeltypes:
                exclude_models = self.modeltypes[exclude_models]
            elif exclude_models in self.modelnames:
                exclude_models = [exclude_models]
            else:
                raise ValueError(f"Exclude not understood: {exclude_models}")
        self.included = include_models if include_models else self.modelnames
        self.excluded = exclude_models if exclude_models else []

    def run_experiments(self, df: pd.DataFrame, labels, features, repeats=100, outxlsx=None, testing_split=0.1):
        modelnames = [modelnm for modelnm in self.included if modelnm not in self.excluded]
        output = pd.DataFrame(index=modelnames, columns=labels)
        drop1 = df.dropna(subset=features)
        for lab in labels:
            drop2 = drop1.dropna(subset=[lab])
            X, Y = drop2[features].as_matrix(), drop2[lab].as_matrix().ravel()
            Y, X = drop_lowNs(10, Y, X)
            for i, modelname in enumerate(modelnames, start=1):
                acc = []
                for repeat in range(1, repeats+1):
                    print(f"\rEvaluating {modelname.upper()} -> {repeat/repeats:.0%}", end="")
                    lX, lY, tX, tY = split_dataset(X, Y, ratio=testing_split, shuff=True, normalize=True)
                    model = self.models[modelname]()
                    model.fit(lX, lY)
                    acc.append((model.predict(tX) == tY).mean())
                output.loc[modelname, lab] = np.mean(acc)
                print()
        if outxlsx is not None:
            output.to_excel(outxlsx)
        return output
