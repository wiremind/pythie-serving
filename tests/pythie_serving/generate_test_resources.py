import csv
import json
import os
import pickle
import random
import string
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from tests.utils import CustomerEncoder

np.random.seed(123456)
RESOURCES_PATH = Path(os.path.dirname(os.path.abspath(__file__)), "resources")


def generate_test_resources(model_platform: str, resource_path: Path):
    """Generates models of each available model platform for testing purposes."""

    if not os.path.exists(f"{resource_path}/{model_platform}-model"):
        Path.mkdir(Path(f"{resource_path}/{model_platform}-model"), parents=True, exist_ok=True)

    nb_features = 10
    nb_samples = 100
    X_train = 100 * np.random.random((nb_samples, nb_features))
    y_train = 100 * np.random.random(nb_samples)

    feature_names = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 10))) for _ in range(nb_features)
    ]

    metadata = {"feature_names": feature_names}
    if model_platform == "sklearn":
        # include bytes features and use a pipeline for encoding
        str_feature = np.array(
            [
                "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 10))).encode()
                for _ in range(nb_samples)
            ]
        ).reshape(-1, 1)
        X_train = np.concatenate([100 * np.random.random((nb_samples, nb_features - 1)), str_feature], axis=1)
        model = Pipeline([("encoder", CustomerEncoder()), ("svr", SVR())])
        model.fit(X_train, y_train)

    elif model_platform == "lightgbm":
        model = lgbm.train(params={"verbose": -1}, train_set=lgbm.Dataset(X_train, y_train), feature_name=feature_names)

    elif model_platform == "xgboost":
        model = xgb.train(params={}, dtrain=xgb.DMatrix(X_train, y_train, feature_names=feature_names))

    elif model_platform == "treelite":
        import treelite

        model = lgbm.train(params={"verbose": -1}, train_set=lgbm.Dataset(X_train, y_train), feature_name=feature_names)
        treelite_model = treelite.Model.from_lightgbm(model)
        treelite_model.export_lib(
            toolchain="clang",
            libpath=f"{resource_path}/{model_platform}-model/{model_platform}-model.so",
        )

    elif model_platform == "table":
        with open(f"{resource_path}/{model_platform}-model/{model_platform}-model.csv", "w") as f:
            csvwriter = csv.writer(f, delimiter=",")
            csvwriter.writerow(feature_names + ["target"])
            for i, row in enumerate(X_train):
                csvwriter.writerow([int(x) for x in row] + [int(y_train[i])])
        metadata["target_name"] = "target"
        metadata["data_type"] = {**{feature_name: "int" for feature_name in feature_names}, "target": "int"}

    else:
        raise Exception(f"Unkown model_platform: {model_platform}")

    if model_platform not in {"treelite", "table"}:
        with open(f"{resource_path}/{model_platform}-model/{model_platform}-model.pickled", "wb") as f:
            f.write(pickle.dumps(model))

    with open(f"{resource_path}/{model_platform}-model/metadata.json", "w") as f:
        f.write(json.dumps(metadata))


if __name__ == "__main__":
    for _model_platform in ("sklearn", "lightgbm", "xgboost", "treelite", "table"):
        generate_test_resources(_model_platform, RESOURCES_PATH)
