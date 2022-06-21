# Pythie serving

## 2.3.1

## Change
* Sklearn wrapper: unpickle models using cloudpickle library to support models pickled with their module
* Upgrade sklearn version to 1.1.0
* Add black + isort to CI and pre-commit configuration.

## 2.3.0

## Change
* Add support of CSV table, served as models
* Refactor parsing + checks of PredictRequest content

## 2.2.0

### Change
* Allow client to request model with string features
* Add support of scikit-learn models

## 2.1.0

### Change
* Allow setting treelite's Predictor `nthread` attribute by setting `TREELITE_NTHREAD` environment variable. Defaults to `1`.

## 2.0.0

### Change
* Add support of [treelite](https://treelite.readthedocs.io/en/latest/) compiled models

### Breaking changes
* Minimal python version required is now `3.8`
* `base_path` field of `models.config` file should now point to the directory containing the model file and not the file itself
