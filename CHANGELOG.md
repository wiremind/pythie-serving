# Pythie serving

## 2.1.0

### Change
* Allow setting treelite's Predictor `nthread` attribute by setting `TREELITE_NTHREAD` environment variable. Defaults to `1`.

## 2.0.0

### Change
* Add support of [treelite](https://treelite.readthedocs.io/en/latest/) compiled models
 
### Breaking changes
* Minimal python version required is now `3.8`
* `base_path` field of `models.config` file should now point to the directory containing the model file and not the file itself