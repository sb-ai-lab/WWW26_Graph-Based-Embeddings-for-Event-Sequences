* `scenario_run_all` runs data splitting before running scenarios. The result is stored in `embeddings_validation.work/folds` Info on data spliting is below

## Data splitting
```python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=10 +total_cpu_count=20 \
    +split_only=True
```

```
Splits data into n folds (train, validation, test),
saves them to files and creates folds.json file with paths to these files.
The saved files are dumped TargetFile objects that contain 
ids and target values (subset of the original data).
Use TargetFile.load(path) to load theese files.
TargetFile = embeddings_validation.file_reader.TargetFile
Main interface of TargetFile are pseudo properties:
* .ids_values
* .target_values
The Target (Luigi Target) of this task is folds.json file with folds information.
The split can be done in two ways (self.conf.validation_schema):
* VALID_TRAIN_TEST
* VALID_CROSS_VAL
For both split types the folds Dict has the same structure:
Keys represent fold number and values are dictionaries with keys:
- 'train': dictionary with keys 'path' and 'shape' representing path to the train data and its shape
- 'valid': dictionary with keys 'path' and 'shape' representing path to the validation data and its shape
- 'test': dictionary with keys 'path' and 'shape' representing path to the test data and its shape

Test data is optional. If it is not provided, the 'test' property is None.
If it's providded, regardless of the split type, the 'test' 
property is always the same (same path and shape for all folds).
If validation_schema is VALID_CROSS_VAL:
* Fold numbers are integers from 0 to self.conf['split']['cv_split_count'] - 1
* Each fold contains train and validation data randomly split from the train data
If validation_schema is VALID_TRAIN_TEST:
* Fold numbers are integers from 0 to self.conf['split']['n_iteration'] - 1
* Each fold is exactly the same (Uses given train, validation and test ids); 
    same path and shape for all folds
```