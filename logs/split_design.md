
# DataSEC Evaluation Split Design

## 1. Dataset

- Dataset: DataSEC V.4
- Total audio files: 4,292
- Main classes: 22
- Subclasses: 28
- Unique groups: 4,290

## 2. Split proportions

The proposed fixed allocation is:

| Partition | Target proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

A feasibility simulation produced:

| Partition | Files | Actual proportion |
|---|---:|---:|
| Training | 2,994 | 69.76% |
| Validation | 643 | 14.98% |
| Testing | 655 | 15.26% |

## 3. Grouping method

- Two confirmed exact-duplicate pairs were identified.
- Each duplicate pair is assigned to one group.
- Both files in each confirmed duplicate group must remain in the same partition.
- All other files are treated as individual groups because original source identifiers are unavailable.
- CLAP similarity is used only to identify candidate pairs for manual inspection. It is not treated as proof of shared recording provenance.

## 4. Stratification and reproducibility

- Groups are allocated separately within each main class.
- Random seed: 42.
- The feasibility simulation confirmed that all 22 classes appear in all three partitions.
- The final allocation will be saved as a reproducible file manifest.

## 5. Evaluation protocol

- Training data will be used for model fitting.
- Validation data will be used for model selection, hyperparameter tuning, calibration and selection of decision thresholds.
- Test data will remain untouched until the final evaluation.
- Test results will not be used to change models, prompts, thresholds or other experimental choices.

## 6. Limitations

The dataset does not provide original recording-source identifiers. Therefore, this procedure controls confirmed exact duplicates but cannot guarantee that recordings from the same original source are separated across partitions.

The evaluation must be described as exact-duplicate-controlled and class-stratified, not as fully source-independent.

## 7. Status

This document records the proposed split design. The feasibility simulation is complete, but the final split manifest has not yet been generated.
