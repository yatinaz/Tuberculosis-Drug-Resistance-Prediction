# Data

The raw genomic data used in this project is not included in this
repository due to file size constraints.

## Dataset Description

| File | Dimensions | Description |
|------|-----------|-------------|
| `X_trainData_1.csv` | 3,393 × 222 | Genomic feature matrix (training) |
| `Y_trainData_1.csv` | 3,393 × 10 | Drug resistance labels (training) |
| `X_testData_1.csv` | — × 222 | Test set features |
| `Y_testData_1_nolabels_*.csv` | — | Test submission templates (one per drug) |

## Features

- **222 binary genomic features**: SNPs and indels encoded as presence (1) / absence (0)
- Each row is one *M. tuberculosis* (MTB) isolate
- **Labels**: 1 = resistant, 0 = susceptible, -1 = not tested (treated as missing)

## Drugs Covered

| Drug | Abbreviation | Class |
|------|-------------|-------|
| Isoniazid | INH | First-line |
| Rifampicin | RIF | First-line |
| Ethambutol | EMB | First-line |
| Pyrazinamide | PZA | First-line |
| Streptomycin | STR | First-line |
| Ofloxacin | OFLX | Fluoroquinolone |
| Amikacin | AMK | Injectable |
| Kanamycin | KAN | Injectable |
| Capreomycin | CAP | Injectable |
| Moxifloxacin | MOXI | Fluoroquinolone |

## Source

IIT Kanpur Computational Genomics course dataset (2020)
Instructor: Prof. Hamim Zafar

## To Reproduce Results

Place the CSV files in this directory (`data/`) and run:

```bash
python tb_pipeline.py
python generate_visualisations.py
```
