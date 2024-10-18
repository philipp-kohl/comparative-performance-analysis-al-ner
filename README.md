# Comparative Performance Analysis of Active Learning Strategies for the Entity Recognition Task

This is the repository for the paper [Comparative Performance Analysis of Active Learning Strategies for the Entity Recognition Task](TODO: Add after publication).
Here you can find additional information and details for each experiment. 

The structure is as follows:
- **google_sheet**: We tracked all experiments in a collaborative Google Tables file. Here you can find a mapping of the experiment_id and run_id to a semantic name for the experiment. This can be helpful for working with the MLflow raw data. Here you see an example row of the data:

| experiment_name                        | corpus    | experiment_id | run_id                              | experiment_type | strategy  |
|----------------------------------------|-----------|---------------|-------------------------------------|-----------------|-----------|
| variance_test_conll_max_number_of_nes  | CoNLL2003 | 11            | 3a4f8e313b794b45b816e8fa095867c1    | robustness      | tag_count |

- **mlflow**: Based on the Google Sheet we downloaded all information from MLflow to provide them for further research. Here is an example for the mlflow_metrics.csv. The __run_id__ can be used for tracking down the experiment with the Google Sheet:

| metric_name    | step | timestamp       | value             | run_id                              | strategy                 | corpus  | experiment_type |
|----------------|------|-----------------|-------------------|-------------------------------------|--------------------------|---------|-----------------|
| test_f1_macro  | 55   | 1722648280077   | 0.0919434130191803 | bffcb7bb722746ddaba81850d09f4105    | representative_diversity | AURC-7  | performance     |

- **results**: The results folder provides a lot of csv files and figures, which were used for the paper, but you can find also additional information.

## Getting started

Prerequisite:
- [git](https://git-scm.com/)
- [poetry](https://python-poetry.org/docs/main/#installing-with-pipx)

Run main.py to recreate results or start your own evaluations based on our data. 

## Citation
If you find this code useful in your research, please cite:

```
@inproceedings{TODO,
  title = {Comparative Performance Analysis of Active Learning Strategies for the Entity Recognition Task},
}
```
