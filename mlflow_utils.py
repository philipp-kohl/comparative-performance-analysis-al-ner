import pandas as pd

def pull_run_metrics_as_df(mlflow_client, run_id, metric_names=None):
    for_pd_collect = []
    for metric in metric_names:
        metric_history = mlflow_client.get_metric_history(run_id=run_id, key=metric)
        pd_convertible_metric_history = [
            {
                'metric_name': mm.key,
                'step': mm.step,
                'timestamp': mm.timestamp,
                'value': mm.value,
            }
            for mm in metric_history
        ]
        for_pd_collect += pd_convertible_metric_history

    metrics_df = pd.DataFrame.from_records(for_pd_collect)
    return metrics_df
