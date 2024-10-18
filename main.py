import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import wilcoxon, friedmanchisquare
from tqdm import tqdm

import mlflow
from mlflow import MlflowClient
from mlflow_utils import pull_run_metrics_as_df

pio.kaleido.scope.mathjax = False

artifact_paths = ["dev/bias_assessment/accuracy_per_label", "dev/bias_assessment/bias",
                  "dev/bias_assessment/bias_distr_diff", "dev/bias_assessment/bias_log_distr_diff",
                  "dev/bias_assessment/error_per_label",
                  "dev/data_distribution", "dev/confidence_assessment/ece"]
artifact_store = defaultdict(list)

strategy_group = {
    "randomizer": "baseline",
    "diversity": "exploration",
    "entropy": "exploitation",
    "fluctuation_history": "exploitation",
    "information_density": "hybrid",
    "k_means_bert": "exploration",
    "least_confidence": "exploitation",
    "margin_confidence": "exploitation",
    "representative_diversity": "exploration",
    "round_robin": "exploitation",
    "sequential_representation_lc": "hybrid",
    "tag_count": "exploitation",
    "tag_flip": "exploitation",
}

corpus_step_size = {
    "CoNLL2003": 500,
    "Medmentions": 100,
    "AURC-7": 375,
    "WNUT": 500,
    "SCIERC": 75,
    "JNLPBA": 375,
    "GermEval": 500,
}


def find_first_parent_run_with_seed_runs(experiment_id: str, parent_seed_runs: pd.DataFrame):
    for idx, parent_seed_run in parent_seed_runs.iterrows():
        parent_seed_run_id = parent_seed_run["run_id"]
        seed_runs = mlflow.search_runs(experiment_ids=[experiment_id],
                                       filter_string=f"tags.mlflow.parentRunId = '{parent_seed_run_id}'")

        if len(seed_runs) > 0:
            return seed_runs


def performance_metrics(experiment_id: str, run_id: str):
    client = MlflowClient()

    child_runs = mlflow.search_runs(experiment_ids=[experiment_id],
                                    filter_string=f"tags.mlflow.parentRunId = '{run_id}'")
    parent_seed_runs = child_runs[child_runs['tags.mlflow.runName'] == 'seed_runs']
    seed_runs = find_first_parent_run_with_seed_runs(experiment_id, parent_seed_runs)

    metrics = []
    for idx, seed_run in seed_runs.iterrows():
        seed_run_id = seed_run["run_id"]
        metrics_for_run = pull_run_metrics_as_df(client, seed_run_id, ["test_f1_macro", "proposing_duration",
                                                                       "iteration_duration", "prediction_duration",
                                                                       "training_duration"])
        metrics_for_run["run_id"] = seed_run_id
        metrics.append(metrics_for_run)

        # Data bias and confidence will be evaluated together. Just check for one
        train_runs = mlflow.search_runs(experiment_ids=[experiment_id],
                                        filter_string=f"tags.mlflow.parentRunId = '{seed_run_id}' and "
                                                      f"tags.assess_data_bias = 'True' and "
                                                      f"status = 'FINISHED'")
        # Tags
        for train_run_idx, train_run in train_runs.iterrows():
            for artifact_path in artifact_paths:
                artifact_store[artifact_path].append(
                    download_artifact(artifact_path, seed_run, train_run, run_id))

    metrics_df = pd.concat(metrics)

    return metrics_df


def download_artifact(path, seed_run, train_run, experiment_run_id):
    train_run_id = train_run["run_id"]
    local_dir = mlflow.artifacts.download_artifacts(run_id=train_run_id, artifact_path=path)
    df = pd.read_csv(Path(local_dir) / "data.csv")
    df["run_id"] = train_run_id
    df["parent_run_id"] = seed_run["run_id"]
    df["experiment_run_id"] = experiment_run_id
    df["experiment_id"] = seed_run["experiment_id"]
    df["corpus"] = seed_run["params.data.data_dir"]
    df["start_time"] = train_run["start_time"]
    df["strategy"] = seed_run["params.teacher.strategy"]
    df["type"] = path.split("/")[-1]

    return df


def process_group(group):
    dfs = []
    corpus = group["corpus"].iloc[0]
    experiment_type = group["experiment_type"].iloc[0]

    # Initialize ThreadPoolExecutor with a number of workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks for concurrent execution
        futures = [
            executor.submit(process_single_experiment, experiment, corpus, experiment_type)
            for idx, experiment in group.iterrows()
        ]

        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if not result.empty:
                    dfs.append(result)
            except Exception as e:
                print(e)

    # Concatenate all dataframes
    return pd.concat(dfs) if dfs else pd.DataFrame()


def process_single_experiment(experiment, corpus, experiment_type):
    experiment_id = experiment["experiment_id"]
    run_id = experiment["run_id"]
    dfs = []

    if experiment_id and run_id:
        try:
            df = performance_metrics(experiment_id, run_id)
            df["strategy"] = experiment["strategy"]
            df["corpus"] = corpus
            df["experiment_type"] = experiment_type
            dfs.append(df)
        except Exception as e:
            print(e)

    return pd.concat(dfs) if dfs else pd.DataFrame()


def prepare_for_display(value: str):
    formatted_str = value.replace('_', ' ')
    formatted_str = formatted_str.title()
    return formatted_str


def download_results(path: str):
    df = pd.read_csv(path, dtype={"experiment_id": str})
    df = df.where(pd.notnull(df), None)
    grouped = df.groupby(["corpus", "experiment_type"])

    all_groups = []
    for name, group in grouped:
        print(f"Process Group: {name}")
        try:
            single_group = process_group(group)
            all_groups.append(single_group)
        except Exception as e:
            print(e)

    all_df: pd.DataFrame = pd.concat(all_groups)

    return all_df


def tuple_to_string(values: Tuple):
    return '-'.join(word.lower().replace(' ', '_') for word in values)


def plot_group(name_df_tuple):
    name, df = name_df_tuple

    # Filter for poster graph
    # df = df[(df["strategy"] == "fluctuation_history") | (df["strategy"] == "k_means_bert") | (df["strategy"] == "margin_confidence") | (df["strategy"] == "randomizer") | (df["strategy"] == "tag_count")]
    print(f"Processing group: {name}")

    dash_styles = {}
    for key in strategy_group:
        dash_styles[key] = "solid"
    dash_styles["randomizer"] = "dash"

    color_palette = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-Green
        '#17becf',  # Teal
        '#ff9896',  # Light Red
        '#98df8a',  # Light Green
        '#c5b0d5'  # Light Purple
    ]
    fig = px.line(df, x='step', y='mean', color='strategy',
                  labels={
                      'step': 'Number of Data Points',
                      'mean': "Mean F1 Macro",
                      'strategy': 'Strategy'
                  },
                  #title='Mean by Step for Different Strategies',
                  title=None,
                  color_discrete_sequence=color_palette,
                  line_dash='strategy',  # This applies the dash styles based on the strategy column
                  line_dash_map=dash_styles  #
                  )

    fig.update_layout(
        xaxis=dict(range=[0, 10000]),
        legend=dict(
            x=0.6,  # Adjust x position (0 is far left, 1 is far right)
            y=0.01,  # Adjust y position (0 is bottom, 1 is top)
            bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: Add a semi-transparent background to the legend
            bordercolor='Black',  # Optional: Add a border color to the legend
            borderwidth=0  # Optional: Add border width
        )
        # Limit x-axis from 0 to 10k
    )

    group_name = tuple_to_string(name)
    fig.write_image(f"./results/{group_name}-mean.png", scale=2)
    fig.write_image(f"./results/{group_name}-mean.pdf", scale=2)
    fig.write_html(f"./results/{group_name}-mean.html")


    fig = px.line(df, x='step', y='std', color='strategy',
                  labels={
                      'step': 'Number of Data Points',
                      'std': prepare_for_display("metric"),
                      'strategy': 'Strategy'
                  },
                  title='Mean by Step for Different Strategies',
                  color_discrete_sequence=color_palette
                  )

    fig.write_image(f"./results/{group_name}-std.png", scale=2)
    fig.write_html(f"./results/{group_name}-std.html")


def plot_per_corpus(all_df: pd.DataFrame, metrics_to_plot: List[str] = None):
    if metrics_to_plot is None:
        metrics_to_plot = ['test_f1_macro']

    mean_and_std_df = (all_df.groupby(['metric_name', 'step', 'corpus', 'experiment_type', 'strategy'])['value']
                       .agg(['mean', 'std']).reset_index())
    filtered_df = mean_and_std_df[mean_and_std_df['metric_name'].isin(metrics_to_plot)]

    grouped = filtered_df.groupby(['metric_name', 'corpus', 'experiment_type'])
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map the process_group function to all the grouped data
        pool.map(plot_group, grouped)


def compute_auc(all_df: pd.DataFrame):
    mean_and_std_df = (all_df.groupby(['metric_name', 'step', 'corpus', 'experiment_type', 'strategy'])['value']
                       .agg(['mean', 'std']).reset_index())
    filtered_df = mean_and_std_df[mean_and_std_df['metric_name'].isin(['test_f1_macro'])]
    filtered_df = filtered_df.sort_values(['corpus', 'strategy', 'step'])
    grouped = filtered_df.groupby(['metric_name', 'corpus', 'experiment_type', 'strategy'])

    def calculate_auc(group):
        return np.trapz(group['mean'], group['step'])

    # Group by 'corpus_id' and apply the AUC calculation
    auc_by_corpus = grouped.apply(calculate_auc)
    auc_df = auc_by_corpus.reset_index(name='auc')
    sorted_auc_df = auc_df.sort_values(['metric_name', 'corpus', 'auc'], ascending=[True, True, False])

    return sorted_auc_df


def compute_auc_diff(auc):
    randomizer_df = auc[auc['strategy'] == 'randomizer'].copy()
    randomizer_df = randomizer_df[['metric_name', 'corpus', 'experiment_type', 'auc']]
    randomizer_df.rename(columns={'auc': 'randomizer_auc'}, inplace=True)
    # Merge the DataFrame with the randomizer AUCs back to the original DataFrame
    merged_df = pd.merge(auc, randomizer_df, on=['metric_name', 'corpus', 'experiment_type'])
    # Calculate the AUC difference
    merged_df['auc_diff'] = merged_df['auc'] - merged_df['randomizer_auc']
    sorted_auc_diff = merged_df.sort_values(['metric_name', 'corpus', 'auc_diff'], ascending=[True, True, False])

    return sorted_auc_diff


def compute_cum_performance_gain(df):
    filtered_df = df[df['metric_name'].isin(['test_f1_macro'])]

    randomizer_df = filtered_df[filtered_df['strategy'] == 'randomizer'].copy()
    randomizer_df = randomizer_df[['metric_name', 'corpus', 'experiment_type', 'step', 'value']]
    randomizer_df.rename(columns={'value': 'randomizer_value'}, inplace=True)

    # Merge the randomizer values back to the original DataFrame
    merged_df = pd.merge(filtered_df, randomizer_df, on=['metric_name', 'corpus', 'experiment_type', 'step'])

    # Calculate the performance difference at each step
    merged_df['performance_diff'] = merged_df['value'] - merged_df['randomizer_value']

    # Filter out the randomizer strategy since we are not interested in calculating CPG for it
    merged_df = merged_df[merged_df['strategy'] != 'randomizer']

    # Calculate the cumulative performance gain for each strategy
    merged_df['cumulative_performance_gain'] = \
        merged_df.groupby(['metric_name', 'strategy', 'corpus', 'experiment_type'])[
            'performance_diff'].cumsum()

    # Calculate the total CPG for each strategy (sum of the cumulative gains)
    cpg = merged_df.groupby(['metric_name', 'strategy', 'corpus', 'experiment_type'])[
        'cumulative_performance_gain'].last().reset_index()

    # Rename column for clarity
    cpg.rename(columns={'cumulative_performance_gain': 'total_cpg'}, inplace=True)

    return cpg


def compute_robustness(df):
    mean_and_std_df = (df.groupby(['metric_name', 'step', 'corpus', 'experiment_type', 'strategy'])['value']
                       .agg(['mean', 'std']).reset_index())
    condition = (mean_and_std_df['metric_name'].isin(['test_f1_macro'])) & (
            mean_and_std_df["experiment_type"] == "robustness")
    filtered_df = mean_and_std_df[condition]
    robustness = filtered_df.groupby("strategy")["std"].mean()
    return robustness.sort_values(ascending=True)


def compute_wilcoxon_test(df):
    # Filter the DataFrame for 'test_f1_macro' metric
    df = df[df['metric_name'] == 'test_f1_macro']
    # Separate the randomizer performance
    randomizer_df = df[df['strategy'] == 'randomizer'].copy()
    randomizer_df = randomizer_df[['step', 'value', 'corpus']]
    randomizer_df.rename(columns={'value': 'randomizer_value'}, inplace=True)
    # Merge the randomizer values back to the original DataFrame
    merged_df = pd.merge(df, randomizer_df, on=['step', 'corpus'])
    # Filter out the randomizer strategy since we are comparing others against it
    filtered_df = merged_df[merged_df['strategy'] != 'randomizer']
    # Perform Wilcoxon signed-rank test for each strategy against the randomizer
    results = {}
    for strategy in filtered_df['strategy'].unique():
        strategy_values = filtered_df[filtered_df['strategy'] == strategy]['value']
        randomizer_values = filtered_df[filtered_df['strategy'] == strategy]['randomizer_value']

        # Ensure the data pairs are correctly aligned
        if len(strategy_values) == len(randomizer_values):
            stat, p_value = wilcoxon(strategy_values, randomizer_values, alternative="greater")
            results[strategy] = {'statistic': stat, 'p_value': p_value}
        else:
            print("Not matching dimensions")
    # Convert the results to a DataFrame for easier viewing
    results_df = pd.DataFrame.from_dict(results, orient='index')
    return results_df.sort_values(by=["p_value"], ascending=[True])


def compute_wilcoxon_test_per_domain(df):
    # Filter the DataFrame for 'test_f1_macro' metric
    df = df[df['metric_name'] == 'test_f1_macro']
    # Separate the randomizer performance
    randomizer_df = df[df['strategy'] == 'randomizer'].copy()
    randomizer_df = randomizer_df[['step', 'value', 'corpus']]
    randomizer_df.rename(columns={'value': 'randomizer_value'}, inplace=True)
    # Merge the randomizer values back to the original DataFrame
    merged_df = pd.merge(df, randomizer_df, on=['step', 'corpus'])
    # Filter out the randomizer strategy since we are comparing others against it
    filtered_df = merged_df[merged_df['strategy'] != 'randomizer']
    # Perform Wilcoxon signed-rank test for each strategy against the randomizer
    strategies = []
    corpora = []
    p_values = []
    for strategy in filtered_df['strategy'].unique():
        for domain in filtered_df['corpus'].unique():
            domain_df = filtered_df[filtered_df["corpus"] == domain]
            strategy_values = domain_df[domain_df['strategy'] == strategy]['value']
            randomizer_values = domain_df[domain_df['strategy'] == strategy]['randomizer_value']

            # Ensure the data pairs are correctly aligned
            if len(strategy_values) == len(randomizer_values):
                stat, p_value = wilcoxon(strategy_values, randomizer_values, alternative="greater")
                strategies.append(strategy)
                corpora.append(domain)
                p_values.append(p_value)
            else:
                print("Not matching dimensions")
    # Convert the results to a DataFrame for easier viewing
    results = {
        'strategy': strategies,
        'corpus': corpora,
        'p_value': p_values,
    }
    results_df = pd.DataFrame.from_dict(results)
    results_df["stat_significant"] = results_df["p_value"] < 0.05
    return results_df.sort_values(by=["corpus", "stat_significant", "p_value"], ascending=[True, False, True])


def compute_friedmanchisquare_test(df):
    # Filter the DataFrame for 'test_f1_macro' metric
    df = df[df['metric_name'] == 'test_f1_macro']
    df = df.sort_values(by=["corpus", "step"], ascending=[True, True])
    # Separate the randomizer performance

    strategy_values_all = []
    for strategy in df['strategy'].unique():
        strategy_values = np.array(df[df['strategy'] == strategy]['value'])
        strategy_values_all.append(strategy_values)

    stat, p_value = friedmanchisquare(*strategy_values_all)

    return p_value


def check_on_normality(df):
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.stats import shapiro

    # Overall Normality Check
    filtered_df = df[(df["experiment_type"] == "performance") & (df["metric_name"] == "test_f1_macro")]

    stat, p_value = shapiro(filtered_df["value"])
    print(f'Overall Shapiro-Wilk Test p-value: {p_value}')

    # Q-Q plot and Histogram for Overall
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    stats.probplot(filtered_df["value"], dist="norm", plot=plt)
    plt.title("Q-Q Plot - Overall")

    plt.subplot(1, 2, 2)
    plt.hist(filtered_df["value"], bins=20, edgecolor='k')
    plt.title("Histogram - Overall")
    plt.show()

    # Normality Check for Each Strategy and Corpus combination
    for strategy in filtered_df["strategy"].unique():
        for corpus in filtered_df["corpus"].unique():
            try:
                f1_scores_strategy = \
                    filtered_df[(filtered_df["strategy"] == strategy) & (filtered_df["corpus"] == corpus)]["value"]
                stat, p_value = shapiro(f1_scores_strategy)
                print(f'Strategy: {strategy}, Corpus: {corpus}, Shapiro-Wilk Test p-value: {p_value}')

                # Q-Q plot and Histogram for each Strategy
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                stats.probplot(f1_scores_strategy, dist="norm", plot=plt)
                plt.title(f"Q-Q Plot - Strategy: {strategy} - Corpus: {corpus}")

                plt.subplot(1, 2, 2)
                plt.hist(f1_scores_strategy, bins=20, edgecolor='k')
                plt.title(f"Histogram - Strategy: {strategy} - Corpus: {corpus}")
                plt.show()
            except Exception as e:
                print(e)


def compute_proposing_duration(df: pd.DataFrame):
    proposing_duration = df[df["metric_name"] == "proposing_duration"]
    proposing_duration["normalized_duration"] = proposing_duration["value"] / proposing_duration["corpus"].map(corpus_step_size)
    fig = px.box(proposing_duration, x="strategy", y="normalized_duration", title=None,
                 labels={
                     'normalized_duration': "Duration (seconds)",
                     'strategy': "Strategies",
                 },
                 )
    fig.update_layout(xaxis_title='',)
    fig.write_image(f"./results/proposing_duration_overall.pdf", scale=2)
    fig.write_html(f"./results/proposing_duration_overall.html")

    return proposing_duration.groupby(["strategy"])["value"].agg(["mean", "min", "max", "median"])


def compute_bias(df_bias: pd.DataFrame):
    means = df_bias.groupby(["strategy", "corpus"])["Bias"].mean().reset_index()
    fig = px.bar(means, x='strategy', y='Bias', color='corpus', barmode='stack',
                 title="Mean Bias by Strategy and Corpus")
    fig.write_image("./results/bias_per_corpus.pdf", scale=2)
    fig.show()

    means = df_bias.groupby(["strategy"])["Bias"].mean().reset_index()
    fig = px.bar(means, x='strategy', y='Bias',
                 title=None)
    fig.update_layout(xaxis_title='', )
    fig.write_image("./results/bias_per_strategy.pdf", scale=2)
    fig.show()

    return means


def compute_ece(df_ece: pd.DataFrame):
    eces = []
    for name, group in df_ece.groupby(["strategy", "corpus", "run_id"]):
        bin_accuracy = group["bin_accuracy"]
        avg_confidence = group["avg_confidence"]
        bin_factor = group["bin_count"] / group["bin_count"].sum()

        differences = bin_accuracy - avg_confidence
        ece = np.sum(bin_factor * differences)
        eces.append(ece)
        print(f"{name}: {ece}")

    print(f"Overall: {np.mean(eces)}")


def compute_robustness_suite(robustness_suite_df: pd.DataFrame):
    def progressive_std(group):
        stds = []
        for i in range(2, len(group) + 1):
            stds.append(group.iloc[:i].std())
        return pd.Series(stds)

    filtered_df = robustness_suite_df[robustness_suite_df["metric_name"] == "test_f1_macro"]
    stds = filtered_df.groupby(["corpus", "step"])["value"].apply(progressive_std).reset_index()
    stds.columns = ['corpus', 'step', 'number_seed_runs', 'progressive_std']
    stds["number_seed_runs"] = stds["number_seed_runs"] + 1

    def normalize_step(df):
        df['normalized_step'] = pd.factorize(df['step'], sort=True)[0]
        return df

    # Apply the function to each group by 'corpus'
    stds = stds.groupby('corpus').apply(normalize_step).reset_index(drop=True)

    stds_without_random_init = stds[stds["normalized_step"] > 0]

    fig = px.line(stds_without_random_init,
                  x='number_seed_runs',
                  y='progressive_std',
                  color='corpus',
                  facet_col='normalized_step',
                  facet_col_wrap=3,  # Adjust the number of columns for subplots
                  title="Progressive Standard Deviation by Step and Corpus")

    fig.write_html("./results/stds.html")

    color_palette = px.colors.qualitative.Bold  # Alternative: 'Bold, 'Plotly', 'Safe', 'Dark24'
    fig = px.line(stds_without_random_init[stds_without_random_init["normalized_step"] == 1],
                  x='number_seed_runs',
                  y='progressive_std',
                  color='corpus',
                  title=None,
                  color_discrete_sequence=color_palette)
    fig.update_layout(
        xaxis_title="Number of Seed Runs",  # Change X axis label
        yaxis_title="Average Standard Deviation"  # Change Y axis label
    )

    fig.write_html("./results/stds_al_cycle_1.html")
    fig.write_image("./results/stds_al_cycle_1.pdf")

    threshold = 0.005

    def check_progressive_std(row, df, threshold):
        current_run = row['number_seed_runs']
        current_std = row['progressive_std']
        corpus = row['corpus']
        step = row['step']

        # Get all subsequent rows for the same corpus and step
        subsequent_rows = df[(df['number_seed_runs'] > current_run) &
                             (df['corpus'] == corpus) &
                             (df['step'] == step)]

        # Check if any subsequent rows have a change in progressive_std below the threshold
        if not subsequent_rows.empty:
            diffs = abs(subsequent_rows['progressive_std'] - current_std)

            # Check if the maximum difference is below the threshold
            if diffs.max() < threshold:
                return True

        return False  # Last entry

    # Apply the function to each row
    stds_without_random_init['change_below_threshold'] = stds_without_random_init.apply(
        lambda row: check_progressive_std(row, stds_without_random_init, threshold), axis=1)
    result = stds_without_random_init[stds_without_random_init['change_below_threshold'] == True].groupby(
        ['corpus', 'normalized_step']).first().reset_index()
    result = result[['corpus', 'normalized_step', 'number_seed_runs']]
    result.columns = ['corpus', 'normalized_step', 'first_seed_run_with_threshold_met']

    result['normalized_step'] = result['normalized_step'].astype(str)
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green (or any 3 colors you prefer)

    # Create a mapping of normalized_step to custom colors
    color_mapping = {
        "1": custom_colors[0],
        "2": custom_colors[1],
        "3": custom_colors[2]
    }

    fig = px.scatter(
        result,
        x='corpus',
        y='first_seed_run_with_threshold_met',
        color='normalized_step',
        title="First Seed Run with Threshold Met per Corpus and Normalized Step",
        labels={
            'corpus': 'Corpus',
            'first_seed_run_with_threshold_met': 'First Seed Run Met',
            'normalized_step': 'Normalized Step'
        },
        color_discrete_sequence=custom_colors
    )
    max_values = result.groupby('normalized_step')['first_seed_run_with_threshold_met'].max()

    # TODO if data points have the same first eed run with threshold met, visualize side by side?
    # Add horizontal lines at the maximum values for each normalized_step
    for step, max_value in max_values.items():
        fig.add_shape(
            type='line',
            x0=-0.5, x1=len(result['corpus'].unique()) - 0.5,  # Span across the entire x-axis
            y0=max_value, y1=max_value,
            line=dict(color=color_mapping[step], dash='dash'),
            xref='x', yref='y'
        )
    # Customize markers
    fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))

    # Display the plot
    fig.write_html("./results/std_threshold.html")


def create_heatmap_for_best_strategies_per_domain(corpus_auc, domain_wilcoxon_df, dst: str):
    heatmap_data = pd.merge(domain_wilcoxon_df, corpus_auc, on=["strategy", "corpus"])
    heatmap_data["AUC"] = heatmap_data["auc_diff"] * heatmap_data["stat_significant"] * (
            heatmap_data["auc_diff"] > 0)
    heatmap_data["strategy_group"] = heatmap_data["strategy"].map(strategy_group)

    # Sort by strategy_group and strategy
    strategy_group_order = {'exploitation': 1, 'exploration': 2, 'hybrid': 3}
    heatmap_data["group_order"] = heatmap_data["strategy_group"].map(strategy_group_order)

    heatmap_data.loc[heatmap_data["strategy"] == "sequential_representation_lc", "strategy"] = "seq_rep_lc"
    heatmap_data.loc[heatmap_data["strategy"] == "representative_diversity", "strategy"] = "rep_diversity"
    heatmap_data = heatmap_data.sort_values(by=["group_order", "strategy"])

    # Pivot the data
    heatmap_data_pivot = heatmap_data.pivot(index='corpus', columns=['strategy_group', 'strategy'], values='AUC')
    heatmap_data_pivot.to_csv("./results/heatmap.csv")

    # Create the multi-category x-axis labels
    x_labels = [list(heatmap_data_pivot.columns.get_level_values(0)),
                list(heatmap_data_pivot.columns.get_level_values(1))]

    # Create the heatmap using go.Heatmap for better control over axis labels
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data_pivot.values,
        x=[x_labels[0], x_labels[1]],  # Multi-category axis labels
        y=heatmap_data_pivot.index,
        colorscale=[[0, '#ffffff'], [1e-6, '#e4ebf2'], [1, '#4e95d9']],
        colorbar=dict(title="AUC"),
        showscale=False,
    ))

    # Update layout for multi-category axis
    fig.update_layout(
        title=None,
        xaxis=dict(title=None,
                   tickangle=-80,
                   tickfont={'size': 18}),
        yaxis=dict(title=None,
                   tickfont={'size': 18})
    )

    # Show the plot
    fig.write_html(dst)
    fig.write_image(dst + ".pdf", width=1200, height=700, scale=2)


def main():
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

    data_path = Path("./mlflow/mlflow_metrics.csv")
    if data_path.exists():
        print(f"Use stored file: {data_path}")
        df = pd.read_csv(data_path)
        df_accuracy_per_label = pd.read_csv(f"./mlflow/accuracy_per_label.csv")
        df_bias = pd.read_csv(f"./mlflow/bias.csv")
        df_bias_distr_diff = pd.read_csv(f"./mlflow/bias_distr_diff.csv")
        df_bias_log_distr_diff = pd.read_csv(f"./mlflow/bias_log_distr_diff.csv")
        df_error_per_label = pd.read_csv(f"./mlflow/error_per_label.csv")
        df_data_distribution = pd.read_csv(f"./mlflow/data_distribution.csv")
        df_ece = pd.read_csv(f"./mlflow/ece.csv")
    else:
        print(f"No file found ({data_path}). Download results and store them.")
        df = download_results("./google_sheet/experiments_mlflow.csv")
        df.to_csv(data_path)

        df_accuracy_per_label = pd.concat(artifact_store["dev/bias_assessment/accuracy_per_label"])
        df_accuracy_per_label.to_csv(f"./mlflow/accuracy_per_label.csv")

        df_bias = pd.concat(artifact_store["dev/bias_assessment/bias"])
        df_bias.to_csv(f"./mlflow/bias.csv")

        df_bias_distr_diff = pd.concat(artifact_store["dev/bias_assessment/bias_distr_diff"])
        df_bias_distr_diff.to_csv(f"./mlflow/bias_distr_diff.csv")

        df_bias_log_distr_diff = pd.concat(artifact_store["dev/bias_assessment/bias_log_distr_diff"])
        df_bias_log_distr_diff.to_csv(f"./mlflow/bias_log_distr_diff.csv")

        df_error_per_label = pd.concat(artifact_store["dev/bias_assessment/error_per_label"])
        df_error_per_label.to_csv(f"./mlflow/error_per_label.csv")

        df_data_distribution = pd.concat(artifact_store["dev/data_distribution"])
        df_data_distribution.to_csv(f"./mlflow/data_distribution.csv")

        df_ece = pd.concat(artifact_store["dev/confidence_assessment/ece"])
        df_ece.to_csv(f"./mlflow/ece.csv")

    performance_df = df[df["experiment_type"] == "performance"]
    robustness_df = df[df["experiment_type"] == "robustness"]
    robustness_suite_df = df[df["experiment_type"] == "robustness_suite"]

    # check_on_normality(df)
    # p_value = compute_friedmanchisquare_test(performance_df)
    # print(f"Fiedman Chisquare test p value: {p_value}")

    plot_per_corpus(df, metrics_to_plot=["test_f1_macro"])
    #
    # compute_robustness_suite(robustness_suite_df)
    #
    auc = compute_auc(performance_df)
    auc.to_csv("./results/auc.csv")

    auc_diff = compute_auc_diff(auc)
    auc_diff.to_csv("./results/auc_diff.csv")

    overall_auc = auc_diff.groupby(["metric_name", "strategy"])["auc_diff"].mean()
    overall_auc.to_csv("./results/auc_overall.csv")

    corpus_auc = auc_diff.groupby(["metric_name", "strategy", "corpus"])["auc_diff"].mean().reset_index()
    corpus_auc = corpus_auc.sort_values(by=["corpus", "auc_diff"], ascending=[True, False])
    corpus_auc.to_csv("./results/corpus_overall.csv")

    gain = compute_cum_performance_gain(performance_df)
    overall_gain = gain.groupby(["metric_name", "strategy"])["total_cpg"].mean()
    overall_gain.to_csv("./results/gain_overall.csv")

    robustness = compute_robustness(robustness_df).reset_index()
    robustness.to_csv("./results/robustness.csv")
    fig = px.bar(robustness, x="strategy", y="std", title=None,
                 labels={"std": "Standard Deviation"}, text_auto=False)
    fig.write_html("./results/robustness.html")
    fig.write_image("./results/robustness.pdf")

    overall_wilcoxon_df = compute_wilcoxon_test(performance_df)
    overall_wilcoxon_df.to_csv("./results/overall_wilcoxon.csv")
    #
    domain_wilcoxon_df = compute_wilcoxon_test_per_domain(performance_df)
    domain_wilcoxon_df.to_csv("./results/domain_wilcoxon.csv")

    create_heatmap_for_best_strategies_per_domain(corpus_auc, domain_wilcoxon_df,
                                                  "./results/domain_wilcoxon_heatmap.html")

    mean_proposing_duration = compute_proposing_duration(performance_df)
    mean_proposing_duration.to_csv("./results/proposing_duration_overall.csv")


    df_bias["bias_log_distr_diff"] = df_bias_log_distr_diff["Bias"]
    df_bias["bias_distr_diff"] = df_bias_log_distr_diff["Bias"]
    bla = compute_bias(df_bias)

    grouped = df_bias.groupby(["experiment_id", "experiment_run_id", "Label"])

    for name, group in grouped:
        group = group.sort_values("start_time")

    def normalize_step(df):
        df['normalized_step'] = pd.factorize(df['step'], sort=True)[0]
        return df

    # Apply the function to each group by 'corpus'
    norm_steps = performance_df.groupby('corpus').apply(normalize_step).reset_index(drop=True)
    norm_steps = norm_steps[norm_steps["metric_name"] == "test_f1_macro"]
    compute_ece(df_ece)

    def process_group(group):
        step_size = corpus_step_size[group.name]

        # Filter for the randomizer strategy within the group
        randomizer_df = group[group['strategy'] == 'randomizer']

        # Get the best F1 score for the randomizer in this corpus
        best_randomizer = randomizer_df.loc[randomizer_df['value'].idxmax()]
        best_randomizer_value = best_randomizer['value']
        best_randomizer_step = best_randomizer['normalized_step']

        # Filter out the randomizer strategy
        non_randomizer_df = group[group['strategy'] != 'randomizer']

        # Find the first normalized_step for each strategy where value >= best_randomizer_value
        def first_matching_step(subgroup):
            filtered_subgroup = subgroup[subgroup['value'] >= best_randomizer_value]
            if not filtered_subgroup.empty:
                first_match = filtered_subgroup.loc[filtered_subgroup['normalized_step'].idxmin()]
                annotated_difference = (best_randomizer_step - first_match['normalized_step']) * step_size
                annotated_difference_ratio = (best_randomizer_step - first_match['normalized_step']) / best_randomizer_step
                first_match['annotated_difference'] = annotated_difference
                first_match['annotated_difference_ratio'] = annotated_difference_ratio
                return first_match
            return None

        # Apply to each strategy within the corpus
        result = non_randomizer_df.groupby('strategy').apply(first_matching_step).dropna()

        return result

    # Apply the function to each group
    final_results = norm_steps.groupby('corpus').apply(process_group).reset_index(drop=True)
    fig = px.bar(
        final_results,
        x='corpus',  # Corpus on the x-axis
        y='annotated_difference_ratio',  # Difference in annotated data points on the y-axis
        color='strategy',  # Color by strategy
        barmode='group',  # Group bars by strategy within each corpus
        title='Difference in Annotated Data Points for Each Corpus and Strategy',
        labels={
            'annotated_difference': 'Annotated Data Points Difference',
            'corpus': 'Corpus',
            'strategy': 'Strategy'
        },
        height=600
    )

    fig.write_html("./results/annotations-diffs.html")

    res: pd.DataFrame = final_results.groupby("strategy")["annotated_difference_ratio"].mean()
    res.to_csv("./results/annotation_diff_ratio_mean.csv")


if __name__ == "__main__":
    main()
