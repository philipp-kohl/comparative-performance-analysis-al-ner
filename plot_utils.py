import pandas as pd
import plotly.graph_objects as go


def plot_metrics_with_std_error_bars(stats_per_step):
    """
    Plots the mean and standard deviation for each step using Plotly with error bars.

    Parameters:
    - stats_per_step (pd.DataFrame): A DataFrame containing the step, mean, and std for each step.
    """
    steps = stats_per_step['step']
    means = stats_per_step['mean']
    stds = stats_per_step['std']

    # Create the plot
    fig = go.Figure()

    # Add mean line with error bars representing the standard deviation
    fig.add_trace(go.Scatter(
        x=steps,
        y=means,
        mode='lines+markers',
        error_y=dict(
            type='data',
            array=stds,
            visible=True,
            color='red',
            thickness=1.5,
            width=2
        ),
        name='Mean with Std Dev'
    ))

    # Add standard deviation band (optional, can remove if error bars are enough)
    fig.add_trace(go.Scatter(
        x=pd.concat([steps, steps[::-1]]),  # x coordinates for upper and lower bounds
        y=pd.concat([means + stds, (means - stds)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Std Dev Band'
    ))

    # Update layout
    fig.update_layout(
        title='Mean and Standard Deviation of Metrics per Step with Error Bars',
        xaxis_title='Step',
        yaxis_title='Mean Value',
        template='plotly_white'
    )

    fig.show()