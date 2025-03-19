import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plotPerColumnDistribution(df):
    """Plots histogram/bar graph for numerical and categorical columns."""
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    
    fig, axes = plt.subplots(nrows=1, ncols=len(df.columns), figsize=(5 * len(df.columns), 5))
    
    if len(df.columns) == 1:
        axes = [axes]
    
    for i, col in enumerate(df.columns):
        if df[col].dtype == 'object':
            df[col].value_counts().plot(kind='bar', ax=axes[i])
        else:
            df[col].hist(ax=axes[i])
        axes[i].set_title(col)
    
    plt.tight_layout()
    return fig

def plotCorrelationMatrix(df):
    """Generates and returns a correlation matrix plot."""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    return fig

def plotScatterMatrix(df):
    """Creates scatter and density plots for numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) < 2:
        return "Not enough numerical columns for scatter plot."
    
    pd.plotting.scatter_matrix(num_df, alpha=0.75, figsize=(10, 10), diagonal="kde")
    plt.tight_layout()
    return plt.gcf()

def process_csv(file):
    """Reads CSV and returns table preview and plots."""
    df = pd.read_csv(file.name)
    
    # Preview data
    preview_html = df.head().to_html()
    
    # Generate plots
    dist_plot = plotPerColumnDistribution(df)
    corr_plot = plotCorrelationMatrix(df)
    scatter_plot = plotScatterMatrix(df)
    
    return preview_html, dist_plot, corr_plot, scatter_plot

# Gradio Interface
iface = gr.Interface(
    fn=process_csv,
    inputs=gr.File(label="Upload CSV File"),
    outputs=[
        gr.HTML(label="Data Preview"),
        gr.Plot(label="Column Distribution"),
        gr.Plot(label="Correlation Matrix"),
        gr.Plot(label="Scatter Matrix")
    ],
    title="Interactive Data Explorer",
    description="Upload a CSV file and explore the dataset with visualizations."
)

iface.launch()
