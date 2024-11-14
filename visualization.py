# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import io

def create_multi_spider_chart(all_dimension_scores, show=True):
    """
    Create a spider chart visualization for multiple inputs.
    
    :param all_dimension_scores: A dictionary with input names as keys and dimension score dictionaries as values.
    :param show: Whether to display the chart using st.pyplot().
    :return: The matplotlib figure object.
    """
    # Extract labels from the first input's dimension scores
    labels = list(next(iter(all_dimension_scores.values())).keys())
    num_vars = len(labels)
    
    # Compute angle of each axis in the plot (in radians)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop
    angles += angles[:1]
    
    # Initialize the radar plot with increased height to accommodate the legend below
    fig, ax = plt.subplots(figsize=(8, 10), subplot_kw=dict(polar=True))
    
    # Define color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_dimension_scores)))
    
    for idx, (input_name, dimension_scores) in enumerate(all_dimension_scores.items()):
        values = list(dimension_scores.values())
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, color=colors[idx], linewidth=2, label=input_name)
        ax.fill(angles, values, color=colors[idx], alpha=0.25)
    
    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    
    # Set the range and labels for the radial axis
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax.set_ylim(0, 100)
    
    # Set the title of the chart
    ax.set_title("Dimension Scores", size=15, y=1.08)
    
    # Determine the number of legend columns based on the number of groups
    num_legend_columns = min(len(all_dimension_scores), 4)  # Adjust the max number of columns as needed
    
    # Reposition legend below the chart
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              fancybox=True, shadow=True, ncol=num_legend_columns, fontsize=10)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    if show:
        st.pyplot(fig)
    return fig

def chart_to_image(fig):
    """
    Converts a matplotlib figure to a PNG image.
    
    :param fig: The matplotlib figure object.
    :return: Bytes object containing the image data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

