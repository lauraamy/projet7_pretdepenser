import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

def plotly_plot(chart_type: str, df):
    """ return plotly plots """

    if chart_type == "Scatter":
        with st.echo():
            fig = px.scatter(
                data_frame=df,
                x="bill_depth_mm",
                y="bill_length_mm",
                color="species",
                title="Bill Depth by Bill Length",
            )
    elif chart_type == "Histogram":
        with st.echo():
            fig = px.histogram(
                data_frame=df,
                x="bill_depth_mm",
                title="Count of Bill Depth Observations",
            )
    elif chart_type == "Bar":
        with st.echo():
            fig = px.histogram(
                data_frame=df,
                x="species",
                y="bill_depth_mm",
                title="Mean Bill Depth by Species",
                histfunc="avg",
            )
            # by default shows stacked bar chart (sum) with individual hover values
    elif chart_type == "Boxplot":
        with st.echo():
            fig = px.box(data_frame=df, x="species", y="bill_depth_mm")
    elif chart_type == "Line":
        with st.echo():
            fig = px.line(
                data_frame=df,
                x=df.index,
                y="bill_length_mm",
                title="Bill Length Over Time",
            )
    elif chart_type == "3D Scatter":
        with st.echo():
            fig = px.scatter_3d(
                data_frame=df,
                x="bill_depth_mm",
                y="bill_length_mm",
                z="body_mass_g",
                color="species",
                title="Interactive 3D Scatterplot!",
            )

    return fig
