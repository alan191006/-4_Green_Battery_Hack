import pandas as pd
import plotly.express as px
from pathlib import Path


def load_data(file_name):
    """Load data from a CSV file."""
    here = Path(__file__).parent
    file_path = here / file_name
    return pd.read_csv(file_path)


def calc_bounds(data, column, n_std=3):
    """Calculate upper and lower bounds based on mean and standard deviation."""
    mean = data[column].mean()
    std = data[column].std()
    upper_bound = mean + n_std * std
    lower_bound = mean - n_std * std
    return mean, upper_bound, lower_bound


def plot_data(data, column, mean, upper_bound, lower_bound, n_std=3):
    """Plot data with mean and bounds."""

    fig = px.line(data, y=column)

    fig.add_hline(
        y=mean,
        line_dash="dash",
        line_color="green",
        annotation_text="Mean",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=upper_bound,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Mean + {n_std}σ",
        annotation_position="top right",
    )
    fig.add_hline(
        y=lower_bound,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Mean - {n_std}σ",
        annotation_position="bottom right",
    )
    fig.show()


if __name__ == "__main__":
    df = load_data("no_missing_training_data.csv")

    mean_price, upper_bound, lower_bound = calc_bounds(df, "price")

    print("Mean Price:", mean_price)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)

    plot_data(df, "price", mean_price, upper_bound, lower_bound)
