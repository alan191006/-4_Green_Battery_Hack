import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_results(profits, market_prices, battery_soc, actions):
    """
    Plot the bids, profits, market prices, and battery state of charge over time.

    :param profits: List of profits for each time step.
    :param market_prices: List of market prices for each time step.
    :param battery_soc: List of battery state of charge values for each time step.
    :param actions: List of actions (buy/sell amount) for each time step.
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Market Price",
            "Profit",
            "Battery State of Charge",
            "Actions",
        ),
        row_heights=[0.3, 0.3, 0.2, 0.2],
    )

    # Plot market prices
    fig.add_trace(
        go.Scatter(
            x=list(range(len(market_prices))),
            y=market_prices,
            mode="lines",
            name="Market Price",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    # Plot profits
    fig.add_trace(
        go.Scatter(
            x=list(range(len(profits))),
            y=profits,
            mode="lines",
            name="Profit",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    # Plot battery state of charge
    fig.add_trace(
        go.Scatter(
            x=list(range(len(battery_soc))),
            y=battery_soc,
            mode="lines",
            name="Battery State of Charge",
            line=dict(color="purple"),
        ),
        row=3,
        col=1,
    )

    # Plot actions
    fig.add_trace(
        go.Scatter(
            x=list(range(len(actions))),
            y=actions,
            mode="lines",
            name="Actions",
            line=dict(color="blue", width=0.5),
        ),
        row=4,
        col=1,
    )

    # Set x-axis label
    fig.update_xaxes(title_text="Time Step")

    # Set y-axis label for battery state of charge
    fig.update_yaxes(title_text="Battery State of Charge (kWh)", row=3, col=1)

    # Set y-axis label for actions
    fig.update_yaxes(title_text="Actions (kWh)", row=4, col=1)

    # Add secondary y-axis for profit subplot
    fig.update_yaxes(
        title_text="Profit",
        row=2,
        col=1,
        secondary_y=True,
        tickformat=".2f",
        showgrid=False,
    )

    # Adjust layout
    fig.update_layout(
        height=800, width=800, title_text="Results Plot", showlegend=False
    )
    fig.show()
