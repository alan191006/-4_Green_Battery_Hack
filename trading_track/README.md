# Trading track submission

This directory contains our team's submission for the Trading track. Here, you'll find our model architectures along necessary files to run including the weight. Please note that it does not include Docker configuration nor experimental script.

## Solution description

We rewrote the initially numpy-ed battery environment (`battery.py`) to PyTorch

> [...] to make it (somewhat) differentiable, so we can simply create a neural network model to predict an action per step, calculate the profit and battery state change within the PyTorch model, and later add up all the profits and directly use back propagation to optimise the actions. It is quite "brute-force" and there are plenty of reasons why it should not work well (e.g. vanishing gradient, inability to find "true" optimal), but somehow it is our best model :p

<p align="right">- Peter Wang, May 2024 -</p>
