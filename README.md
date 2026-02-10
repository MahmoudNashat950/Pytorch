Regression with Missing Features using Autoencoders
Overview

This project tackles regression with missing data.
We have two data sources:

X1: 40 numerical features (range [0–1])

X2: 60 numerical features (range [0–1])

Goal: Predict a target value y from the concatenated input x = [X1 + X2] even if up to 90% of features are missing during inference.

Challenge

Standard regression on concatenated features fails when many features are missing.

Missing features are simulated during training by randomly dropping 0–90% of features per example.

We need a network that can recover missing features before regression.

Solution Design

Use three denoising autoencoders:

Autoencoder for X1

Autoencoder for X2

Autoencoder for concatenated [X1 + X2]

Each autoencoder produces a reconstruction loss.

The recovered vector is fed into a regressor to predict y (regression loss).

Final loss = average of 3 autoencoder losses + regression loss.

Features

Handles extreme missing data (up to 90%)

Denoising autoencoders recover missing features

Modular design: separate encoders for each source + concatenated input

PyTorch implementation, GPU-ready

Usage

Simulate missing data with random dropout:

x1_noise = drop_out(x1, r)  # r in [0, 0.9]
x2_noise = drop_out(x2, r)


Forward pass through the 3 autoencoders and regressor.

Compute loss:

loss = (loss_x1 + loss_x2 + loss_concat + loss_regression)/4


Backpropagate and optimize.
