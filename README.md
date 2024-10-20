# Learning Curve Simulation

This repository contains the materials for replicating simulations from the "Mapping the Learning Curve" paper.

- The `Preprocessing` folder contains Jupyter notebooks for processing Gesture and Sentence input data. The processed data are saved as `.npy` files, enabling more efficient loading for simulations.

- The `Simulation` folder includes Python scripts for running each simulation. It also contains the generated KNN classification results (`/allEpoch_dfs`) and RNN model performance data (`/performance`). The `/allEpoch_dfs` files are used for visualization and further analysis.

- The `Visualization and Analysis` folder contains R scripts for visualizing multi-dimensional learning curves. It demonstrates the use of four key measures (**start**, **max**, **tmax**, and **end-start**) to visualize and conduct statistical inference on the constructed learning curves.

## Replicating the R Analysis

- The `fuse_data.R` script gathers data from `/allEpoch_dfs` and exports it to an `.Rd` file.
- The `result.R` script uses this output for plotting and conducting significance tests.
