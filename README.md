# LiDAR Sampling Robustness

This project tests whether projected-row dropout improves LiDAR semantic segmentation under reduced vertical sampling without materially reducing full-scan accuracy.

## Why

LiDAR segmentation models are usually trained and evaluated with one sampling pattern. Their accuracy can drop when the vertical sampling pattern changes. We want to measure that failure and test a simple training-time intervention.

## Dataset

I used SemanticKITTI in a previous project and already have it available locally. Reusing it lets me focus on the robustness experiment instead of spending time on another dataset setup. It also provides point-level semantic labels and a standard sequence split.

SemanticKITTI is based on KITTI odometry scans captured with a Velodyne HDL-64E. The sensor uses 64 vertical laser beams while rotating through a 360-degree horizontal field of view. It runs at 10 Hz and records roughly 100,000 points per scan. This produces the structured vertical sampling pattern studied in this project.

## Current direction

The current plan is to use Pointcept with its SpConv SparseUNet baseline. One model will use the standard training pipeline. A second, otherwise identical model will add projected-row dropout during training.

## Plan

1. Validate structured and random sampling masks on SemanticKITTI.
2. Train two identical models with and without projected-row dropout.
3. Evaluate both models on full, structured, and random sampling conditions.
4. Compare robustness gains against clean-scan and class-specific regressions.

The sampling conditions are synthetic proxies. They do not represent real 16-beam or 32-beam sensors.
