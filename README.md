# Subgoal Decomposition on Smooth Trajectories

Subgoal extraction for robotic manipulation trajectories. 

## Overview

Current hierarchical robotic manipulation policies such as RVT-2 and PerAct rely on heuristic-based keyframe extraction methods that assume explicit pauses in demonstrations. However, smooth trajectories often do not contain clear transition points, making subgoal extraction challenging.

This project investigates whether trajectory simplification methods can provide meaningful subgoals for hierarchical policy learning on smooth demonstrations. Furthermore, we investigate more effective subgoal decomposition methods for smooth trajectories that do not contain explicit pauses or clear transition points.

## Method

We apply the Ramer-Douglas-Peucker (RDP) algorithm to end-effector trajectories to extract sparse keyframes from smooth demonstrations. These keyframes are then used to train:

- RVT-2 as the high-level subgoal predictor
- Goal-conditioned DP3 as the low-level controller

We evaluate the hierarchical policy on RLBench tasks using both original and smoothed demonstrations.

## Repository Structure 

## Project Website
