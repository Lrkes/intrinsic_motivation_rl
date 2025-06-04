# Intrinsic Motivation in RL – Gymnasium Experiments

## Overview
This project explores the use of intrinsic motivation on reinforcement learning agents.

## Motivation
I previously implemented rl algorithms in Gymnasium and want to now use that as a baseline to explore how intrinsic motivation can improve learning in classic RL environments like FrozenLake.

## Research Questions
How does adding a count-based intrinsic motivation bonus affect the learning speed and performance of a Q-learning agent in FrozenLake?

How does the agent’s exploration behavior change with and without the bonus?

Additionally, I reflect on larger questions—such as how agents might learn to adapt their own intrinsic motivation automatically (as humans do)—as future research directions.

Project Goals
Implement and compare baseline Q-learning and Q-learning with a count-based exploration bonus.

Analyze learning curves, state visitation, and sample efficiency.

Document findings, difficulties, and insights to contribute to help my own learning.

Experimental Progress and Decisions
I first applied Q-learning (with and without count-based intrinsic motivation) to FrozenLake. Based on single-run learning curves and state visit plots, I did not observe significant improvements with count-based bonuses. Given the stochastic nature of FrozenLake, I decided to try a more deterministic environment.

I moved on to the Taxi-v3 environment, where the agent’s behavior is deterministic and the reward structure is sparse. However, a side-by-side comparison of single runs—with and without the count-based bonus—showed only moderate or negligible improvements in learning speed or final performance.

Instead of going deeper into more granular plotting on these environments, I decided to look for environments where intrinsic motivation is more likely to show a clear benefit.

My next step is to identify and experiment with environments that are well-suited for intrinsic motivation methods—especially those that are deterministic, sparse-reward, and challenging for exploration, such as NChain-v0, CliffWalking-v0, or deterministic FrozenLake.

Structure
/qlearning_baseline/ — Classic Q-learning implementation.

/intrinsic_motivation/ — Q-learning with count-based exploration bonus.

/analysis/ — Plots, comparisons, and notes.

README.md — This document.

How to Use
Clone the repo.

Run the baseline and intrinsic motivation scripts.

Use analysis notebooks to visualize learning progress and exploration.


Acknowledgements
The Q-learning baseline code is adapted from my RL internship in Japan.
See the original project here: [https://github.com/Lrkes/ReinforcementLearningInternship].