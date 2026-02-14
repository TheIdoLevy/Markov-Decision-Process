# MDP Solver: Value and Policy Iteration

This repository contains a Python implementation of a **Markov Decision Process (MDP)** solver. It utilizes two fundamental reinforcement learning algorithms—**Value Iteration** and **Policy Iteration**-to find the optimal policy and utility values for a given grid-based environment.

---

## ## Features

* **MDP Modeling:** Supports custom states, reward matrices, and transition probabilities.
* **Stochastic Transitions:** Includes a transition model where actions have a success probability () and potential "drift" into diagonal states.
* **Value Iteration:** Computes the optimal utility for each state by applying the iterative Bellman Equation.
* **Policy Iteration:** Alternates between policy evaluation and policy improvement to achieve convergence.
* **Visualization:** * Generates **Seismic Heatmaps** of state utilities using `matplotlib`.
* Plots convergence graphs showing the number of evaluation iterations per policy step.
* Prints a formatted grid of the final policy (Directional arrows, 'x' for blocks, 'o' for terminals).



---

## ## Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install numpy matplotlib

```

---

## ## How It Works

### ### Transition Model

The agent attempts to move in one of four directions: **Up, Down, Left, or Right**.

* **Success ():** 80% chance to move to the intended state (cusomizable).
* **Drift ():** 20% chance (split between two diagonal directions) to move elsewhere.
* **Obstacles:** If the agent hits a blocked state or boundary, the probability is redistributed back to the successful transition (depending on the specific logic in `calc_prob`).

### ### Algorithms

1. **Value Iteration:** Iteratively updates state utilities until the maximum change () is below the threshold .
2. **Policy Iteration:** Starts with an initial policy, evaluates it to find utilities, and then improves the policy greedily until no further changes occur.

---

## ## Usage

1. **Prepare Data:** The script expects a `.npz` file containing:
* `states`: A grid where `1` = reachable, `0` = blocked, `-1` = terminal.
* `rewards`: A grid of numerical reward values for each state.


2. **Configure Path:** In the `main()` function, update the `filename` variable:
```python
filename = "your_data_file.npz"

```


3. **Run:**
```bash
python mdp_solver.py

```



---

## ## Output Examples

### ### Utility Heatmap

A seismic color map where blue represents negative utility, white represents neutral, and red represents high utility.

### ### Text-Based Policy

The console will print a grid like this:

```text
^  ^  ^  o  
^  x  ^  o  
>  >  >  v  

```

* `^, v, <, >`: Directions
* `x`: Blocked State
* `o`: Terminal State

---
