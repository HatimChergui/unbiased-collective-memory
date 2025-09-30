# Unbiased Collective Memory for Agentic Negotiation


`If you use this code or any (modified) part of it, please cite the corresponding paper:`

**H. Chergui, M. Catalan Cid, P. Sayyad Khodashenas, D. Camps Mur, C. Verikoukis, ''Toward an Unbiased Collective Memory for Efficient LLM-Based Agentic 6G Cross-Domain Management,'' [Online] Available: https://arxiv.org/submit/6843423/view, 2025.**

## Overview

The **Unbiased Collective Memory** is a Python module designed to enhance the decision-making of negotiating LLM-powered agents by mitigating common cognitive biases in memory retrieval. Instead of relying purely on the most recent or semantically similar past negotiation outcomes, this system applies calculated penalties and bonuses to promote retrieval of **objectively high-performing strategies** and **salient learning moments (failures)**.

This approach addresses the following biases:

1.  **Temporal Bias (Recency/Primacy):** Controlled by a `time-decay score` that weights memories based on their age.
2.  **Anchor Bias:** Mitigated by an **Anchor Penalty** that suppresses memories too similar to the first proposal of the current negotiation.
3.  **Confirmation/availability Biases:** Addressed by an **Inflection Bonus** that increases the salience of past negotiation failures (SLA violations or unresolved talks), ensuring the agent learns from mistakes.

-----

## 🚀 Setup and Integration

This module requires standard Python libraries, primarily NumPy for numerical operations.

### Dependencies

```bash
pip install numpy
```

### Usage Example

```python
from your_module import UnbiasedCollectiveMemory

# 1. Initialize the Memory (with Debaising Parameters)
# Default values provide a balanced approach to bias mitigation.
memory = UnbiasedCollectiveMemory(
    alpha=1.0,           # Weight for Semantic Similarity
    beta=0.5,            # Weight for Time Decay
    delta=1.5,           # Inflection Bonus, which boosts failure memories and diversity
    decay_rate_factor=5.0, # Controls the steepness of time decay (higher value = slower memory decay)
    anchor_penalty_factor=1.0 # Strength of penalty for being close to the Anchor
)

# 2. Define the Initial Anchor Point (The First Proposal)
# This is crucial for calculating the Anchor Penalty during retrieval.
initial_anchor_config = {
    "ran_bandwidth_mhz": 25.0, 
    "edge_cpu_frequency_ghz": 30.0
}
memory.initial_anchor_point = initial_anchor_config

# 3. Distill a Past Negotiation Outcome (Logging a Strategy)
# Strategies 1 & 2 are similar to the anchor but Strategy 2 failed (High Inflection Bonus).
memory.distill_strategy({
    "trial_number": 1, 
    "agreed_config": {"ran_bw": 26.0, "edge_cpu": 31.0},
    "saved_energy_percent": 15.0,
    "description": "Compromise with low-medium BW. High energy savings.",
    "sla_violation_occurred": False, "unresolved_negotiation": False,
    "final_metrics": {"current_traffic_arrival_rate_bps": 50000000}
})

# 4. Query the Memory
current_trial = 10 
query_context = {
    "current_trial_number": current_trial,
    "keywords": ["medium", "traffic", "balanced", "performance"],
    "initial_anchor_point": initial_anchor_config # Pass the anchor for penalty calculation
}

retrieval = memory.query_memory(query_context=query_context)

print(f"Top Retrieved Strategy (Trial {current_trial}):\n")
for i, strategy in enumerate(retrieval['retrieved_strategies']):
    print(f"Rank {i+1}: {strategy['description']}")
    print(f"  > Trial Age: {current_trial - strategy['context']['trial_number']}")

```

-----

## 🛠️ Class Methods and Parameters

### `UnbiasedCollectiveMemory(self, ...)`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `alpha` | `float` | `1.0` | **Semantic Weight.** Emphasizes keyword matching relevance. |
| `beta` | `float` | `0.5` | **Temporal Weight.** Controls how much recency influences the score. |
| `delta` | `float` | `0.5` | **Inflection Bonus.** Added to strategies that resulted in failure/SLA violation to encourage learning from mistakes. |
| `decay_rate_factor` | `float` | `5.0` | **decay factor.** Controls the steepness of time decay (higher value = slower memory decay). |
| `anchor_penalty_factor` | `float` | `0.5` | **Anchor Bias Mitigation.** The strength of the penalty applied to memories too close to the `initial_anchor_point`. |

### Key Methods

#### `distill_strategy(self, outcome_data: Dict[str, Any])`

Logs the outcome of a completed negotiation trial. It calculates an objective `performance_score` (between -1.0 and $\approx 1.5$) based on SLA status and energy savings, which is used internally for debiased retrieval.

#### `query_memory(self, query_context: Dict[str, Any])`

Retrieves the top 5 relevant strategies based on the combined debiased scoring mechanism:

$$\text{Final Score} = \alpha \cdot \text{Semantic} + \beta \cdot \text{Time Decay} + \delta \cdot \text{Inflection Bonus} - \text{Anchor Penalty}$$

The `query_context` **must** include the `initial_anchor_point` (the configuration of the very first agent proposal in the negotiation) to correctly calculate the anchor penalty.
