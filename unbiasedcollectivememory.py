import numpy as np
from typing import Dict, Any, List, Optional
import re
from collections import defaultdict
import json

# --- 1. Reusable Unbiased Collective Memory Class ---

class UnbiasedCollectiveMemory:
    """
    A standalone Collective Memory system designed to mitigate temporal and anchor biases,
    and prioritize performance based on multiple weighted factors.
    """

    def __init__(self, 
                 alpha: float = 1.0, 
                 beta: float = 0.5, 
                 delta: float = 0.5,
                 decay_rate_factor: float = 80.0,
                 anchor_penalty_factor: float = 0.5):
        """
        Initializes the memory with debiasing parameters.
        alpha (semantic), beta (temporal), delta (inflection bonus for failure).
        """
        self.distilled_strategies: List[Dict[str, Any]] = []
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.decay_rate_factor = decay_rate_factor
        self.anchor_penalty_factor = anchor_penalty_factor

        # NOTE: Using placeholder ranges for normalization in anchor penalty
        self.MAX_RAN_BW = 40.0
        self.MIN_RAN_BW = 5.0
        self.MAX_EDGE_CPU = 45.0

    def _tokenize_text(self, text: str) -> set:
        """Helper to tokenize text for semantic similarity."""
        return set(re.findall(r'\b\w+\b', text.lower()))

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Helper to compute Jaccard similarity (semantic relevance)."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def _calculate_anchor_penalty(self, strategy: Dict[str, Any], anchor: Dict[str, float]) -> float:
        """Calculates a penalty score for memories too close to the initial anchor point."""
        if not anchor:
            return 0.0
        
        anchor_bw = anchor.get("ran_bandwidth_mhz")
        anchor_cpu = anchor.get("edge_cpu_frequency_ghz")
        strategy_bw = strategy["action"].get("last_ran_proposal_mhz")
        strategy_cpu = strategy["action"].get("last_edge_proposal_ghz")
        
        if strategy_bw is None or strategy_cpu is None or anchor_bw is None or anchor_cpu is None:
            return 0.0

        ran_range = self.MAX_RAN_BW - self.MIN_RAN_BW
        cpu_range = self.MAX_EDGE_CPU

        # Normalize the deviation of each dimension relative to its operating range (0 to 1)
        norm_bw_dev = abs(strategy_bw - anchor_bw) / ran_range if ran_range > 0 else 0
        norm_cpu_dev = abs(strategy_cpu - anchor_cpu) / cpu_range if cpu_range > 0 else 0

        deviation = (norm_bw_dev + norm_cpu_dev) / 2
        
        # Penalty is proportional to anchor_penalty_factor and inversely proportional to deviation (1-deviation)
        penalty = self.anchor_penalty_factor * (1.0 - deviation)
        return max(0.0, penalty)


    def distill_strategy(self, outcome_data: Dict[str, Any]):
        """Distills and stores a single strategy episode into the collective memory."""
        sla_violation_occurred = outcome_data.get("sla_violation_occurred", False)
        unresolved_negotiation = outcome_data.get("unresolved_negotiation", False)
        
        performance_score = -1.0
        if not sla_violation_occurred and not unresolved_negotiation:
            # A placeholder objective score: ranges from 0.0 to 1.0 (Higher is better)
            performance_score = 0.5 + (outcome_data.get("saved_energy_percent", 0.0) / 200.0)

        strategy = {
            "context": {
                "trial_number": outcome_data.get("trial_number", 0),
                "traffic_level_category": outcome_data.get("traffic_level_category", "medium")
            },
            "action": {
                "last_ran_proposal_mhz": outcome_data["agreed_config"].get("ran_bw"),
                "last_edge_proposal_ghz": outcome_data["agreed_config"].get("edge_cpu")
            },
            "outcome_summary": {
                "negotiation_result": "success" if performance_score > 0 else "failed",
                "sla_violation_occurred": sla_violation_occurred,
                "saved_energy_percent": outcome_data.get("saved_energy_percent", 0.0)
            },
            "description": outcome_data.get("description", "A negotiated outcome."),
            "performance_score": performance_score
        }
        self.distilled_strategies.append(strategy)

    def query_memory(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves the top N strategies using a debiased scoring mechanism.

        Score = (alpha * Semantic) + (beta * Time Decay) + (delta * Inflection Bonus) - (Anchor Penalty)

        :param query_context: Dictionary containing 'current_trial_number', 'keywords', 
                              and 'initial_anchor_point'.
        :return: A dictionary containing the top 5 'retrieved_strategies'.
        """
        current_trial_number = query_context.get("current_trial_number", 0) 
        query_keywords_set = self._tokenize_text(" ".join(query_context.get("keywords", [])))
        initial_anchor = query_context.get("initial_anchor_point", None)
        
        scored_candidates = []
        for strategy in self.distilled_strategies:
            # 1. Semantic Score (alpha)
            strategy_text = strategy["description"]
            strategy_keywords_set = self._tokenize_text(strategy_text)
            semantic_similarity = self._jaccard_similarity(query_keywords_set, strategy_keywords_set)

            # 2. Temporal Score (beta): Combats recency/primacy bias
            age = current_trial_number - strategy["context"].get("trial_number", 0)
            time_decay_score = np.exp(-max(0, age) / self.decay_rate_factor) 
            
            # 3. Anchor Penalty: Combats anchor bias by penalizing closeness to the initial proposal
            anchor_penalty = self._calculate_anchor_penalty(strategy, initial_anchor) if initial_anchor else 0.0
            
            # 4. Inflection Bonus (delta): Boosts past failure/SLA violation memories
            inflection_bonus = 0.0
            is_failure = strategy['outcome_summary'].get('negotiation_result') in ['unresolved_negotiation', 'agreement_with_sla_violation']
            if is_failure:
                inflection_bonus = self.delta

            # Final Score: Weighted combination of all debiased factors
            final_score = (self.alpha * semantic_similarity) + (self.beta * time_decay_score) + inflection_bonus - anchor_penalty
            
            scored_candidates.append({"strategy": strategy, "final_score": final_score})
        
        # Sort and select top N
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        top_n = 5
        selected_strategies = [candidate["strategy"] for candidate in scored_candidates[:top_n]]

        return {
            "retrieved_strategies": selected_strategies,
            "query_memory_average_score": np.mean([c["final_score"] for c in scored_candidates[:top_n]]) if scored_candidates else 0.0
        }

# --- 2. Mockup of Agentic Use ---

def demonstrate_memory_use():
    """
    Demonstrates the lifecycle of the unbiased collective memory.
    """
    # 1. Initialize the memory with debiasing enabled
    memory = UnbiasedCollectiveMemory(alpha=1.0, beta=0.5, delta=1.5, anchor_penalty_factor=1.0)
    
    # Define the initial anchor point for the negotiation (simulating the first proposal)
    INITIAL_ANCHOR = {
        "ran_bandwidth_mhz": 25.0, 
        "edge_cpu_frequency_ghz": 30.0
    }
    memory.initial_anchor_point = INITIAL_ANCHOR
    
    # 2. Log Past Episodes (Distill Strategies)
    # Strategy 1 (Old, High Energy Saving, Close to Anchor - Should be penalized)
    memory.distill_strategy({
        "trial_number": 1, "traffic_level_category": "medium", 
        "agreed_config": {"ran_bw": 26.0, "edge_cpu": 31.0}, "saved_energy_percent": 15.0,
        "description": "Compromise with low-medium BW and CPU. High energy savings.",
        "sla_violation_occurred": False, "unresolved_negotiation": False,
        "final_metrics": {"current_traffic_arrival_rate_bps": 50000000}
    })
    
    # Strategy 2 (Recent, Failure, Far from Anchor - High Inflection Bonus)
    memory.distill_strategy({
        "trial_number": 8, "traffic_level_category": "high", 
        "agreed_config": {"ran_bw": 40.0, "edge_cpu": 45.0}, "saved_energy_percent": 0.0,
        "description": "Aggressive, high capacity proposal led to SLA violation.",
        "sla_violation_occurred": True, "unresolved_negotiation": False,
        "final_metrics": {"current_traffic_arrival_rate_bps": 80000000}
    })

    # Strategy 3 (Old, Medium Performance, Far from Anchor - Lower Score)
    memory.distill_strategy({
        "trial_number": 3, "traffic_level_category": "low", 
        "agreed_config": {"ran_bw": 10.0, "edge_cpu": 28.0}, "saved_energy_percent": 5.0,
        "description": "Conservative, low BW for low traffic. Minimal performance.",
        "sla_violation_occurred": False, "unresolved_negotiation": False,
        "final_metrics": {"current_traffic_arrival_rate_bps": 30000000}
    })

    # Strategy 4 (Very Recent, Good Performance, Far from Anchor - Should score highly)
    memory.distill_strategy({
        "trial_number": 9, "traffic_level_category": "medium", 
        "agreed_config": {"ran_bw": 35.0, "edge_cpu": 40.0}, "saved_energy_percent": 10.0,
        "description": "Balanced high performance setting for medium traffic.",
        "sla_violation_occurred": False, "unresolved_negotiation": False,
        "final_metrics": {"current_traffic_arrival_rate_bps": 55000000}
    })
    
    print("--- Memory Distillation Complete ---")
    print(f"Total Strategies Stored: {len(memory.distilled_strategies)}\n")
    print(f"Initial Anchor Point (Trial 0): {INITIAL_ANCHOR}\n")


    # 3. Agent Queries the Memory for Guidance (Simulating Trial 10)
    CURRENT_TRIAL = 10
    QUERY_CONTEXT = {
        "current_trial_number": CURRENT_TRIAL,
        "keywords": ["medium", "traffic", "balanced", "energy", "performance"],
        "initial_anchor_point": INITIAL_ANCHOR
    }

    retrieval = memory.query_memory(query_context=QUERY_CONTEXT)
    
    # 4. Display Retrieved Strategies (Debiased Ranking)
    print("--- Memory Retrieval (Debiased Ranking) ---")
    print(f"Query Context: Keywords='medium, traffic, balanced', Trial={CURRENT_TRIAL}\n")
    
    retrieved_strategies = retrieval['retrieved_strategies']
    
    for i, strategy in enumerate(retrieved_strategies):
        # Recalculate anchor deviation for demonstration printout
        anchor_deviation = memory._calculate_anchor_penalty(strategy, INITIAL_ANCHOR)
        
        print(f"Rank {i+1} (Score: {strategy['final_score']:.2f}):")
        print(f"  > Description: {strategy['description']}")
        print(f"  > Trial: {strategy['context']['trial_number']} (Age: {CURRENT_TRIAL - strategy['context']['trial_number']})")
        print(f"  > BW/CPU: {strategy['action']['last_ran_proposal_mhz']:.1f} / {strategy['action']['last_edge_proposal_ghz']:.1f}")
        print(f"  > Performance Score (Objective): {strategy['performance_score']:.2f}")
        print(f"  > **Anchor Penalty (Internal): {anchor_deviation:.2f}**\n")


# 5. Run the demonstration
if __name__ == "__main__":
    demonstrate_memory_use()
