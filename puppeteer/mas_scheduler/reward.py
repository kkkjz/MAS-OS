"""Reward calculation for Scheduler and Router.

Reward functions:
- Router: r_rou_{i,t} = α * h_{i,t} + η * R^task_i
  - h_{i,t} = useful_count / routed_count (hit rate from agent feedback)
  - R^task_i = final task reward from evaluator

- Scheduler: r_sch_{i,t} = w_{i,t} * R^task_i - λ_tok * c_{i,t}
  - w_{i,t} = (t / T_i)^γ (time-based weight, later steps more important)
  - c_{i,t} = token consumption at step t
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import torch

logger = logging.getLogger("MAS")


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    
    # Router reward hyperparameters
    alpha: float = 1.0        # Weight for hit rate h_{i,t}
    eta: float = 0.5          # Weight for task reward R^task on router
    
    # Scheduler reward hyperparameters
    gamma_time: float = 2.0   # Exponent for time weight w_{i,t} = (t/T)^γ
    lambda_tok: float = 0.001 # Token cost penalty coefficient
    
    # Discount factor for return calculation
    gamma_discount: float = 1.0  # 1.0 = no discounting (episodic)
    
    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 10.0  # Clip rewards to [-clip, clip]


@dataclass
class StepRecord:
    """Record of a single step in the trajectory."""
    
    step_idx: int                     # t (1-indexed for formulas)
    
    # Scheduler decision
    scheduler_action: str             # Agent name chosen
    scheduler_log_prob: float = 0.0   # log π(a|s) for scheduler
    scheduler_prompt: str = ""        # Full prompt used
    
    # Router decision
    router_action: List[int] = field(default_factory=list)  # Indices selected
    router_log_prob: float = 0.0      # log π(a|s) for router
    router_prompt: str = ""           # Full prompt used
    
    # Agent feedback (for router hit rate)
    routed_count: int = 0             # k_{i,t}: number of memories routed
    useful_count: int = 0             # u_{i,t}: number marked useful by agent
    
    # Token consumption (for scheduler cost)
    token_count: int = 0              # c_{i,t}: tokens consumed this step
    
    # Computed rewards (filled after episode ends)
    router_reward: float = 0.0
    scheduler_reward: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Compute h_{i,t} = u_{i,t} / k_{i,t}."""
        if self.routed_count == 0:
            return 0.0
        return self.useful_count / self.routed_count


@dataclass  
class EpisodeTrajectory:
    """Complete trajectory for one episode (task)."""
    
    task_id: str
    steps: List[StepRecord] = field(default_factory=list)
    task_reward: float = 0.0          # R^task_i from evaluator
    total_tokens: int = 0             # Sum of all token costs
    
    @property
    def total_steps(self) -> int:
        """T_i: total number of steps."""
        return len(self.steps)
    
    def add_step(self, step: StepRecord):
        """Add a step record."""
        self.steps.append(step)
        self.total_tokens += step.token_count
    
    def set_task_reward(self, reward: float):
        """Set the final task reward from evaluator."""
        self.task_reward = reward


class RewardCalculator:
    """
    Computes rewards for Scheduler and Router based on episode trajectory.
    
    Key formulas:
    - Router: r^rou_{i,t} = α * h_{i,t} + η * R^task_i
    - Scheduler: r^sch_{i,t} = w_{i,t} * R^task_i - λ_tok * c_{i,t}
      where w_{i,t} = (t / T_i)^γ
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
    
    def compute_rewards(self, trajectory: EpisodeTrajectory) -> EpisodeTrajectory:
        """
        Compute all step rewards for an episode trajectory.
        
        This should be called AFTER the episode ends and task_reward is known.
        
        Args:
            trajectory: The episode trajectory with steps recorded
            
        Returns:
            The same trajectory with rewards filled in
        """
        T_i = trajectory.total_steps
        R_task = trajectory.task_reward
        
        if T_i == 0:
            logger.warning("[Reward] Empty trajectory, nothing to compute")
            return trajectory
        
        logger.info(f"[Reward] Computing rewards for {T_i} steps, R^task = {R_task:.4f}")
        
        for step in trajectory.steps:
            t = step.step_idx  # 1-indexed
            
            # Router reward: r^rou = α * h_{i,t} + η * R^task
            h_it = step.hit_rate
            r_rou = self.config.alpha * h_it + self.config.eta * R_task
            step.router_reward = self._clip_reward(r_rou)
            
            # Scheduler reward: r^sch = w_{i,t} * R^task - λ_tok * c_{i,t}
            # w_{i,t} = (t / T_i)^γ
            w_it = (t / T_i) ** self.config.gamma_time
            c_it = step.token_count
            r_sch = w_it * R_task - self.config.lambda_tok * c_it
            step.scheduler_reward = self._clip_reward(r_sch)
            
            logger.debug(
                f"[Reward] Step {t}: h={h_it:.3f}, w={w_it:.3f}, "
                f"r_rou={step.router_reward:.4f}, r_sch={step.scheduler_reward:.4f}"
            )
        
        return trajectory
    
    def compute_returns(
        self,
        trajectory: EpisodeTrajectory,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute returns G_t for each step (for policy gradient).
        
        G_t = sum_{t'=t}^{T} γ^{t'-t} * r_{t'}
        
        Args:
            trajectory: Trajectory with rewards computed
            
        Returns:
            (scheduler_returns, router_returns) lists
        """
        T = trajectory.total_steps
        gamma = self.config.gamma_discount
        
        # Collect rewards in order
        sch_rewards = [s.scheduler_reward for s in trajectory.steps]
        rou_rewards = [s.router_reward for s in trajectory.steps]
        
        # Compute returns (backward cumulative sum with discount)
        sch_returns = self._compute_discounted_returns(sch_rewards, gamma)
        rou_returns = self._compute_discounted_returns(rou_rewards, gamma)
        
        # Optional normalization
        if self.config.normalize_rewards:
            sch_returns = self._normalize(sch_returns)
            rou_returns = self._normalize(rou_returns)
        
        return sch_returns, rou_returns
    
    def _compute_discounted_returns(
        self,
        rewards: List[float],
        gamma: float,
    ) -> List[float]:
        """Compute discounted cumulative returns from the end."""
        T = len(rewards)
        returns = [0.0] * T
        
        # G_T = r_T
        # G_t = r_t + γ * G_{t+1}
        running_return = 0.0
        for t in range(T - 1, -1, -1):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _normalize(self, values: List[float]) -> List[float]:
        """Normalize values to have zero mean and unit variance."""
        if len(values) == 0:
            return values
        
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
        std = math.sqrt(var) + 1e-8
        
        return [(v - mean) / std for v in values]
    
    def _clip_reward(self, reward: float) -> float:
        """Clip reward to configured range."""
        return max(-self.config.reward_clip, min(self.config.reward_clip, reward))


def compute_task_reward_from_evaluator(
    correct: bool,
    task_type: str = "default",
    partial_score: Optional[float] = None,
) -> float:
    """
    Convert evaluator result to numerical task reward R^task.
    
    Args:
        correct: Whether the task was solved correctly
        task_type: Type of task (affects reward scale)
        partial_score: Optional partial credit [0, 1]
        
    Returns:
        Numerical reward value
    """
    if partial_score is not None:
        # Use partial score directly, scaled to [-1, 1]
        return 2.0 * partial_score - 1.0
    
    # Binary reward based on task type
    reward_scales = {
        "MMLU-Pro": 1.0,
        "GSM-Hard": 1.0,
        "gsm-hard": 1.0,
        "SRDD": 1.0,
        "CW": 0.5,  # Creative writing is harder to evaluate
        "default": 1.0,
    }
    
    scale = reward_scales.get(task_type, 1.0)
    return scale if correct else -scale


class TrajectoryBuffer:
    """
    Buffer to store episode trajectories for batch training.
    
    Supports:
    - Collecting multiple episodes
    - Converting to training batches
    - Separating scheduler and router data
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.trajectories: List[EpisodeTrajectory] = []
        self.reward_calculator = RewardCalculator()
    
    def add(self, trajectory: EpisodeTrajectory):
        """Add a completed trajectory."""
        if len(self.trajectories) >= self.max_size:
            self.trajectories.pop(0)  # Remove oldest
        self.trajectories.append(trajectory)
    
    def clear(self):
        """Clear all trajectories."""
        self.trajectories.clear()
    
    def get_training_batch(
        self,
        compute_rewards: bool = True,
    ) -> Dict[str, List]:
        """
        Convert trajectories to training batch format.
        
        Returns dict with:
        - scheduler_prompts: List of prompts
        - scheduler_actions: List of action strings
        - scheduler_returns: List of returns G_t
        - scheduler_log_probs: List of original log probs
        - router_prompts: List of prompts
        - router_actions: List of action index lists
        - router_returns: List of returns G_t
        - router_log_probs: List of original log probs
        """
        batch = {
            "scheduler_prompts": [],
            "scheduler_actions": [],
            "scheduler_returns": [],
            "scheduler_log_probs": [],
            "router_prompts": [],
            "router_actions": [],
            "router_returns": [],
            "router_log_probs": [],
        }
        
        for traj in self.trajectories:
            if compute_rewards:
                traj = self.reward_calculator.compute_rewards(traj)
            
            sch_returns, rou_returns = self.reward_calculator.compute_returns(traj)
            
            for i, step in enumerate(traj.steps):
                # Scheduler data
                batch["scheduler_prompts"].append(step.scheduler_prompt)
                batch["scheduler_actions"].append(step.scheduler_action)
                batch["scheduler_returns"].append(sch_returns[i])
                batch["scheduler_log_probs"].append(step.scheduler_log_prob)
                
                # Router data (only if routing happened)
                if step.router_prompt:
                    batch["router_prompts"].append(step.router_prompt)
                    batch["router_actions"].append(step.router_action)
                    batch["router_returns"].append(rou_returns[i])
                    batch["router_log_probs"].append(step.router_log_prob)
        
        return batch
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def total_steps(self) -> int:
        """Total number of steps across all trajectories."""
        return sum(t.total_steps for t in self.trajectories)

