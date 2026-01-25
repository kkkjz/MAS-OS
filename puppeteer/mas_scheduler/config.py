"""Configuration for MAS-style scheduler."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class MASConfig:
    """Configuration for MAS scheduler components."""
    
    # Maximum steps before forcing termination
    max_steps: int = 12
    
    # Memory retrieval parameters
    top_m: int = 7  # TopM candidates from Memer
    top_n: int = 3   # TopN actually routed to agent
    
    # Semantic similarity threshold for memory edges
    similarity_threshold: float = 0.8
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Optional[str] = None
    # Disable embedding-based semantic edges and retrieval (fallback to keyword overlap).
    # Set True only if you have sentence-transformers installed and want semantic linking.
    use_embeddings: bool = True
    
    # LLM settings
    # OpenAI-compatible model name used for auxiliary LLM calls
    # (memory summarization / usefulness feedback / fallback reasoning)
    llm_model: str = ""
    llm_temperature: float = 0.0
    
    # Whether to use LLM for scheduling (False = rule-based fallback)
    use_llm_scheduler: bool = False
    use_llm_router: bool = False
    use_llm_memer: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    # Feedback collection (for router reward during training)
    # Set to False during evaluation to skip feedback API calls
    collect_feedback: bool = False
    
    # OpenAI API settings (loaded from global config)
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None


@dataclass
class DualLoRAConfig:
    """Configuration for Dual LoRA model (Scheduler + Router experts)."""
    
    # Base model
    base_model_name: str = "meta-llama/Llama-3.1-8B"
    
    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Quantization for memory efficiency
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # Device settings
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    
    # Router reward: r_rou = α * h_{i,t} + η * R^task
    alpha: float = 1.0        # Weight for hit rate
    eta: float = 0.5          # Weight for task reward on router
    
    # Scheduler reward: r_sch = w_{i,t} * R^task - λ_tok * c_{i,t}
    gamma_time: float = 2.0   # Exponent for time weight w = (t/T)^γ
    lambda_tok: float = 0.001 # Token cost penalty
    
    # Return calculation
    gamma_discount: float = 1.0  # Discount factor (1.0 = no discounting)
    
    # Normalization
    normalize_rewards: bool = True
    reward_clip: float = 10.0


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    
    # Dual LoRA model config
    lora: DualLoRAConfig = field(default_factory=DualLoRAConfig)
    
    # Reward config
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    
    # PPO/GRPO specific
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Buffer settings
    buffer_size: int = 1000
    min_batch_size: int = 16  # Minimum samples before training
    
    # Checkpointing
    save_steps: int = 100
    checkpoint_dir: str = "./checkpoints/dual_lora"
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    
    # verl specific settings
    use_verl: bool = True
    verl_algorithm: str = "grpo"  # "ppo", "grpo", "reinforce"


def load_mas_config() -> MASConfig:
    """Load MAS config, merging with puppeteer's global config for API keys."""
    config = MASConfig()
    
    # 使用绝对路径加载配置，避免依赖当前工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    global_config_path = os.path.join(project_root, "config", "global.yaml")
    
    if os.path.exists(global_config_path):
        with open(global_config_path, "r", encoding="utf-8") as f:
            global_config = yaml.safe_load(f)
        
        api_keys = global_config.get("api_keys", {})
        config.openai_api_key = api_keys.get("openai_api_key")
        config.openai_base_url = api_keys.get("openai_base_url")
        # Optional: allow specifying model name in global config
        # (supports both "llm_model" and "openai_model" keys)
        config.llm_model = api_keys.get("llm_model") or api_keys.get("openai_model") or config.llm_model
        
        # Use graph settings if available
        graph_config = global_config.get("graph", {})
        if "max_step_num" in graph_config:
            config.max_steps = graph_config["max_step_num"]
    
    return config


DEFAULT_MAS_CONFIG = load_mas_config()

