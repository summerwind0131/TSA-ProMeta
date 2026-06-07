import argparse

class Config:
    """Holds model and training configurations."""
    def __init__(self, args):
        self.query_size = 128
        self.batch_size = args.batch_size
        self.max_support_size = args.max_support_size
        
        # Learning Rates
        self.inner_lr = args.inner_lr
        self.outer_lr = args.outer_lr
        
        # Training loop
        self.inner_step = 5
        self.epochs = args.epochs
        self.patience = args.patience
        self.dropout_rate = args.dropout
        self.experiment_name = args.experiment_name
        
        # Architecture parameters
        self.hidden_dim = 128
        self.embed_dim = 64
        self.num_heads = 2
        self.num_layers = 2
        
        # Loss parameters
        self.focal_alpha = 0.75
        self.focal_gamma = 2.0
        self.l1_lambda = args.l1_lambda

        # Task-similarity aware ProMeta
        self.tsa_enable = args.tsa_enable
        self.num_task_groups = args.num_task_groups
        self.tsa_param_keys = [
            key.strip() for key in args.tsa_param_keys.split(",") if key.strip()
        ]
        self.tsa_selector_steps = args.tsa_selector_steps
        self.tsa_assignment_metric = args.tsa_assignment_metric
        self.tsa_warmup_checkpoint = args.tsa_warmup_checkpoint
        self.tsa_selector_source = args.tsa_selector_source
        self.tsa_assignment_source = args.tsa_assignment_source
        self.tsa_distance_mode = args.tsa_distance_mode
        self.tsa_gate_distance_weight = args.tsa_gate_distance_weight
        self.tsa_selector_l1_lambda = args.tsa_selector_l1_lambda
        self.tsa_routing_schedule = args.tsa_routing_schedule
        self.tsa_switch_threshold = args.tsa_switch_threshold
        self.tsa_min_group_fraction = args.tsa_min_group_fraction
        self.tsa_max_group_fraction = args.tsa_max_group_fraction
        if self.tsa_gate_distance_weight < 0:
            raise ValueError("--tsa_gate_distance_weight must be non-negative")
        if self.tsa_selector_l1_lambda < 0:
            raise ValueError("--tsa_selector_l1_lambda must be non-negative")
        if self.tsa_switch_threshold < 0:
            raise ValueError("--tsa_switch_threshold must be non-negative")
        if not 0 <= self.tsa_min_group_fraction <= 1:
            raise ValueError("--tsa_min_group_fraction must be between 0 and 1")
        if not 0 < self.tsa_max_group_fraction <= 1:
            raise ValueError("--tsa_max_group_fraction must be in (0, 1]")
        if self.tsa_min_group_fraction > self.tsa_max_group_fraction:
            raise ValueError("--tsa_min_group_fraction cannot exceed --tsa_max_group_fraction")
        if self.num_task_groups * self.tsa_min_group_fraction > 1:
            raise ValueError("num_task_groups * tsa_min_group_fraction cannot exceed 1")
        if self.num_task_groups * self.tsa_max_group_fraction < 1:
            raise ValueError("num_task_groups * tsa_max_group_fraction must be at least 1")

def parse_args():
    parser = argparse.ArgumentParser(description="ProMeta: Pathway-Gated Meta-Learning for Proteomics")
    
    # Path Arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing input pkl files")
    parser.add_argument("--proteomics_csv", type=str, required=True, help="Path to preprocessed proteomics CSV")
    parser.add_argument("--cpdb_path", type=str, required=True, help="Path to CPDB pathways .tab file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save models and logs")
    
    # Training Arguments
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU Device ID")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--support_size", type=int, default=32, help="Support set size (k-shot)")
    parser.add_argument("--max_support_size", type=int, default=32, help="Maximum support size reserved before query sampling")
    parser.add_argument("--batch_size", type=int, default=4, help="Meta-batch size (tasks per batch)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of meta-training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Validation-AUROC early-stopping patience; 0 disables it")
    parser.add_argument("--experiment_name", type=str, default="ProMeta", help="Experiment label saved in result JSON files")
    
    # Hyperparameters
    parser.add_argument("--outer_lr", type=float, default=1e-4, help="Outer loop learning rate")
    parser.add_argument("--inner_lr", type=float, default=0.005, help="Inner loop learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--l1_lambda", type=float, default=1e-3, help="L1 regularization coefficient for gate")

    # Task-similarity aware ProMeta arguments
    parser.add_argument("--tsa_enable", action="store_true", help="Enable task-similarity aware multi-initialization ProMeta")
    parser.add_argument("--num_task_groups", type=int, default=5, help="Number of TSA task groups / adaptive initializations")
    parser.add_argument("--tsa_param_keys", type=str, default="classifier,tokenizer.gate_logits", help="Comma-separated adaptive parameter name prefixes used for TSA clustering")
    parser.add_argument("--tsa_selector_steps", type=int, default=10, help="Support-only adaptation steps used to estimate task parameters for TSA assignment")
    parser.add_argument("--tsa_assignment_metric", type=str, default="l2", choices=["l2"], help="Distance metric used for TSA group assignment")
    parser.add_argument("--tsa_warmup_checkpoint", type=str, default="", help="Baseline ProMeta checkpoint used to initialize TSA clustering")
    parser.add_argument(
        "--tsa_selector_source",
        type=str,
        default="frozen_warmup",
        choices=["frozen_warmup", "live_model"],
        help="Parameter source used to estimate TSA task vectors",
    )
    parser.add_argument(
        "--tsa_assignment_source",
        type=str,
        default="current_group",
        choices=["current_group", "fixed_centroid"],
        help="Prototype source used to assign tasks to TSA groups",
    )
    parser.add_argument(
        "--tsa_distance_mode",
        type=str,
        default="block_mean_l2",
        choices=["block_mean_l2", "global_l2"],
        help="Distance aggregation used for TSA group assignment",
    )
    parser.add_argument(
        "--tsa_gate_distance_weight",
        type=float,
        default=1.0,
        help="Weight applied to the gate parameter block in block_mean_l2 distance",
    )
    parser.add_argument(
        "--tsa_selector_l1_lambda",
        type=float,
        default=1e-3,
        help="Gate L1 coefficient used only while estimating TSA task vectors",
    )
    parser.add_argument(
        "--tsa_routing_schedule",
        type=str,
        default="epoch_snapshot",
        choices=["epoch_snapshot", "online"],
        help="Freeze one assignment map per epoch or select groups inside each batch",
    )
    parser.add_argument(
        "--tsa_switch_threshold",
        type=float,
        default=0.05,
        help="Relative distance improvement required before a task may switch groups",
    )
    parser.add_argument(
        "--tsa_min_group_fraction",
        type=float,
        default=0.05,
        help="Minimum fraction of training tasks assigned to every group in epoch_snapshot mode",
    )
    parser.add_argument(
        "--tsa_max_group_fraction",
        type=float,
        default=0.50,
        help="Maximum fraction of training tasks assigned to one group in epoch_snapshot mode",
    )
    
    args = parser.parse_args()
    config = Config(args)
    return args, config
