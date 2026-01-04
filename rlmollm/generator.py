"""
High-level API for RLMolLM molecular generation and optimization.
"""
import torch
import numpy as np
import json
import os
import sys
import random
import pandas as pd
from typing import Dict, List, Union, Optional
from rdkit import RDLogger

# Import from package
from rlmollm.population.population import Population
from rlmollm.utils.util import (
    parse_mutation_params,
    parse_train_flags,
    initialize_gan_operators,
    initialize_scaffold_handler,
    generate_initial_population_from_scaffold,
    generate_initial_population_without_scaffold,
    save_models,
)
from rlmollm.utils.training_utils import (
    train_and_evolve,
    setup_scoring_operator,
)


class RLMolLMGenerator:
    """
    High-level API for molecular generation using reinforcement learning-enhanced language models.
    
    This class provides a simple interface for optimizing molecules towards target properties
    using genetic algorithms with optional reinforcement learning (PPO).
    
    Example:
        >>> gen = RLMolLMGenerator(
        ...     checkpoint_path="model_weights/latest_model.pt",
        ...     config_path="config/default.json"
        ... )
        >>> molecules = gen.optimize(
        ...     target_properties={'qed': 1.0, 'logp': 2.5},
        ...     population_size=2000,
        ...     generations=20
        ... )
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
        seed: int = 42
    ):
        """
        Initialize the RLMolLMGenerator.
        
        Args:
            checkpoint_path: Path to model checkpoint file (.pt)
            config_path: Path to configuration JSON file
            device: Device to run on ("cuda" or "cpu")
            verbose: Whether to print progress information
            seed: Random seed for reproducibility
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.verbose = verbose
        self.seed = seed
        
        # Set random seeds for reproducibility
        self._set_random_seed(seed)
        
        # Suppress RDKit warnings if not verbose
        if not verbose:
            lg = RDLogger.logger()
            lg.setLevel(RDLogger.CRITICAL)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Use default config if none provided
            self.config = self._get_default_config()
        
        if self.verbose:
            print(f"✓ RLMolLMGenerator initialized")
            print(f"  Device: {self.device}")
            print(f"  Checkpoint: {self.checkpoint_path}")
            if config_path:
                print(f"  Config: {self.config_path}")
    
    def _set_random_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "gan_operators": [
                {
                    "model_directory": "ibm/MoLFormer-XL-both-10pct",
                    "checkpoint_path": self.checkpoint_path,
                    "num_steps": 1
                }
            ],
            "scoring_functions": [
                {"name": "qed", "weight": 1.0},
                {"name": "logp", "weight": 1.0},
                {"name": "sa", "weight": 1.0}
            ]
        }
    
    def optimize(
        self,
        target_properties: Dict[str, float],
        model_type: str = 'lm',
        initial_population_file: Optional[str] = None,
        population_size: int = 200,
        generations: int = 10,
        mutation_parameter: float = 0.4,
        use_scaffold: bool = False,
        scaffold_smiles: Optional[str] = None,
        mask_mode: str = "replace",
        output_dir: Optional[str] = None,
        return_dataframe: bool = False,
        batch_size: int = 32,
        top_k: int = 5,
        init_top_k: int = 5,
        lr: float = 0.00002,
        dataset_type: str = "guacamol",
        **kwargs
    ) -> Union[List[str], pd.DataFrame]:
        """
        Optimize molecules towards target properties using genetic algorithm.
        
        Args:
            target_properties: Dictionary of target properties and their values
                              (e.g., {'qed': 1.0, 'logp': 2.5, 'sa': 1.0})
            model_type: Model configuration type
                       Options: 'lm' (default), 'lm_ppo', 'alm', 'alm_ppo', 'lm_ng', 'lm_ng_ppo'
            initial_population_file: Path to initial population CSV file
                                    Required for non-scaffold MLM-based generation
            population_size: Size of the molecular population (default: 200)
            generations: Number of GA iterations (default: 10)
            mutation_parameter: Mutation rate, range 0.4-1.0 (default: 0.4)
            use_scaffold: Whether to use scaffold-based generation
            scaffold_smiles: Scaffold SMILES string (required if use_scaffold=True)
                           Use # as attachment points (e.g., "#c1cc(#)ccc1#")
            mask_mode: Masking strategy for generation
                      Options: 'replace', 'random', 'sample_partition'
            output_dir: Output directory for results (optional)
            return_dataframe: If True, return DataFrame with properties; if False, return list of SMILES
            batch_size: Batch size for training (default: 32)
            top_k: Number of top predictions for generator evaluation (default: 5)
            init_top_k: Number of top predictions for initial population (default: 5)
            lr: Learning rate (default: 0.00002)
            dataset_type: Dataset for empirical sampling ('moses', 'guacamol', 'zinc', 'gdb')
            **kwargs: Additional parameters passed to training
            
        Returns:
            List of SMILES strings or pandas DataFrame with SMILES and properties
        """
        if self.verbose:
            print(f"\n🧬 Starting optimization...")
            print(f"  Target properties: {target_properties}")
            print(f"  Model type: {model_type}")
            print(f"  Population size: {population_size}")
            print(f"  Generations: {generations}")
            print(f"  Mutation parameter: {mutation_parameter}")
            if use_scaffold:
                print(f"  Scaffold: {scaffold_smiles}")
        
        # Update config with target properties
        self._update_config_with_targets(target_properties)
        
        # Parse model type to get training flags
        generator_only, mlm_loss, use_ppo, no_merge = self._parse_model_type(model_type)
        
        # Set up mutation parameters
        mutation_samples_list = [population_size]
        mutation_parameter_list = [mutation_parameter]
        
        # Initialize GAN operators with checkpoint
        # Create a mock args object for compatibility
        class Args:
            pass
        
        args = Args()
        args.use_ppo = use_ppo
        args.ppo_epochs = 1
        args.ppo_interval = 1
        args.clip_ratio = 0.2
        args.entropy_coef = 0.01
        args.value_coef = 0.5
        args.reward_scale = 1.5 if use_ppo else 1.0
        args.invalid_penalty = -0.9 if use_ppo else -1.0
        args.use_optimized_ppo = True
        args.random_init = False
        args.mlm_loss = mlm_loss
        args.lr = lr
        args.epochs = generations
        args.batch_size = batch_size
        args.top_k = top_k
        args.init_top_k = init_top_k
        args.population_size = population_size
        args.mutation_samples = population_size
        args.generator_only = generator_only
        args.no_merge = no_merge
        args.add_randomized_smiles = False
        args.reset_all_smiles = False
        args.use_reinforce = False
        args.use_scaffold = use_scaffold
        args.mask_mode = mask_mode
        args.dataset_type = dataset_type
        args.output_directory = output_dir
        args.run_id = 'run'
        args.data_file = initial_population_file
        args.seed = self.seed
        
        # Override saved_generator path in config with checkpoint_path
        # This ensures we use the correct checkpoint file
        for operator in self.config.get('gan_operators', []):
            if 'saved_generator' in operator or self.checkpoint_path:
                operator['saved_generator'] = self.checkpoint_path
        
        # Initialize GAN operators
        gan_operators = initialize_gan_operators(
            self.config, 
            self.device, 
            mutation_parameter_list, 
            args
        )
        
        # Initialize scoring operator
        scoring_operator = setup_scoring_operator(self.config, self.device)
        
        # Initialize scaffold handler if needed
        scaffold_handler = None
        if use_scaffold:
            if scaffold_smiles is None:
                raise ValueError("scaffold_smiles must be provided when use_scaffold=True")
            if 'scaffold_operator' not in self.config:
                # Add scaffold config if not present
                self.config['scaffold_operator'] = {
                    "fixed_substructure": scaffold_smiles,
                    "mask_mode": mask_mode
                }
            scaffold_handler = initialize_scaffold_handler(self.config, gan_operators, self.device)
        
        # Handle initial population
        data_file = initial_population_file
        if data_file is None:
            if use_scaffold and scaffold_handler is not None:
                # Generate scaffold-based population
                if self.verbose:
                    print(f"  Generating initial population from scaffold...")
                data_file = generate_initial_population_from_scaffold(
                    scaffold_handler=scaffold_handler,
                    gan_operators=gan_operators,
                    args=args,
                    log_file=None
                )
            else:
                # Generate population without scaffold
                if self.verbose:
                    print(f"  Generating initial population...")
                
                # Extract configuration
                min_masks = self.config.get('no_scaffold_operator', {}).get('min_masks', 3)
                max_masks = self.config.get('no_scaffold_operator', {}).get('max_masks', 25)
                use_empirical = self.config.get('no_scaffold_operator', {}).get('use_empirical', False)
                dataset_name = dataset_type
                
                data_file = generate_initial_population_without_scaffold(
                    gan_operators=gan_operators,
                    population_size=population_size,
                    batch_size=10,
                    top_k=init_top_k,
                    log_file=None,
                    output_directory=output_dir or "./temp_output",
                    min_masks=min_masks,
                    max_masks=max_masks,
                    use_empirical=use_empirical,
                    dataset_name=dataset_name
                )
        
        # Initialize population
        population = Population(
            gan_operators,
            scoring_operator,
            use_scaffold=use_scaffold,
            scaffold_handler=scaffold_handler,
            mask_mode=mask_mode,
            use_optimized_ppo=True
        )
        
        # Load initial population
        if self.verbose:
            print(f"  Loading population from: {data_file}")
        
        population.read_population_dict_from_file(data_file, population_size)
        all_smiles = set(population.population_sequences)
        
        if self.verbose:
            print(f"  Initial population: {len(all_smiles)} molecules")
        
        # Note: Initial population with properties will be saved in train_and_evolve() at epoch 0
        
        if self.verbose:
            print(f"\n🔬 Running optimization for {generations} generations...")
        
        # Set up logging if output directory specified
        log_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_file = open(f"{output_dir}/run.log", 'w')
            print(f"Logging to {output_dir}/run.log", file=log_file, flush=True)
            print(args, file=log_file, flush=True)
        
        # Train and evolve
        population, all_smiles = train_and_evolve(
            population=population,
            gan_operators=gan_operators,
            args=args,
            train_flags=None,
            all_smiles=all_smiles,
            log_file=log_file,
            scaffold_handler=scaffold_handler,
            use_scaffold=use_scaffold
        )
        
        # Save results if output directory specified
        if output_dir:
            # Save population to CSV
            df = self._population_to_dataframe(population)
            df.to_csv(f"{output_dir}/final_population.csv", index=False)
            if log_file:
                log_file.close()
            if self.verbose:
                print(f"  Saved results to: {output_dir}")
                print(f"  Log file: {output_dir}/run.log")
        
        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"  Final population: {population.population_size} molecules")
        
        # Return results
        if return_dataframe:
            return self._population_to_dataframe(population)
        else:
            return population.population_sequences
    
    def _update_config_with_targets(self, target_properties: Dict[str, float]):
        """Update configuration with target properties."""
        from rlmollm.scoring.property_configs import get_all_properties
        
        # Get all available ADMET properties
        available_admet_properties = set(get_all_properties())
        
        # Properties that are RDKit-based (NOT ADMET-AI properties)
        rdkit_properties = {'synth', 'drug', 'logP', 'number', 'tpsa'}
        
        # Mapping of user-facing names to internal names
        property_name_mapping = {
            'qed': 'drug',
            'sa': 'synth',       # sa -> synth
            'sa_score': 'synth', # sa_score -> synth (alternative name)
            'logp': 'logP',      # lowercase logp -> capital P logP
        }
        
        # Get scoring operator config
        scoring_operator = self.config.get('scoring_operator', {})
        
        # Identify which target properties are ADMET vs non-ADMET
        admet_props_in_targets = []
        non_admet_props_in_targets = []
        
        for prop_name in target_properties.keys():
            internal_name = property_name_mapping.get(prop_name, prop_name)
            # Check if it's a RDKit property first, otherwise check if it's in ADMET configs
            if internal_name in rdkit_properties:
                non_admet_props_in_targets.append(internal_name)
            elif internal_name in available_admet_properties:
                admet_props_in_targets.append(internal_name)
            else:
                # Unknown property, treat as non-ADMET
                non_admet_props_in_targets.append(internal_name)
        
        # Update scoring_admet_names with new ADMET properties
        if admet_props_in_targets:
            current_admet_names = scoring_operator.get('scoring_admet_names', [])
            updated_admet_names = list(set(current_admet_names + admet_props_in_targets))
            scoring_operator['scoring_admet_names'] = updated_admet_names
            
            if self.verbose:
                print(f"  Updated ADMET properties: {updated_admet_names}")
        
        # REPLACE (not add to) scoring_names and selection_names with target properties
        # This ensures we ONLY optimize for the properties the user specified
        
        # Map all user-facing names to internal names for selection
        all_internal_names = []
        for prop_name in target_properties.keys():
            internal_name = property_name_mapping.get(prop_name, prop_name)
            all_internal_names.append(internal_name)
        
        # REPLACE scoring_names with ONLY the non-ADMET properties from targets
        # Keep 'number' as it's a validity constraint, not an optimization target
        updated_scoring_names = list(set(non_admet_props_in_targets + ['number']))
        scoring_operator['scoring_names'] = updated_scoring_names
        
        # REPLACE selection_names with ONLY the properties from targets (both ADMET and non-ADMET)
        updated_selection_names = list(set(all_internal_names))
        scoring_operator['selection_names'] = updated_selection_names
        
        self.config['scoring_operator'] = scoring_operator
        
        if self.verbose:
            print(f"  Updated scoring properties: {updated_scoring_names}")
            print(f"  Updated selection properties: {updated_selection_names}")
        
        # Update scoring functions in config
        if 'scoring_functions' not in self.config:
            self.config['scoring_functions'] = []
        
        # Clear existing scoring functions and add new ones based on targets
        self.config['scoring_functions'] = []
        for prop_name, target_value in target_properties.items():
            self.config['scoring_functions'].append({
                "name": prop_name,
                "weight": 1.0,
                "target": target_value
            })
    
    def _parse_model_type(self, model_type: str) -> tuple:
        """
        Parse model type string to get training flags.
        
        Returns:
            Tuple of (generator_only, mlm_loss, use_ppo, no_merge)
        """
        model_type = model_type.lower()
        
        generator_only = 'alm' in model_type
        mlm_loss = 'alm' in model_type
        use_ppo = 'ppo' in model_type
        no_merge = 'ng' in model_type
        
        return generator_only, mlm_loss, use_ppo, no_merge
    
    def _population_to_dataframe(self, population: Population) -> pd.DataFrame:
        """
        Convert population to pandas DataFrame with all properties.
        
        Args:
            population: Population object
            
        Returns:
            DataFrame with columns: smiles, properties, fitness_score
        """
        # Get the population dictionary which contains all data
        pop_dict = population.population_dict
        
        # Convert to DataFrame
        df = pd.DataFrame(pop_dict)
        
        # Reverse mapping: internal names -> user-facing names for better UX
        reverse_mapping = {
            'drug': 'qed',
            'drug_raw': 'qed_raw',
            'synth': 'sa',
            'synth_raw': 'sa_raw',
            'logP': 'logp',
            'logP_raw': 'logp_raw',
        }
        
        # Rename columns to user-facing names
        df = df.rename(columns=reverse_mapping)
        
        return df

