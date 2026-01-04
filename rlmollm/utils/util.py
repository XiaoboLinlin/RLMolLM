# util.py
import torch
import numpy as np
import re
import os
import csv
from datetime import datetime
import random
import sys

def setup_logging(args):
    """Set up logging for the training process.
    
    Args:
        args: Command line arguments
        
    Returns:
        log_file: File object for logging
    """
    log_file = None
    if args.output_directory is not None:
        os.makedirs(args.output_directory, exist_ok=True)
        
        # Use specified log file or create default one
        if args.log_file is not None:
            log_file_name = args.log_file
        else:
            log_file_name = f"{args.output_directory}/{args.run_id}.log"
        log_file = open(log_file_name, 'w')
        print(f"Logging to {log_file_name}")
    
    return log_file

def parse_mutation_params(mutation_samples_str, mutation_parameter_str):
    """Parse mutation parameters from comma-separated strings.
    
    Args:
        mutation_samples_str: String with comma-separated integers
        mutation_parameter_str: String with comma-separated floats
        
    Returns:
        Tuple of (mutation_samples_list, mutation_parameter_list)
    """
    mutation_samples_list = [int(x.strip()) for x in mutation_samples_str.split(',')]
    mutation_parameter_list = [float(x.strip()) for x in mutation_parameter_str.split(',')]
    return mutation_samples_list, mutation_parameter_list

def parse_train_flags(train_flags_str):
    """Parse train flags from comma-separated string.
    
    Args:
        train_flags_str: String with comma-separated integers (0 or 1)
        
    Returns:
        List of integers representing train flags
    """
    if train_flags_str is None:
        return None
    return [int(x) for x in train_flags_str.split(',')]

# def log_training_progress(epoch, total_epochs, children_valid, children_novel, 
#                           children_positive, children_accepted, train_disc_loss, 
#                           train_gen_loss, population_averages, log_file):
#     """Log training progress information.
    
#     Args:
#         Various training metrics
#         log_file: File object for logging
#     """
#     # Log time and metrics
#     print(f'[{epoch}/{total_epochs}] time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t'
#           f'valid: {children_valid}\t'
#           f'novel: {children_novel}\t'
#           f'positive: {children_positive}\t'
#           f'accepted: {children_accepted}\t'
#           f'dloss: {train_disc_loss}\t'
#           f'gloss: {train_gen_loss}', 
#           file=log_file, flush=True, end='\t')
    
#     # Log population averages
#     for key, value in population_averages.items():
#         print(f'{key}: {value:.4f}', file=log_file, end='\t')
#     print(file=log_file, flush=True)
def log_training_progress(epoch, total_epochs, children_valid, children_novel, 
                         children_positive, children_accepted, train_disc_loss, 
                         train_gen_loss, population_averages, log_file,
                         all_valid_count=None, all_count=None, children_unique=None):
    """Log training progress information.
    
    Args:
        children_valid: Number of valid molecules generated
        children_novel: Number of novel valid molecules (not in training set)
        children_unique: Number of unique valid molecules (no duplicates among valid ones)
        children_positive: Number of molecules with fitness > 0
        children_accepted: Number of molecules added to population
        
    Note: children_unique <= children_novel <= children_valid
    """
    # First line with main metrics
    main_metrics = (f'[{epoch}/{total_epochs}] time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t'
          f'valid: {children_valid}\t'
          f'novel: {children_novel}')
    
    # Add uniqueness if provided
    if children_unique is not None:
        main_metrics += f'\tunique: {children_unique}'
    
    main_metrics += (f'\tpositive: {children_positive}\t'
          f'accepted: {children_accepted}\t'
          f'dloss: {train_disc_loss}\t'
          f'gloss: {train_gen_loss}')
    
    print(main_metrics, file=log_file, flush=True)
    
    # Second line with all sequence validity statistics if provided
    if all_valid_count is not None and all_count is not None and all_count > 0:
        all_valid_rate = all_valid_count / all_count
        all_valid_stats = f'all_valid: {all_valid_count}\tall_count: {all_count}\tall_valid_rate: {all_valid_rate:.4f}'
        print(all_valid_stats, file=log_file, flush=True, end='\t')
    else:
        # Just add a tab to align with population averages
        print('\t', file=log_file, flush=True, end='')
    
    # Population averages on the same line as all_valid stats
    for key, value in population_averages.items():
        print(f'{key}: {value:.4f}', file=log_file, end='\t')
    print(file=log_file, flush=True)

def save_models(gan_operators, output_directory, run_id, log_file):
    """Save trained models to disk.
    
    Args:
        gan_operators: List of GAN operators
        output_directory: Directory to save models
        run_id: Identifier for the run
        log_file: File object for logging
    """
    # Check if this is a "best_model" or "model_epoch_1" or should be a "latest_model"
    prefix = run_id
    if not run_id.startswith("best_model") and not run_id.startswith("model_epoch"):
        # For models saved from training_combined.py, use latest_model prefix
        prefix = f"latest_model_{run_id}"
    
    for i, gan in enumerate(gan_operators):
        gen_filename = f"{output_directory}/{prefix}_generator_{i}.pt"
        if not gan.generator_only:
            disc_filename = f"{output_directory}/{prefix}_discriminator_{i}.pt"
            gan.save(gen_filename, disc_filename)
            print(f"Saved model weights to {gen_filename} and {disc_filename}", file=log_file, flush=True)
        else:
            # For generator-only mode
            torch.save(gan._gen.state_dict(), gen_filename)
            print(f"Saved generator weights to {gen_filename}", file=log_file, flush=True)
            
def setup_device_and_logging(args):
    """Set up device and logging.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (device, run_id, data_file, log_file)
    """
    # Use GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_id = args.run_id
    data_file = args.data_file
    
    # Set up logging
    log_file = None
    if args.output_directory is not None:
        os.makedirs(args.output_directory, exist_ok=True)
        
        # Use specified log file or create default one
        if args.log_file is not None:
            log_file_name = args.log_file
        else:
            log_file_name = f"{args.output_directory}/{args.run_id}.log"
        log_file = open(log_file_name, 'w')
        print(f"Logging to {log_file_name}")
    
    return device, run_id, data_file, log_file

def initialize_gan_operators(config, device, mutation_parameter_list, args):
    """Initialize GAN operators from config.
    
    Args:
        config: Configuration dictionary
        device: Device for tensor operations
        mutation_parameter_list: List of mutation parameters
        args: Command line arguments
        
    Returns:
        List of GAN operators
    """
    from rlmollm.models.gan import Gan
    import os
    
    gan_operators = []
    for i, operator_config in enumerate(config['gan_operators']):
        if device is not None and 'device' in operator_config:
            operator_config['device'] = device
            
        mutation_param = mutation_parameter_list[min(i, len(mutation_parameter_list)-1)]
        
        # Create a copy of operator_config to modify
        config_copy = operator_config.copy()
        
        # Check if model_directory points to a .pt file
        model_dir = config_copy.get('model_directory', '')
        if model_dir.endswith('.pt') and os.path.exists(model_dir):
            print(f"Detected .pt file in model_directory: {model_dir}")
            print("Using base model with saved generator weights...")
            
            # Use base model directory for model structure
            config_copy['model_directory'] = 'model_weights'
            # Set the .pt file as saved_generator
            config_copy['saved_generator'] = model_dir
            
        gan_operators.append(
            Gan(**config_copy,
                mutation_parameter=mutation_param, 
                lr=args.lr, 
                generator_only=args.generator_only, 
                top_k=args.top_k, 
                random_init=args.random_init)
        )
    
    return gan_operators

def initialize_scaffold_handler(config, gan_operators, device):
    """Initialize scaffold handler if scaffold config is present.
    
    Args:
        config: Configuration dictionary
        gan_operators: List of GAN operators
        device: Device for tensor operations
        
    Returns:
        ScaffoldHandler object or None
    """
    from rlmollm.scaffold.scaffold_handler import ScaffoldHandler
    
    if 'scaffold_operator' not in config:
        return None
        
    # Use tokenizer from first GAN operator
    tokenizer = gan_operators[0]._tokenizer
    
    return ScaffoldHandler(config['scaffold_operator'], tokenizer, device)

# def generate_initial_population_from_scaffold(scaffold_handler, gan_operators, args, log_file):
#     """Generate initial population from scaffold.
    
#     Args:
#         scaffold_handler: ScaffoldHandler object
#         gan_operators: List of GAN operators
#         args: Command line arguments
#         log_file: File object for logging
        
#     Returns:
#         Path to generated initial population file
#     """
#     print("Generating initial population from scaffold...", file=log_file, flush=True)
    
#     # Use first GAN operator for generation
#     gan = gan_operators[0]
    
#     # Generate molecules - use args.top_k instead of hardcoded 1
#     generated_smiles = scaffold_handler.generate_initial_population(
#         gan=gan,
#         population_size=args.population_size,
#         batch_size=args.batch_size,
#         top_k=args.init_top_k  # Use the top_k from arguments
#     )

#     # TODO: make sure generated_smiles and valid and unique
    
#     # Save to file
#     initial_population_file = f"{args.output_directory}/initial_population.csv"
#     with open(initial_population_file, 'w', newline='') as f:
#         writer = csv.writer(f, delimiter='\t')
#         writer.writerow(['smiles'])  # Header
#         for smiles in generated_smiles:
#             writer.writerow([smiles])
    
#     print(f"Generated {len(generated_smiles)} molecules and saved to {initial_population_file}", 
#           file=log_file, flush=True)
    
#     return initial_population_file

def validate_smiles(smiles_list, log_file=None):
    """Validate and filter SMILES to ensure they are valid and unique.
    
    Args:
        smiles_list (list): List of SMILES strings to validate
        log_file: Optional file object for logging
        
    Returns:
        list: Filtered list of valid and unique canonical SMILES
        dict: Statistics about validation results
    """
    from rdkit import Chem
    from rdkit import RDLogger
    
    # Suppress RDKit logging
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    if log_file:
        print(f"Validating {len(smiles_list)} generated SMILES...", file=log_file, flush=True)
    
    # Track valid molecules
    valid_smiles = []
    invalid_count = 0
    duplicate_count = 0
    unique_canonical_smiles = set()
    
    for smiles in smiles_list:
        # Skip empty strings
        if not smiles or len(smiles.strip()) == 0:
            invalid_count += 1
            continue
            
        # Convert to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Check if conversion was successful
        if mol is not None:
            # Get canonical SMILES to ensure uniqueness
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # Only add if we haven't seen this molecule before
            if canonical_smiles not in unique_canonical_smiles:
                unique_canonical_smiles.add(canonical_smiles)
                valid_smiles.append(canonical_smiles)  # Store canonical version
            else:
                duplicate_count += 1
        else:
            invalid_count += 1
    
    stats = {
        "total": len(smiles_list),
        "valid": len(valid_smiles),
        "invalid": invalid_count,
        "duplicates": duplicate_count
    }
    
    # Report results if logging is enabled
    if log_file:
        print(f"Validation results:", file=log_file, flush=True)
        print(f"  - Total generated: {stats['total']}", file=log_file, flush=True)
        print(f"  - Valid and unique: {stats['valid']}", file=log_file, flush=True)
        print(f"  - Invalid: {stats['invalid']}", file=log_file, flush=True)
        print(f"  - Duplicates: {stats['duplicates']}", file=log_file, flush=True)
    
    return valid_smiles, stats

def generate_initial_population_from_scaffold(scaffold_handler, gan_operators, args, log_file):
    """Generate initial population from scaffold.
    
    Args:
        scaffold_handler: ScaffoldHandler object
        gan_operators: List of GAN operators
        args: Command line arguments
        log_file: File object for logging
        
    Returns:
        Path to generated initial population file
    """
    print("Generating initial population from scaffold...", file=log_file, flush=True)
    
    # Use first GAN operator for generation
    gan = gan_operators[0]
    
    # Generate initial batch of molecules
    generated_smiles = scaffold_handler.generate_initial_population(
        gan=gan,
        population_size=args.population_size * 2,
        batch_size=args.batch_size,
        top_k=args.init_top_k
    )

    # Validate and filter generated SMILES
    valid_smiles, stats = validate_smiles(generated_smiles, log_file)
    
    # Generate more molecules if needed to reach target population size
    attempts = 0
    max_attempts = args.population_size
    
    while len(valid_smiles) < args.population_size and attempts < max_attempts:
        attempts += 1
        # additional_needed = args.population_size - len(valid_smiles)
        
        print(f"Attempt {attempts}/{max_attempts}", 
              file=log_file, flush=True)
        
        additional_smiles = scaffold_handler.generate_initial_population(
            gan=gan,
            population_size=args.population_size * 2,  # Generate extra to account for potential invalids
            batch_size=args.batch_size,
            top_k=args.init_top_k
        )
        
        # Validate and filter the additional molecules
        new_valid_smiles, new_stats = validate_smiles(additional_smiles, log_file)
        
        # Add new valid SMILES to our collection, avoiding duplicates
        unique_set = set(valid_smiles)
        for smiles in new_valid_smiles:
            if smiles not in unique_set:
                valid_smiles.append(smiles)
                unique_set.add(smiles)
        
        print(f"Now have {len(valid_smiles)}/{args.population_size} molecules", 
              file=log_file, flush=True)
    
    # Use actual valid SMILES as our population, up to the requested size
    final_smiles = valid_smiles[:args.population_size]
    
    # Save to file
    initial_population_file = f"{args.output_directory}/initial_population.csv"
    os.makedirs(os.path.dirname(initial_population_file), exist_ok=True)  # Create dir if doesn't exist
    with open(initial_population_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['smiles'])  # Header
        for smiles in final_smiles:
            writer.writerow([smiles])
    
    print(f"Generated {len(final_smiles)} valid, unique molecules and saved to {initial_population_file}", 
          file=log_file, flush=True)
    
    return initial_population_file

def generate_masked_molecules_no_scaffold(num_molecules, min_masks=3, max_masks=25, use_empirical=False, dataset_name="moses"):
    """Generate masked molecules without scaffold.
    
    Args:
        num_molecules: Number of molecules to generate
        min_masks: Minimum number of masks per molecule (used if use_empirical=False)
        max_masks: Maximum number of masks per molecule (used if use_empirical=False)
        use_empirical: If True, use empirical token distributions from real datasets
        dataset_name: Dataset to use for empirical sampling ('moses' or 'guacamol')
        
    Returns:
        List of masked molecules as strings
    """
    import random
    masked_molecules = set()
    max_attempts = num_molecules 
    attempts = 0
    
    # Try to import empirical sampling if requested
    sample_token_count = None
    if use_empirical:
        try:
            from rlmollm.utils.token_distributions import sample_token_count
        except ImportError:
            print("Warning: Could not import empirical token distributions, falling back to random.randint")
            use_empirical = False
    
    while len(masked_molecules) < num_molecules and attempts < max_attempts:
        attempts += 1
        
        # Choose number of masks
        if use_empirical and sample_token_count is not None:
            # Use empirical distribution from real datasets
            num_masks = sample_token_count(dataset_name)
        else:
            # Use uniform random distribution
            num_masks = random.randint(min_masks, max_masks)
        
        # Create a molecule that is just [MASK] tokens
        result = "[MASK]" * num_masks
        
        # Add to set if unique
        if result not in masked_molecules:
            masked_molecules.add(result)
    
    return list(masked_molecules)

def generate_initial_population_without_scaffold(gan_operators, population_size, batch_size=10, top_k=5, log_file=None, output_directory="./output", min_masks=3, max_masks=25, use_empirical=False, dataset_name="moses"):
    """Generate initial population without scaffold.
    
    Args:
        gan_operators: List of GAN operators
        population_size: Desired population size
        batch_size: Batch size for molecule generation (default: 10)
        top_k: Number of top predictions for molecule generation (default: 5)
        log_file: File to write log messages to
        output_directory: Directory to save the output file (default: ./output)
        min_masks: Minimum number of masks per molecule (default: 3)
        max_masks: Maximum number of masks per molecule (default: 25)
        use_empirical: If True, use empirical token distributions from real datasets (default: False)
        dataset_name: Dataset to use for empirical sampling ('moses', 'guacamol', or 'zinc', default: 'moses')
        
    Returns:
        Path to generated molecules file
    """
    import os
    import rdkit.Chem
    import torch
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Output file path - save directly in the output directory
    output_file = f"{output_directory}/initial_population.csv"
    
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"Initial population file already exists at {output_file}", file=log_file, flush=True)
        return output_file
    
    gan = gan_operators[0]  # Use the first GAN operator
    results = []
    
    print(f"Generating initial population without scaffold...", file=log_file, flush=True)
    print(f"Target population size: {population_size}, Batch size: {batch_size}, Top-k: {top_k}", file=log_file, flush=True)
    
    if use_empirical:
        print(f"Using empirical token distributions from {dataset_name} dataset", file=log_file, flush=True)
        try:
            from rlmollm.utils.token_distributions import DATASET_STATS
            token_range = f"{min(DATASET_STATS[dataset_name].keys())}-{max(DATASET_STATS[dataset_name].keys())}"
            avg_tokens = sum(k*v for k,v in DATASET_STATS[dataset_name].items())
            print(f"Empirical distribution: {token_range} tokens, average ~{avg_tokens:.1f}", file=log_file, flush=True)
        except ImportError:
            print("Warning: Could not load empirical distributions, falling back to random", file=log_file, flush=True)
    else:
        print(f"Using uniform random distribution: min_masks={min_masks}, max_masks={max_masks}", file=log_file, flush=True)
    
    # Use attempt-based generation
    max_attempts = population_size * 5  # Maximum number of attempts to generate molecules
    attempt = 0
    total_processed = 0
    total_valid = 0
    
    while len(results) < population_size and attempt < max_attempts:
        attempt += 1
        print(f"\nAttempt {attempt}/{max_attempts}", file=log_file, flush=True)
        print(f"Current progress: {len(results)}/{population_size} molecules ({len(results)/population_size*100:.1f}%)", file=log_file, flush=True)
        
        # Generate a batch of masked molecules
        masked_mols = generate_masked_molecules_no_scaffold(batch_size, min_masks, max_masks, use_empirical, dataset_name)
        print(f"Generated {len(masked_mols)} masked molecules for this batch", file=log_file, flush=True)
        
        # Tokenize batch
        batch = gan._tokenizer(masked_mols, padding=True, return_tensors='pt')
        batch_ids = batch['input_ids'].to(gan._device)
        batch_mask = batch['attention_mask'].to(gan._device)
        
        # Generate token probabilities
        with torch.no_grad():
            fake = gan._gen(input_ids=batch_ids, attention_mask=batch_mask, hard=False).detach().cpu()
            batch_ids = batch_ids.detach().cpu()
        
        # Process entire batch at once
        total_processed += batch_ids.size(0)
        
        # Process each sequence in batch
        batch_smiles = []
        for i in range(fake.size(0)):
            # Get masked sequence
            input_ids = batch_ids[i]
            
            # Find mask token positions
            masked_index = torch.nonzero(input_ids == gan._tokenizer.mask_token_id, as_tuple=False).flatten()
            if len(masked_index) == 0:
                continue
                
            # Get probabilities at mask positions
            probs = fake[i, masked_index, :]
            
            # Get top-k predictions for each mask
            values, predictions = probs.topk(top_k)
            
            # Try different combinations of top-k predictions
            possible_indices = torch.zeros(len(predictions), dtype=torch.long)
            for k in range(top_k):
                indices = None
                if k == 0:
                    # Take top predictions
                    indices = predictions[:,0]
                else:
                    # Find next best prediction
                    max_score = -1
                    best_index = -1
                    for j in range(len(predictions)):
                        current_indices = possible_indices.detach().clone()
                        current_indices[j] += 1
                        current_score = torch.prod(torch.gather(values, 1, current_indices.unsqueeze(1)))
                        if current_score > max_score:
                            max_score = current_score
                            best_index = j
                    
                    if best_index == -1:
                        break
                        
                    possible_indices[best_index] += 1
                    indices = torch.gather(predictions, 1, possible_indices.unsqueeze(1)).flatten()
                
                # Fill in masks with predictions
                new_ids = input_ids.clone()
                new_ids[masked_index] = indices
                smiles = gan._tokenizer.decode(new_ids, skip_special_tokens=True).replace(' ','').replace('##','')
                batch_smiles.append(smiles)
        
        # Validate SMILES
        valid_smiles = []
        for smiles in batch_smiles:
            mol = rdkit.Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
        
        # Add valid SMILES to results
        results.extend(valid_smiles)
        total_valid += len(valid_smiles)
        
        print(f"Batch results:", file=log_file, flush=True)
        print(f"  - Generated {len(batch_smiles)} SMILES from {len(masked_mols)} masked molecules", file=log_file, flush=True)
        print(f"  - Valid SMILES: {len(valid_smiles)} ({len(valid_smiles)/len(batch_smiles)*100:.1f}% validity)", file=log_file, flush=True)
        print(f"  - Running total: {len(results)}/{population_size} valid molecules", file=log_file, flush=True)
        
        # Stop if we've generated enough molecules
        if len(results) >= population_size:
            results = results[:population_size]  # Trim to exact population size
            break
    
    print(f"\nFinal statistics:", file=log_file, flush=True)
    print(f"  - Total processed: {total_processed}", file=log_file, flush=True)
    print(f"  - Total valid: {total_valid}", file=log_file, flush=True)
    print(f"  - Final population size: {len(results)}", file=log_file, flush=True)
    print(f"  - Overall validity rate: {total_valid/total_processed*100:.1f}%", file=log_file, flush=True)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write("smiles\n")  # Header
        for smiles in results:
            f.write(f"{smiles}\n")
    
    print(f"Saved initial population to {output_file}", file=log_file, flush=True)
    return output_file