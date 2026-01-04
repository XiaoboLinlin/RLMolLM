# training_utils.py
import torch
import numpy as np
import csv
import os
import glob
from tqdm import tqdm

def train_and_evolve(population, gan_operators, args, train_flags, all_smiles, log_file, scaffold_handler=None, use_scaffold=False):
    """Train models and evolve the population.
    
    Args:
        population: Population object
        gan_operators: List of GAN operators
        args: Command line arguments
        train_flags: List of flags for training
        all_smiles: Set of all seen SMILES
        log_file: File object for logging
        scaffold_handler: Optional ScaffoldHandler object
        
    Returns:
        Updated population and all_smiles set
    """
    from rlmollm.utils.util import log_training_progress, save_models
    
    # Variables to track best model
    best_fitness_mean = 0.0
    best_valid_rate = 0.0
    best_combined_score = 0.0  # Combined score: 0.5 * all_valid_rate + 0.5 * children_fitness_mean
    best_epoch = 0
    best_saved = False
    best_model_prefix = None
    
    # Prepare output files
    new_seq_output_file = None
    population_output_file = None
    if args.output_directory is not None:
        # Delete all existing .pt files in the output directory
        pt_files = glob.glob(f"{args.output_directory}/*.pt")
        for pt_file in pt_files:
            # try:
            os.remove(pt_file)
            # print(f"Deleted existing model file: {pt_file}", file=log_file, flush=True)
            # except Exception as e:
            #     print(f"Failed to delete {pt_file}: {str(e)}", file=log_file, flush=True)
        
        new_seq_output_file = open(f"{args.output_directory}/{args.run_id}_new_sequences.tsv", 'w')
        population.write_population_dict_header(new_seq_output_file, add_epoch=True)
        population_output_file = open(f"{args.output_directory}/{args.run_id}_population.tsv", 'w')
        population.write_population_dict_header(population_output_file)
    
    # Log initial conditions
    population_averages = population.get_population_averages()
    log_training_progress(
        epoch=0, 
        total_epochs=args.epochs,
        children_valid=0, 
        children_novel=0, 
        children_positive=0, 
        children_accepted=0,
        train_disc_loss='0.0000', 
        train_gen_loss='0.0000',
        population_averages=population_averages,
        log_file=log_file
    )
    
    # Save initial population with properties at epoch 0 for baseline comparison
    if args.output_directory is not None:
        import pandas as pd
        initial_pop_dict = population.population_dict
        initial_df = pd.DataFrame(initial_pop_dict)
        initial_output_file = f"{args.output_directory}/{args.run_id}_initial_population_properties.csv"
        initial_df.to_csv(initial_output_file, index=False)
        print(f"Saved initial population with properties to: {initial_output_file}", 
              file=log_file, flush=True)
    
    # Parse mutation parameters
    if isinstance(args.mutation_samples, str):
        mutation_samples_list = [int(x.strip()) for x in args.mutation_samples.split(',')]
    else:
        mutation_samples_list = [args.mutation_samples]
    
    # Run training and selection
    print(f"Starting training for {args.epochs} epochs...", file=log_file, flush=True)
    
    for i in range(args.epochs):
        # Initialize flags for this epoch
        ppo_was_run = False
        
        # Setup data loader with full population
        print(f"Training epoch {i+1} with {len(population.population_sequences):,} molecules", file=log_file, flush=True)
        print(f"Creating DataLoader with batch_size={args.batch_size}...", file=log_file, flush=True)
        
        train_loader = torch.utils.data.DataLoader(
            population.population_sequences,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to avoid issues
            pin_memory=False,  # Don't pin memory to avoid GPU memory issues
            drop_last=True  # Drop incomplete batches
        )
        
        print(f"DataLoader created, starting training...", file=log_file, flush=True)

        # Train GAN
        train_disc_loss = '0.0000'
        train_gen_loss = '0.0000'
        if (args.generator_only) or (args.mlm_loss):
            print(f"Starting MLM training for epoch {i+1}...", file=log_file, flush=True)
            train_disc_loss, train_gen_loss = population.train_gans(train_loader, train_flags, log_file=log_file, population_size=args.population_size)
            print(f"MLM training completed for epoch {i+1}, MLM loss: {train_gen_loss}", file=log_file, flush=True)


        if args.use_reinforce:
            # Train with REINFORCE using the same loader
            reinforce_loss, reinforce_reward, reinforce_valid_rate = population.train_reinforce(
                train_loader, 
                epochs=1,
                gamma=0.99
            )
            reinforce_loss = f"{reinforce_loss:.4f}"
            reinforce_reward = f"{reinforce_reward:.4f}"
            reinforce_valid_rate = f"{reinforce_valid_rate:.4f}"
            
        # Train with PPO if enabled
        if args.use_ppo:
            # Only run PPO every ppo_interval epochs
            if (i % args.ppo_interval) == 0:
                print(f"Running PPO training for epoch {i+1}...", file=log_file, flush=True)
                ppo_loss, ppo_reward, ppo_valid_rate = population.train_ppo(
                    dataloader=train_loader,
                    ppo_epochs=args.ppo_epochs,
                    clip_ratio=args.clip_ratio,
                    lr=args.lr,
                    entropy_coef=args.entropy_coef,
                    value_coef=args.value_coef,
                    reward_scale=args.reward_scale,
                    invalid_penalty=args.invalid_penalty,
                    batch_size=args.batch_size,
                    use_scaffold=use_scaffold,
                    scaffold_handler=scaffold_handler
                )
                # Store PPO metrics for later logging
                ppo_was_run = True
            else:
                ppo_was_run = False
            
        # Generate child population using generator
        child_population_dict, children_valid, (all_valid_count, all_count) = population.generate_child_population_dict(
            mutation_samples_list, 
            previous_set=all_smiles, 
            return_valid=True,
            add_randomized_smiles=args.add_randomized_smiles
        )

        children_novel = len(child_population_dict[population._scoring_operator.data_column_name])
        
        # Calculate uniqueness of generated children (unique AND valid)
        children_unique = None
        if children_novel > 0:
            # Get the valid SMILES from child population
            # Note: child_population_dict already contains only valid SMILES after filtering
            # in sequences_to_population_dict() -> prepare_data_for_scoring()
            child_smiles = child_population_dict[population._scoring_operator.data_column_name]
            unique_valid_smiles = set(child_smiles)
            children_unique = len(unique_valid_smiles)
            
            # Sanity check: unique should be <= novel
            assert children_unique <= children_novel, f"Unique ({children_unique}) should be <= Novel ({children_novel})"

        # Eliminate children with zero fitness
        children_fitness = child_population_dict[population._fitness_column_name]
        # Calculate fitness mean
        children_fitness_mean = 0.0
        if len(children_fitness) > 0:
            children_fitness_mean = np.mean(children_fitness)
        
        non_zero_indices = np.nonzero(children_fitness)[0]
        children_positive = children_novel
        if len(non_zero_indices) < len(children_fitness):
            for key in child_population_dict:
                child_population_dict[key] = [child_population_dict[key][x] for x in non_zero_indices]
            children_positive = len(non_zero_indices)

        # Merge population
        children_accepted = 0
        if not args.no_merge:
            children_accepted = population.merge_child_population_dict(child_population_dict)

        # Option for all smiles
        if args.reset_all_smiles:
            all_smiles = set(population.population_sequences)

        # Updated population averages
        population_averages = population.get_population_averages()
        

        log_training_progress(
            epoch=i+1, 
            total_epochs=args.epochs,
            children_valid=children_valid, 
            children_novel=children_novel, 
            children_positive=children_positive, 
            children_accepted=children_accepted,
            train_disc_loss=train_disc_loss, 
            train_gen_loss=train_gen_loss,
            population_averages=population_averages,
            all_valid_count=all_valid_count,
            all_count=all_count,
            children_unique=children_unique,
            log_file=log_file
        )
        
        # Log epoch completion
        print(f"Epoch {i+1}/{args.epochs} completed - MLM Loss: {train_gen_loss}", file=log_file, flush=True)

        # Log metrics that were calculated earlier
        if len(children_fitness) > 0:
            print(f"Children fitness mean: {children_fitness_mean:.4f}", file=log_file, flush=True)
            
        if args.use_reinforce:
            print(f"REINFORCE Training - Loss: {reinforce_loss}, Reward: {reinforce_reward}, Valid Rate: {reinforce_valid_rate}", 
                file=log_file, flush=True)
                
        if ppo_was_run:
            print(f"PPO metrics - Loss: {ppo_loss:.4f}, Reward: {ppo_reward:.4f}, Valid rate (top 1): {ppo_valid_rate:.4f}", 
                file=log_file, flush=True)

        # Save model after the first epoch
        if i == 0 and args.output_directory is not None:
            first_epoch_prefix = f"model_epoch_1"
            save_models(gan_operators, args.output_directory, first_epoch_prefix, log_file)
            print(f"Saved model after first epoch", file=log_file, flush=True)

        # Calculate validity rate for all sequences
        all_valid_rate = all_valid_count / max(1, all_count)
        
        # Check if this is the best model so far using combined score
        # Combined score: 0.5 * all_valid_rate + 0.5 * children_fitness_mean
        current_combined_score = 0.5 * all_valid_rate + 0.5 * children_fitness_mean
        is_best = False
        
        if current_combined_score > best_combined_score:
            best_combined_score = current_combined_score
            best_fitness_mean = children_fitness_mean
            best_valid_rate = all_valid_rate
            best_epoch = i
            is_best = True
        
        # Save best model if this is the best epoch
        if is_best and args.output_directory is not None:
            # Delete previous best model if it exists
            if best_model_prefix is not None:
                for j, gan in enumerate(gan_operators):
                    gen_filename = f"{args.output_directory}/{best_model_prefix}_generator_{j}.pt"
                    if os.path.exists(gen_filename):
                        os.remove(gen_filename)
                    
                    # Only check for discriminator if not in generator_only mode
                    if not gan.generator_only:
                        disc_filename = f"{args.output_directory}/{best_model_prefix}_discriminator_{j}.pt"
                        if os.path.exists(disc_filename):
                            os.remove(disc_filename)
            
            # Set the new best model prefix
            best_model_prefix = f"best_model_epoch_{i+1}"
            
            # Save new best model
            save_models(gan_operators, args.output_directory, best_model_prefix, log_file)
            best_saved = True
            print(f"Saved best model at epoch {i+1} with combined score: {best_combined_score:.4f} (valid_rate: {best_valid_rate:.4f}, fitness: {best_fitness_mean:.4f})", 
                  file=log_file, flush=True)


        # Write new sequences
        if new_seq_output_file is not None:
            population.write_population_dict_values(new_seq_output_file, child_population_dict, epoch=i+1)
    
    # Write final population to file
    if population_output_file is not None:
        population.write_population_dict_values(population_output_file)
        population_output_file.close()
    
    # Close output files
    if new_seq_output_file is not None:
        new_seq_output_file.close()
    
    # Print best epoch information at the end
    if best_saved:
        print(f"Best model was at epoch {best_epoch+1} with combined score: {best_combined_score:.4f} (valid_rate: {best_valid_rate:.4f}, fitness: {best_fitness_mean:.4f})", 
              file=log_file, flush=True)
    
    print("Training completed!", file=log_file, flush=True)
    
    return population, all_smiles

def setup_scoring_operator(config, device):
    """Initialize scoring operator from config.
    
    Args:
        config: Configuration dictionary
        device: Device for tensor operations
        
    Returns:
        Scoring operator object
    """
    from rlmollm.scoring.admet_scoring import ADMETScoring
    
    scoring_config = config['scoring_operator']
    if device is not None and 'device' in scoring_config.get('scoring_parameters', {}):
        scoring_config['scoring_parameters']['device'] = device
    
    return ADMETScoring(**scoring_config)