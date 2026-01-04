from rlmollm.scoring.molecule_scoring import MoleculeScoring
import numpy as np
import rdkit
import rdkit.Chem
import torch
from rlmollm.models.rl import PPOTrainer
from rlmollm.models.rl.ppo_trainer_optimized import PPOTrainerOptimized
import torch.nn.functional as F
from tqdm import tqdm

def randomize_smiles(input_smiles, retries=5):
    """Generate a randmized version of an input smiles
    
    Args:
        input_smiles (str): smiles string representation of a molecule

    Returns:
        str with a randomized smiles for the input molecule
    """
    for _ in range(retries):
        try:
            mol = rdkit.Chem.MolFromSmiles(input_smiles)
            atom_order = list(range(mol.GetNumAtoms()))
            np.random.shuffle(atom_order)
            new_mol = rdkit.Chem.RenumberAtoms(mol, atom_order)
            randomized_smiles = rdkit.Chem.MolToSmiles(new_mol, canonical=False)
            if randomized_smiles != input_smiles:
                return randomized_smiles
        except:
            continue

    return None

class Population():
    """Class to store a population of sequences and apply mutation/recombination/selection."""

    def __init__(self, gan_operators, scoring_operator, use_scaffold=False, scaffold_handler=None, mask_mode="replace", use_optimized_ppo=False):
        """Constructor for Population class.

        Args:
            gan_operators (List[gan]): List of gans used for generating mutations
            scoring_operator (ScoringInterface): Scoring operator to score population sequences
            use_scaffold (bool): Whether to use scaffold constraints
            scaffold_handler: ScaffoldHandler object
            mask_mode (str): Mode for masking tokens: "replace" or "sample_partition"
            use_optimized_ppo (bool): Whether to use optimized PPO implementation (default: True)
        """
        super().__init__()

        # store evolution_operators and scoring_operator
        self._gan_operators = gan_operators
        self._scoring_operator = scoring_operator
        self.use_scaffold = use_scaffold
        self.scaffold_handler = scaffold_handler
        self.mask_mode = mask_mode
        self.use_optimized_ppo = use_optimized_ppo

        # intialize population dict
        self._population_dict = {}
        self._data_column_name = self._scoring_operator.data_column_name
        self._fitness_column_name = self._scoring_operator.fitness_column_name
        self._column_names = self._scoring_operator.column_names
        
        # Initialize PPO trainer if needed
        self._ppo_trainer = None

    @property
    def population_dict(self):
        """Get population dictionary.

        Returns:
            Dict[str,] with data and scores for population

        """
        return self._population_dict

    @property
    def population_size(self):
        """ Get population size.

        Returns;
            int with population size
        """
        if len(self._population_dict) > 0:
            return len(self._population_dict[self._data_column_name])
        else:
            return 0

    @property
    def population_sequences(self):
        """Get population sequences.
        
        Returns:
            List[str] with values for data column name key in population dict

        """
        return self._population_dict[self._data_column_name]

    def read_population_dict_from_file(self, population_file, population_size=1000, delimiter='\t'):
        """Read a population file and store contents in population_dict

        Args:
            population_file (str): Path to population file (should be a delimited file with header row with column names)
            population_size (int): Number of data rows in the population.
            delimiter (str): Delimiter for parsing population file

        """
        # create a dictionary to store column names and order from file
        header_dict = {}
        reverse_header_dict = {}

        # clear population_dict
        self._population_dict = {}

        # store sequences that must be scored
        sequences_to_score = []

        # flag to generate metrics if not provided
        generate_metrics = False

        # First pass: count total lines for progress bar
        print(f"Counting molecules in {population_file}...")
        with open(population_file, 'r') as input_file:
            total_lines = sum(1 for _ in input_file) - 1  # Exclude header
        
        # Calculate how many molecules to read initially (1.5x target to account for filtering)
        initial_read_size = int(population_size * 1.5) if population_size > 0 else -1
        
        if population_size > 0 and total_lines > initial_read_size:
            progress_total = initial_read_size
            print(f"Will load {initial_read_size} molecules initially (target: {population_size} after filtering)")
        else:
            progress_total = total_lines
            print(f"Loading all {total_lines} molecules")
        
        # Second pass: read with progress bar
        with open(population_file, 'r') as input_file:
            row_counter = -1
            # Disable tqdm when output is redirected (e.g., to log files)
            import sys
            show_progress = sys.stdout.isatty()
            pbar = tqdm(total=progress_total, desc="Loading molecules", unit="molecules", disable=not show_progress)
            
            for row in input_file:
                row_counter += 1
                # read header row
                if row_counter == 0:
                    header_names = [x.strip() for x in row.split(delimiter)]
                    column_counter = 0
                    for name in header_names:
                        header_dict[name] = column_counter
                        column_counter += 1

                    # construct reverse map
                    for key in header_dict:
                        reverse_header_dict[header_dict[key]] = key

                    # make sure that file has data column
                    if self._data_column_name not in header_dict:
                        raise ValueError('Error: %s does not have %s as a header column' % (population_file, self._data_column_name))
                
                    # check that all metrics are provided
                    required_column_names = self._scoring_operator.column_names
                    for column_name in required_column_names:
                        self._population_dict[column_name] = []
                        if column_name not in header_dict:
                            generate_metrics = True

                    continue

                # non-header rows
                row_split = [x.strip() for x in row.split(delimiter)]

                if not generate_metrics:
                    for i in range(len(row_split)):
                        if reverse_header_dict[i] != self._data_column_name:
                            self._population_dict[reverse_header_dict[i]].append(float(row_split[i]))
                        else:
                            self._population_dict[reverse_header_dict[i]].append(row_split[i])
                else:
                    sequences_to_score.append(row_split[header_dict[self._data_column_name]])

                # Update progress bar
                pbar.update(1)

                # stop reading if initial_read_size is met
                if initial_read_size >= 0 and row_counter >= initial_read_size:
                    break
            
            pbar.close()
            print(f"Successfully loaded {row_counter} molecules from dataset")

        # check if scoring is needed
        if len(sequences_to_score) > 0:
            self._population_dict = self.sequences_to_population_dict(sequences_to_score)

        # adjust population to desired size (trim if too many, expand if too few)
        # Skip adjustment if population_size is negative (means use all molecules)
        current_size = len(self._population_dict[self._data_column_name])
        if population_size > 0 and current_size != population_size:
            self._fill_population_dict(population_size)

    def generate_child_population_dict(self, mutation_samples, weighted=False, previous_set=None, db_dict=None, batch_size=10, return_valid=False, add_randomized_smiles=False):
        """Generate a child population dict from current population dict

        Args:
            mutation_samples (List[int]): Number of mutation samples for each gan object
            weight (bool): Option to weight sampling for mutation
            previous_set (set[str]): Set of previously visited sequences
            db_dict (Dict[str,]): Dictionary with keys cursor and query_string
            batch_size (int): Batch size for generating mutations and recombinations
            return_valid (bool): Option to return number of valid molecules generated

        Returns:
            Dict[str,] with child population dict

        """
        # check that mutation samples are valid
        if len(mutation_samples) != len(self._gan_operators):
            raise ValueError('Error length of mutation samples is not equal to length of evolution operators')

        # setup defaults
        if previous_set is None:
            previous_set = set()

        if db_dict is None:
            db_dict = {}

        # store generated sequences from mutations
        possible_sequences = []
        possible_sequences_all = []
        masked_sequences = []

        # mutation samples
        samples = None
        for i in range(len(mutation_samples)):
            samples = self._sample_population_dict(mutation_samples[i], weighted)
            for j in range(0, mutation_samples[i], batch_size):
                start_index = j
                end_index = min(start_index + batch_size, mutation_samples[i])
                total_batch = samples[start_index:end_index].tolist()

                # add randomized smiles if requested
                if add_randomized_smiles:
                    for example in samples[start_index:end_index]:
                        r_smiles = randomize_smiles(example)
                        if r_smiles is not None:
                            total_batch.append(r_smiles)      

                generated_sequences, generated_sequences_all, m_sequences = self._gan_operators[i].evaluate_generator(total_batch, self.use_scaffold, self.scaffold_handler, self.mask_mode)
                possible_sequences += generated_sequences
                possible_sequences_all += generated_sequences_all
                masked_sequences += m_sequences
        # Calculate validity for all sequences if requested
        all_valid_count = 0
        all_count = len(possible_sequences_all)

        if return_valid:
            for sequence in possible_sequences_all:
                prepared_data = self._scoring_operator.prepare_data_for_scoring(sequence)
                if prepared_data is not None:
                    all_valid_count += 1

        child_population_dict = None
        child_valid = -1

        if return_valid:
            child_population_dict, child_valid = self.sequences_to_population_dict(possible_sequences, previous_set, db_dict, return_valid)
            return child_population_dict, child_valid, (all_valid_count, all_count)
        else:
            child_population_dict = self.sequences_to_population_dict(possible_sequences, previous_set, db_dict, return_valid)
            return child_population_dict

        # # generate dictionary from possible sequences
        # return self.sequences_to_population_dict(possible_sequences, previous_set, db_dict, return_valid)


    def _normalize_population_dict_lengths(self):
        """Ensure all keys in population_dict have the same length.
        
        If lengths are inconsistent, trim all keys to the length of _data_column_name.
        """
        if not self._population_dict:
            return
        
        # Get the reference length from data column
        reference_length = len(self._population_dict.get(self._data_column_name, []))
        
        # Check and fix any inconsistent lengths
        for key in self._column_names:
            if key in self._population_dict:
                current_length = len(self._population_dict[key])
                if current_length != reference_length:
                    if current_length > reference_length:
                        # Trim to reference length
                        self._population_dict[key] = self._population_dict[key][:reference_length]
                    else:
                        # This should not happen - data column should be the shortest
                        raise ValueError(f"Key '{key}' has length {current_length} < reference length {reference_length}")

    def merge_child_population_dict(self, child_population_dict, max_size=-1):
        """Merge child population dict with current population dict.

        Args:
            child_population_dict (Dict[str,]): Dictionary for child population
            max_size (int): allows population to grow to max_size, otherwise merged population maintains previous size

        Returns:
            int number of child population merged into original population

        """
        # Normalize population dict before merge to fix any length inconsistencies from previous operations
        self._normalize_population_dict_lengths()
        
        # save population size before merge
        original_population_size = self.population_size

        # append to current population
        for key in self._column_names:
            self._population_dict[key] += child_population_dict[key]
        
        # Normalize after merge in case there were any inconsistencies introduced
        self._normalize_population_dict_lengths()

        # allow population growth if max_size is set
        cutoff_size = original_population_size
        if max_size > 0:
            cutoff_size = min(max_size, len(self._population_dict[self._data_column_name]))

        # population size is maintained by the merge
        selection_index = None
        if len(self._scoring_operator.selection_names) < 1:
            # After merge, select from the current (merged) population size
            current_population_size = len(self._population_dict[self._data_column_name])
            selection_index = np.random.choice(current_population_size, cutoff_size, replace=False)
        else:
            selection_index = np.argsort(-1.0*np.array(self._population_dict[self._fitness_column_name]))[:cutoff_size]

        # apply selection
        for key in self._column_names:
            self._population_dict[key] = [self._population_dict[key][x] for x in selection_index]

        # count children accepted
        children_accepted = np.sum(selection_index >= original_population_size)

        return children_accepted

    def write_population_dict_header(self, output_file, add_epoch=False):
        """Write header for population dict

        Args:
            output_file (file object): Write enabled file object

        """
        if not add_epoch:
            output_file.write('\t'.join(self._column_names) + '\n')
        else:
            output_file.write('\t'.join(self._column_names + ['epoch']) + '\n')

    def write_population_dict_values(self, output_file, population_dict=None, epoch=None):
        """Write values for population dict

        Args:
            output_file (file object): Write enabled file object
            population_dict (Dict[str,]): Dictionary used for output

        """
        # default is to write current population
        population_dict_to_write = self._population_dict if population_dict is None else population_dict
        population_size = len(population_dict_to_write[self._data_column_name])
        for i in range(population_size):
            row_data = []
            for key in self._column_names:
                if key == self._data_column_name:
                    row_data.append(population_dict_to_write[key][i])
                else:
                    row_data.append('%.6f' % (population_dict_to_write[key][i]))

            # option to write epoch to output file
            if epoch is not None:
                row_data.append(str(epoch))

            output_file.write('\t'.join(row_data) + '\n')

    def get_population_averages(self):
        """Get average of scoring metrics for population
        
        Returns:
            Dict[str, float] with averages for population metrics

        """
        averages_dict = {}
        for key in self._column_names:
            if key != self._data_column_name:
                averages_dict[key] = np.mean(self._population_dict[key])

        return averages_dict

    def _sample_population_dict(self, number_of_samples, weighted):
        """Return sample from data column of population dict

        Args:
            number_of_samples (int): Number of samples to draw
            weighted (bool): Option to weight samples by softmax of fitness

        Returns:
            List[str] with sampled sequences from data column of population dict

        """
        if weighted:
            # softmax weights from fitness
            weights = np.exp(self._population_dict[self._fitness_column_name])
            weights /= np.sum(weights)
            return np.random.choice(self._population_dict[self._data_column_name], number_of_samples, p=weights)
        else:
            return np.random.choice(self._population_dict[self._data_column_name], number_of_samples)

    def sequences_to_population_dict(self, sequences, previous_set=None, db_dict=None, return_valid=False):
        """Generation a population_dict from a list of sequences

        Args:
            sequences (List[str]): List of sequences for population
            previous_set (set[str]): Set of previously visited sequences
            db_dict (Dict[str,]): Dictionary with keys cursor and query_string
            return_valid (bool): Option to return count of valid molecules

        Returns:
            Dict[str,] produced by the scoring operator

        """
        # setup defaults
        if previous_set is None:
            previous_set = set()

        if db_dict is None:
            db_dict = {}

        sequences_to_keep = []

        # valid sequences produced
        valid_counter = 0

        for sequence in sequences:

            # check if sequence is viable
            prepared_data = self._scoring_operator.prepare_data_for_scoring(sequence)
            if prepared_data is not None:

                valid_counter += 1

                # attempt to make canonical - for cases like molecules generation where cleaned_data and sequence don't have same type
                canonical_data = self._scoring_operator.make_canonical(prepared_data)

                # check if data has already been recorded
                if canonical_data in previous_set:
                    continue

                # check if data has already been recorded in db
                if ('cursor' in db_dict) and ('query_string' in db_dict):
                    canonical_query = (canonical_data,)
                    cursor = db_dict['cursor']
                    cursor.execute(db_dict['query_string'], canonical_query)
                    if cursor.fetchone()[0] == 1:
                        continue

                # add to population
                sequences_to_keep.append(prepared_data)
                previous_set.add(canonical_data)

        # generate population
        if return_valid:
            return self._scoring_operator.generate_scores(sequences_to_keep), valid_counter
        else:
            return self._scoring_operator.generate_scores(sequences_to_keep)

    def train_gans(self, train_loader, train_flags=None, log_file=None, population_size=None):
        """Train GANs associated with the population.
        
        Args:
            train_loader (torch.utils.data.DataLoader): ataloader to iterate through dataset
            log_file: Optional file object for logging progress
            population_size: Population size used for training (-1 means full dataset)

        Returns:
            two str with comma separated discriminator and generator loss
        
        """
        train_disc_loss = []
        train_gen_loss = []
        counter = 0
        for gan in self._gan_operators:
            flag = True
            if train_flags is not None:
                flag = (train_flags[counter] == 1)

            if flag:
                _, disc_loss, gen_loss = gan.train_epoch(train_loader, log_file=log_file, population_size=population_size)
                train_disc_loss.append(disc_loss)
                train_gen_loss.append(gen_loss)
            else:
                train_disc_loss.append(0.0)
                train_gen_loss.append(0.0)

            counter += 1

        return ','.join(['%0.4f' % x for x in train_disc_loss]), ','.join(['%0.4f' % x for x in train_gen_loss])

    def _fill_population_dict(self, desired_size):
        """Adjust population_dict to a desired size by trimming or making random copies.

        Args:
            desired_size (int): Desired number of elements for each key in the population_dict

        """
        # Skip adjustment if desired_size is negative (means use all molecules)
        if desired_size <= 0:
            print(f"Population adjustment skipped (desired_size={desired_size}, using all molecules)")
            return
            
        current_size = len(self._population_dict[self._data_column_name])
        
        if current_size > desired_size:
            # Trim down to desired size by randomly selecting molecules
            selected_indices = np.random.choice(current_size, desired_size, replace=False)
            # Use _column_names to ensure consistent handling of all columns
            for key in self._column_names:
                self._population_dict[key] = [self._population_dict[key][i] for i in selected_indices]
            print(f"Trimmed population from {current_size} to {desired_size} molecules")
            
        elif current_size < desired_size:
            # Fill up to desired size by making random copies
            copy_indices = np.random.choice(current_size, desired_size - current_size)
            for index in copy_indices:
                # Use _column_names to ensure consistent handling of all columns
                for key in self._column_names:
                    self._population_dict[key].append(self._population_dict[key][index])
            print(f"Expanded population from {current_size} to {desired_size} molecules")

    # ## reinforce
    # def train_reinforce(self, dataloader, epochs=1, gamma=0.99):
    #     """Train GANs with REINFORCE algorithm to optimize molecular properties.
        
    #     Args:
    #         dataloader (torch.utils.data.DataLoader): Dataloader to iterate through dataset
    #         epochs (int): Number of epochs for training
    #         gamma (float): Discount factor for rewards
            
    #     Returns:
    #         train_loss (float): Average training loss
    #     """
    #     # Initialize for tracking
    #     total_loss = 0.0
    #     batch_counter = 0
    #     rewards = []
    #     valid_molecules = 0
    #     total_molecules = 0
        
    #     # For each GAN operator
    #     for gan in self._gan_operators:
    #         gan._gen.train()
    #         optimizer = torch.optim.Adam(gan._gen.parameters(), lr=0.0001)
            
    #         # Train for multiple epochs
    #         for _ in range(epochs):
    #             # Iterate through dataloader
    #             for batch in dataloader:
    #                 # Zero gradients
    #                 optimizer.zero_grad()
                    
    #                 # Generate molecules
    #                 batch_molecules = []
    #                 batch_log_probs = []
                    
    #                 # Process each molecule in batch
    #                 for smiles in batch:
    #                     # Tokenize
    #                     tokens = gan._tokenizer(smiles, padding=True, return_tensors='pt')
    #                     input_ids = tokens['input_ids'].to(gan._device)
    #                     attention_mask = tokens['attention_mask'].to(gan._device)
                        
    #                     # Apply masking
    #                     # task = np.random.choice(['replace', 'insert', 'delete'])
    #                     task = 'replace'
    #                     masked_ids, masked_mask, _ = gan.generate_masks(input_ids.cpu(), attention_mask.cpu(), task)
    #                     masked_ids = masked_ids.to(gan._device)
    #                     masked_mask = masked_mask.to(gan._device)
    #                     # Generate probabilities
    #                     logits = gan._gen(input_ids=masked_ids, attention_mask=masked_mask, hard=False, raw=True)
                        
    #                     # Find masked positions
    #                     masked_index = torch.nonzero(masked_ids[0] == gan._tokenizer.mask_token_id, as_tuple=False).flatten()
                        
    #                     if len(masked_index) == 0:
    #                         continue
                        
    #                     # Sample actions and calculate log probs
    #                     new_mol_ids = masked_ids.clone()
    #                     mol_log_probs = []
                        
    #                     for idx in masked_index:
    #                         # Get probabilities at masked position
    #                         probs = torch.nn.functional.softmax(logits[0, idx], dim=-1)
    #                         m = torch.distributions.Categorical(probs)
                            
    #                         # Sample token
    #                         token = m.sample()
    #                         log_prob = m.log_prob(token)
                            
    #                         # Save
    #                         mol_log_probs.append(log_prob)
    #                         new_mol_ids[0, idx] = token
                        
    #                     # Convert to SMILES
    #                     new_smiles = gan._tokenizer.decode(new_mol_ids[0], skip_special_tokens=True).replace(' ','').replace('##','')
                        
    #                     # Add to batch
    #                     batch_molecules.append(new_smiles)
    #                     if mol_log_probs:
    #                         batch_log_probs.append(torch.stack(mol_log_probs).mean())  # Average log prob per molecule
                    
    #                 # Calculate rewards
    #                 batch_rewards = []
    #                 for smiles in batch_molecules:
    #                     mol = self._scoring_operator.prepare_data_for_scoring(smiles)
    #                     if mol is not None:
    #                         # Score valid molecule
    #                         scores = self._scoring_operator.generate_scores([mol])
    #                         reward = scores[self._fitness_column_name][0]
    #                         valid_molecules += 1
    #                     else:
    #                         # Invalid molecule penalty
    #                         reward = -0.5 #-0.1
                        
    #                     total_molecules += 1
    #                     batch_rewards.append(reward)
    #                     rewards.append(reward)
                    
    #                 # Skip if no valid actions
    #                 if not batch_log_probs:
    #                     continue
                    
    #                 # Convert to tensors
    #                 batch_rewards = torch.tensor(batch_rewards, device=gan._device, dtype=torch.float32)
    #                 batch_log_probs = torch.stack(batch_log_probs)
                    
    #                 # Normalize rewards for stability
    #                 if len(batch_rewards) > 1:
    #                     batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-8)
                    
    #                 # Calculate REINFORCE loss
    #                 loss = -(batch_log_probs * batch_rewards).mean()
                    
    #                 # Backprop
    #                 loss.backward()
    #                 optimizer.step()
                    
    #                 # Track
    #                 total_loss += loss.item()
    #                 batch_counter += 1
        
    #     # Return statistics
    #     avg_loss = total_loss / max(1, batch_counter)
    #     avg_reward = sum(rewards) / max(1, len(rewards))
    #     valid_rate = valid_molecules / max(1, total_molecules)
        
    #     return avg_loss, avg_reward, valid_rate 
        ## reinforce

    ## PPO
    def train_ppo(self, dataloader, ppo_epochs=4, clip_ratio=0.2, lr=0.00005, 
            entropy_coef=0.01, value_coef=0.5, reward_scale=1.0, 
            invalid_penalty=-0.1, batch_size=32, use_scaffold=False, scaffold_handler=None):
        """Train GANs with PPO algorithm to optimize molecular properties.
        
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to iterate through dataset
            ppo_epochs (int): Number of PPO epochs per batch
            clip_ratio (float): PPO clipping parameter
            lr (float): Learning rate
            entropy_coef (float): Coefficient for entropy term in loss
            value_coef (float): Coefficient for value function loss
            reward_scale (float): Scaling factor for rewards
            invalid_penalty (float): Penalty for invalid molecules
            batch_size (int): Batch size for training
            use_scaffold (bool): Whether to use scaffold constraints
            scaffold_handler: ScaffoldHandler object
            
        Returns:
            tuple with training metrics (ppo_loss, avg_reward, valid_rate)
        """
        # Initialize PPO trainer if not already created
        if self._ppo_trainer is None:
            device = self._gan_operators[0]._device
            if self.use_optimized_ppo:
                print(f"🚀 Using optimized PPO implementation for faster training", flush=True)
                self._ppo_trainer = PPOTrainerOptimized(
                    gan_operators=self._gan_operators,
                    scoring_operator=self._scoring_operator,
                    device=device,
                    lr=lr
                )
            else:
                print(f"Using original PPO implementation", flush=True)
                self._ppo_trainer = PPOTrainer(
                    gan_operators=self._gan_operators,
                    scoring_operator=self._scoring_operator,
                    device=device,
                    lr=lr
                )
        
        # Delegate to PPO trainer
        return self._ppo_trainer.train_ppo(
            dataloader=dataloader,
            ppo_epochs=ppo_epochs,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            reward_scale=reward_scale,
            invalid_penalty=invalid_penalty,
            batch_size=batch_size,
            use_scaffold=use_scaffold,
            scaffold_handler=scaffold_handler,
            mask_mode=self.mask_mode
        )
    ## PPO