import torch
import torch.optim as optim
from rlmollm.models.generator import Generator
from rlmollm.models.discriminator import Discriminator
from copy import deepcopy
import numpy as np
import time
from tqdm import tqdm
import transformers
import json
from token_splits import pretokenizer_dict
import rdkit.Chem as Chem
import random
import itertools
import rlmollm.utils.util as util

class Gan:
    """Class for training and evaluation of generator and discriminator models."""


    def __init__(self, model_directory, tokenizer_directory, tokenizer_type='bert', mutation_parameter=0.5, lr=0.00001, device="cpu", saved_generator=None, saved_discriminator=None, generator_only=False, top_k=5, random_init=False):
        """Constructor for Gan class.
        
        Args:
            model_directory (str): Directory to be used to initialize models using hugging face
            tokenizer (hugging face tokenizer): Tokenizer determines conversion of text to token ids 
            mutation_parameter (float): probability of a token being replaced by a mask for input to generator
            lr (float): learning rate for AdamW optimizer
            device (str): device for training
            saved_generator (pytorch model): weights to initialize generator
            saved_discriminator (pytorch model): weights to initialize discriminator
        """
        super().__init__()

        # initialize class data
        self._device = torch.device(device)
        self._lr = lr
        self._mutation_parameter = mutation_parameter
        self._top_k = top_k

        # initialize tokenizer
        self._tokenizer = None
        try:
            with open(tokenizer_directory + '/config.json', 'r') as f:
                tokenizer_config = json.load(f)
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_directory, **tokenizer_config)
        except:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_directory, use_auth_token=True)
        self._tokenizer.backend_tokenizer.pre_tokenizer = pretokenizer_dict[tokenizer_type]
        
        # allows case with only generator for evaluation
        self._gen = Generator(model_directory, self._tokenizer, random_init).to(self._device)
        self._disc = None
        self._optimizer_disc = None
        self._optimizer_gen = None
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._optimizer_gen = optim.AdamW(self._gen.parameters(), lr=self._lr)
        if not generator_only:
            self._disc = Discriminator(model_directory, random_init).to(self._device)
            self._optimizer_disc = optim.AdamW(self._disc.parameters(), lr=self._lr)

        if saved_generator != None:
            self._gen.load_state_dict(torch.load(saved_generator))

        # note: this will fail if it conflicts with generator_only
        if saved_discriminator != None:
            self._disc.load_state_dict(torch.load(saved_discriminator))

    @property
    def generator_only(self):
        """Get value for generator only property.

        Returns:
            bool that determines whether gan is generator only

        """
        return (self._disc is None)

    def train_step(self, smiles_batch):
        """Perform training step based on a batch of smiles.
        
        Args:
            batch (List[str]): List of smiles strings for molecules

        Returns:
            Tuple with discriminator loss and generator loss
        """
        # tokenize smiles
        batch = self._tokenizer(smiles_batch, padding=True, return_tensors='pt')

        # set models to training mode
        self._gen.train()

        # used to fill input for loss calculation
        real_label = 1.
        fake_label = 0. 

        # track losses
        metric_disc_loss = 0.0
        metric_gen_loss = 0.0

        # update D
        if self._disc is not None:
            self._disc.train()
            self._disc.zero_grad()

            # determine loss on disc from real data
            batch_ids = batch['input_ids'].to(self._device)
            batch_mask = batch['attention_mask'].to(self._device)
            output = self._disc(input_ids=batch_ids, one_hot_tokens=None, attention_mask=batch_mask).view(-1)
            label = torch.full((batch_ids.shape[0],), real_label, dtype=torch.float, device=self._device)
            err_disc_real = self._criterion(output, label)
            err_disc_real.backward()

            # determine loss on disc from fake data
            # task = np.random.choice(['replace','insert','delete','combine'])
            batch_ids, batch_mask, _ = self.generate_masks(batch['input_ids'], batch['attention_mask'], 'replace')

            batch_ids = batch_ids.to(self._device)
            batch_mask = batch_mask.to(self._device)
            fake = self._gen(input_ids=batch_ids, attention_mask=batch_mask)
            label.fill_(fake_label)
            output = self._disc(input_ids=None, one_hot_tokens=fake.detach(), attention_mask=batch_mask).view(-1)
            err_disc_fake = self._criterion(output, label)
            err_disc_fake.backward()

            # update disc
            self._optimizer_disc.step()
            metric_disc_loss = (1.0 + err_disc_fake.item() + err_disc_real.item()) / 2.0

            # determine loss on generator
            self._gen.zero_grad()
            label.fill_(real_label)
            output = self._disc(input_ids=None, one_hot_tokens=fake, attention_mask=batch_mask).view(-1)
            err_gen = self._criterion(output, label)
            err_gen.backward()
            self._optimizer_gen.step()       

            metric_gen_loss = 1.0 + err_gen.item()

        else:
            self._gen.zero_grad()
            mlm_loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            batch_ids, batch_mask, batch_labels = self.generate_masks(batch['input_ids'], batch['attention_mask'], 'replace')
            batch_ids = batch_ids.to(self._device)
            batch_mask = batch_mask.to(self._device)
            batch_labels = batch_labels.to(self._device)
            fake = self._gen(input_ids=batch_ids, attention_mask=batch_mask, hard=False, raw=True)
            masked_lm_loss = mlm_loss_fct(fake.view(-1, self._gen.embedding.config.vocab_size), batch_labels.view(-1))
            masked_lm_loss.backward()
            self._optimizer_gen.step()  
            metric_gen_loss = masked_lm_loss.item()

        return (metric_disc_loss, metric_gen_loss)

    def train_epoch(self, dataloader, log_file=None, population_size=None):
        """Perform training step based on a batch of smiles.
        
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to iterate through dataset
            log_file: Optional file object for logging progress
            population_size: Population size used for training (-1 means full dataset)

        Returns:
            Tuple with time, discriminator loss, and generator loss
        """
        # initial values
        t = time.time()
        metric_disc_loss = 0.0
        metric_gen_loss = 0.0 
        batch_counter = 0

        # iterate through dataloader and train with progress tracking
        total_batches = len(dataloader)
        # Disable tqdm output when logging to file to avoid control characters in logs
        show_progress = log_file is None
        batch_pbar = tqdm(dataloader, desc="MLM Training", leave=False, unit="batch", disable=not show_progress)
        
        for batch_idx, batch in enumerate(batch_pbar):
            (batch_disc_loss, batch_gen_loss) = self.train_step(batch)
            metric_disc_loss += batch_disc_loss
            metric_gen_loss += batch_gen_loss
            batch_counter += 1
            
            # Update progress bar with current MLM loss (terminal only)
            batch_pbar.set_postfix({
                'MLM_Loss': f"{batch_gen_loss:.4f}"
            })
            
            # Log progress every 1% of batches to log file (only for full dataset training)
            if log_file and population_size == -1:
                log_interval = min(100, max(1, total_batches // 100))
                if batch_idx % log_interval == 0:
                    progress_pct = (batch_idx + 1) / total_batches * 100
                    print(f"  Batch {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%) - MLM Loss: {batch_gen_loss:.4f}", 
                          file=log_file, flush=True)

        return (time.time() - t), metric_disc_loss / batch_counter, metric_gen_loss / batch_counter

    # def generate_masks(self, batch_ids, batch_mask, task='replace', max_num_token=35):
    #     """Randomly mask token ids for use in generation

    #     Args:
    #         batch_ids (tensor): token ids for molecules sequences
    #         batch_mask (tensor): attention mask for molecule sequences
    #         task (str): replace or sample_partition
    #         max_num_token (int): maximum number of tokens allowed (including special tokens)

    #     Returns:
    #         tensor with randomly masksed token ids
    #         tensor with updated attention mask
    #     """

    #     # make copies of original data
    #     masked_ids = deepcopy(batch_ids)
    #     updated_attention_mask = deepcopy(batch_mask)
        
    #     # Process each sequence in the batch
    #     final_sequences = []
    #     sequence_lengths = []

    #     for i in range(len(masked_ids)):
    #         # each sequence has CLS and SEP tokens at beginning and end
    #         number_of_tokens = torch.count_nonzero(masked_ids[i]) - 2
            
    #         # Print debug info
    #         # print(f"Sequence {i}: total tokens: {torch.count_nonzero(masked_ids[i])}, non-special tokens: {number_of_tokens}")
    #         # print(f"Tokenizer mask token ID: {self._tokenizer.mask_token_id}")
    #         # print(f"Original sequence: {masked_ids[i]}")
            
    #         if number_of_tokens <= 0:
    #             # Raise an error for empty sequences
    #             raise ValueError(f"Empty sequence found at index {i}. Sequence must contain at least one token besides CLS and SEP.")

    #         if task == 'sample_partition':
    #             # Sample partition masking - more efficient implementation
    #             import itertools
                
    #             # Extract non-padding tokens and work with meaningful sequence only
    #             seq_length = torch.count_nonzero(masked_ids[i]).item()
    #             actual_sequence = masked_ids[i][:seq_length].clone()
                
    #             # Choose mask positions and apply them
    #             number_of_mutations = max(1, np.random.binomial(number_of_tokens, self._mutation_parameter))
    #             mask_positions = np.random.choice(
    #                 np.arange(1, number_of_tokens+1), 
    #                 min(number_of_mutations, number_of_tokens),
    #                 replace=False
    #             )
    #             for pos in mask_positions:
    #                 actual_sequence[pos] = self._tokenizer.mask_token_id
    #                 # Also update the original masked_ids tensor
    #                 masked_ids[i][pos] = self._tokenizer.mask_token_id
                
    #             # Find and group adjacent mask positions
    #             mask_indices = torch.nonzero(actual_sequence == self._tokenizer.mask_token_id, as_tuple=True)[0].tolist()
               
                
    #             mask_groups = [list(g) for _, g in itertools.groupby(mask_indices, lambda x, c=itertools.count(): x-next(c))]
                
    #             if not mask_groups:
    #                 raise ValueError(f"No mask tokens found in sequence {i} after applying mutations")
                
    #             # Create compressed sequence with one mask per group
    #             compressed_sequence = []
    #             last_pos = 0
                
    #             for group in mask_groups:
    #                 # Add tokens before this group
    #                 if group[0] > last_pos:
    #                     compressed_sequence.extend(actual_sequence[last_pos:group[0]].tolist())
                    
    #                 # Add single mask for this group
    #                 compressed_sequence.append(self._tokenizer.mask_token_id)
                    
    #                 # Update position
    #                 last_pos = group[-1] + 1
                
    #             # Add remaining tokens after the last group
    #             if last_pos < len(actual_sequence):
    #                 compressed_sequence.extend(actual_sequence[last_pos:].tolist())
                
    #             # Find mask positions in compressed sequence
    #             compressed_mask_indices = [i for i, token in enumerate(compressed_sequence) 
    #                                       if token == self._tokenizer.mask_token_id]
                
    #             # Calculate available space for extra masks
    #             available_slots = max(0, max_num_token - len(compressed_sequence))
    #             extra_masks = min(available_slots, random.randint(0, available_slots))
                
    #             # If we can add extra masks, do so
    #             if extra_masks > 0:
    #                 # Distribute extra masks among positions
    #                 extra_masks_distribution = [0] * len(compressed_mask_indices)
    #                 for _ in range(extra_masks):
    #                     idx = random.randint(0, len(compressed_mask_indices) - 1)
    #                     extra_masks_distribution[idx] += 1
                    
    #                 # Build final sequence with extra masks
    #                 final_sequence = []
    #                 last_pos = 0
                    
    #                 for idx, pos in enumerate(compressed_mask_indices):
    #                     # Add tokens before this mask
    #                     final_sequence.extend(compressed_sequence[last_pos:pos+1])
                        
    #                     # Add extra masks
    #                     final_sequence.extend([self._tokenizer.mask_token_id] * extra_masks_distribution[idx])
                        
    #                     last_pos = pos + 1
                    
    #                 # Add remaining tokens
    #                 if last_pos < len(compressed_sequence):
    #                     final_sequence.extend(compressed_sequence[last_pos:])
                    
    #                 final_sequences.append(final_sequence)
    #             else:
    #                 # No extra masks to add, use compressed sequence as is
    #                 final_sequences.append(compressed_sequence)
                
    #             sequence_lengths.append(len(final_sequences[-1]))

    #         else:  # Default to 'replace'
    #             # binomial distribution based off mutation_parameter with a minimum of 1 mask
    #             number_of_mutations = np.random.binomial(number_of_tokens, self._mutation_parameter)
    #             number_of_mutations = max(1, number_of_mutations)
    #             mutation_locations = np.random.choice(np.arange(1, number_of_tokens+1), number_of_mutations, replace=False)

    #             # apply mutations
    #             for location in mutation_locations:
    #                 masked_ids[i][location] = self._tokenizer.mask_token_id
                
    #             # Extract only the non-padding tokens
    #             seq_length = torch.count_nonzero(masked_ids[i]).item()
    #             actual_sequence = masked_ids[i][:seq_length].clone().tolist()
                
    #             # Store the sequence (already-masked sequence without padding)
    #             final_sequences.append(actual_sequence)
    #             sequence_lengths.append(seq_length)
        
    #     # Now process all sequences together
    #     max_length = max(sequence_lengths)
        
    #     # Create new tensors with the correct size
    #     new_masked_ids = torch.zeros((len(masked_ids), max_length), dtype=batch_ids.dtype, device=batch_ids.device)
    #     new_attention_mask = torch.zeros((len(masked_ids), max_length), dtype=batch_mask.dtype, device=batch_mask.device)
        
    #     # Fill new tensors with the processed sequences
    #     for i in range(len(final_sequences)):
    #         seq = final_sequences[i]
    #         seq_len = len(seq)
            
    #         # Convert to tensor if it's a list
    #         if isinstance(seq, list):
    #             seq_tensor = torch.tensor(seq, dtype=batch_ids.dtype, device=batch_ids.device)
    #         else:
    #             seq_tensor = seq
                
    #         # Copy sequence data
    #         new_masked_ids[i, :seq_len] = seq_tensor
            
    #         # Set attention mask
    #         new_attention_mask[i, :seq_len] = 1
        
    #     # Create labels based on masks, handling different sizes efficiently
    #     labels = torch.full_like(new_masked_ids, -100)  # Initialize all labels as -100 (ignore)
        
    #     # Calculate the overlapping size to avoid dimension errors
    #     overlap_width = min(batch_ids.size(1), new_masked_ids.size(1))
        
    #     # Create mask for positions that have mask tokens and are within the overlapping region
    #     mask_positions = (new_masked_ids[:, :overlap_width] == self._tokenizer.mask_token_id)
        
    #     # Only set labels for positions that have masks and are within the overlap
    #     labels[:, :overlap_width] = torch.where(mask_positions, batch_ids[:, :overlap_width], labels[:, :overlap_width])
        
    #     return new_masked_ids, new_attention_mask, labels


    def generate_masks(self, batch_ids, batch_mask, task='replace'):
        """Randomly mask token ids for use in generation

        Args:
            batch_ids (tensor): token ids for molecules sequences
            batch_mask (tensor): attention mask for molecule sequences
            task (str): replace, insert, or delete

        Returns:
            tensor with randomly masksed token ids
            tensor with updated attention mask
        """

        # make copies of original data
        masked_ids = deepcopy(batch_ids)
        updated_attention_mask = deepcopy(batch_mask)
        masked_ids_cols = len(masked_ids[0])

        # padding to allow larger molecules through recombination or insertion
        if task == "combine":
            if masked_ids_cols < self._tokenizer.model_max_length:
                updated_masked_ids_cols = min(self._tokenizer.model_max_length, 2*masked_ids_cols-1)
                masked_ids = torch.nn.functional.pad(masked_ids, (0,updated_masked_ids_cols-masked_ids_cols), 'constant', 0)
                updated_attention_mask = torch.nn.functional.pad(updated_attention_mask, (0,updated_masked_ids_cols-masked_ids_cols), 'constant', 0)
                masked_ids_cols = updated_masked_ids_cols
        elif task == "insert":
            if masked_ids_cols < self._tokenizer.model_max_length:
                masked_ids = torch.nn.functional.pad(masked_ids, (0,1), 'constant', 0)
                updated_attention_mask = torch.nn.functional.pad(updated_attention_mask, (0,1), 'constant', 0)
                masked_ids_cols = masked_ids_cols + 1

        # sets for insert/delete tasks
        insert_set = set()
        delete_set = set()

        for i in range(len(masked_ids)):

            # for combine task, sample another molecules
            if task == "combine":

                # determine lenghts for parents
                parent_length = torch.count_nonzero(masked_ids[i]).item()
                second_parent_index = np.random.choice(len(masked_ids))
                second_parent_length = torch.count_nonzero(masked_ids[second_parent_index]).item()

                # check if either parent is empty
                if (parent_length > 2) and (second_parent_length > 2) and (masked_ids_cols > 4):
                    end_index = np.random.choice(np.arange(2,parent_length))
                    end_index = min(masked_ids_cols-3, end_index)
            
                    start_index = np.random.choice(np.arange(1,second_parent_length-1))
                    start_index = max(end_index+1+second_parent_length-masked_ids_cols, start_index)
                    updated_length = second_parent_length - start_index

                    # overwrite masked_ids with combination
                    temp_ids = torch.zeros_like(masked_ids[i])
                    temp_ids[:end_index] = masked_ids[i,:end_index]
                    temp_ids[end_index] = self._tokenizer.mask_token_id
                    temp_ids[end_index+1:end_index+updated_length+1] = masked_ids[second_parent_index,start_index:second_parent_length]
                    masked_ids[i] = temp_ids

                    updated_attention_mask[i,:] = 0
                    updated_attention_mask[i,:end_index+updated_length+1] = 1

            # each sequence has CLS and SEP tokens at beginning and end
            number_of_tokens = torch.count_nonzero(masked_ids[i]) - 2

            if number_of_tokens == 0:
                # corner case for empty inputs
                if len(masked_ids[i]) > 2:
                    masked_ids[i][0] = self._tokenizer.cls_token_id
                    masked_ids[i][1] = self._tokenizer.mask_token_id
                    masked_ids[i][2] = self._tokenizer.sep_token_id
                    updated_attention_mask[i][:3] = 1
            else:
                # binomial distribution based off mutation_parameter with a minumum of 1 mask
                number_of_mutations = np.random.binomial(number_of_tokens, self._mutation_parameter)
                number_of_mutations = max(1, number_of_mutations)
                mutation_locations = set(np.random.choice(np.arange(1,number_of_tokens+1), number_of_mutations, replace=False))

                if task == 'insert':
                    if (number_of_tokens + 2) < len(masked_ids[i]):
                        insert_set.add(mutation_locations.pop())
                elif task == 'delete':
                    if number_of_tokens > 1:
                        selected_delete_location = mutation_locations.pop()
                        if selected_delete_location < number_of_tokens:
                            delete_set.add(selected_delete_location + 1)
                        mutation_locations.add(selected_delete_location)

                # apply mutations
                for location in mutation_locations:
                    masked_ids[i][location] = self._tokenizer.mask_token_id

                # apply insertion and deletion if specified
                if len(insert_set) > 0:
                    insert_location = insert_set.pop()
                    updated_attention_mask[i, number_of_tokens+2] = 1
                    temp_ids = torch.zeros_like(masked_ids[i])
                    temp_ids[:insert_location] = masked_ids[i,:insert_location]
                    temp_ids[insert_location] = self._tokenizer.mask_token_id
                    temp_ids[insert_location+1:] = masked_ids[i,insert_location:-1]
                    masked_ids[i] = temp_ids
                elif len(delete_set) > 0:
                    delete_location = delete_set.pop()
                    updated_attention_mask[i, number_of_tokens+1] = 0
                    temp_ids = torch.zeros_like(masked_ids[i])
                    temp_ids[:delete_location] = masked_ids[i,:delete_location]
                    temp_ids[delete_location:-1] = masked_ids[i,delete_location+1:]
                    masked_ids[i] = temp_ids          

        # labels based on masks
        labels = None
        if task == 'replace':
            labels = torch.where(masked_ids == self._tokenizer.mask_token_id, batch_ids, -100)

        return masked_ids, updated_attention_mask, labels

    
    def generate_scaffold_masks(self, batch_ids, batch_mask, use_scaffold, scaffold_handler, task='replace'):
        """Generate masks that respect scaffold structure.
        
        Args:
            batch_ids: Token IDs for molecules
            batch_mask: Attention mask for molecules
            scaffold_handler: ScaffoldHandler containing scaffold template
            
        Returns:
            Tuple of (masked_ids, updated_attention_mask, labels)
        """
        from copy import deepcopy
        import numpy as np
        import rdkit.Chem as Chem
        
        # Create new tensors to store results
        all_masked_ids = []
        all_updated_masks = []
        all_labels = []
        
        # Process each molecule separately
        for i in range(len(batch_ids)):
            # Create single-molecule tensors
            single_ids = batch_ids[i:i+1]
            single_mask = batch_mask[i:i+1]
            
            # Decode to SMILES for scaffold identification
            smiles = self._tokenizer.decode(single_ids[0], skip_special_tokens=True).replace(' ','').replace('##','')
            
            # Create a copy for masking
            modified_ids = single_ids.clone()
            
            # Get the fixed substructure (scaffold) from handler
            fixed_substr = scaffold_handler.fixed_substructure
            
            # Use RDKit to identify scaffold in molecule
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                raise ValueError(f"Invalid molecule SMILES: {smiles}")
            
            # Create a molecule from the scaffold template (replace # with dummy atoms)
            smarts_pattern = fixed_substr.replace('#', '*')  # Use * as attachment point
            pattern = Chem.MolFromSmarts(smarts_pattern)
            # scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
            
            if pattern is None:
                raise ValueError(f"Invalid scaffold SMILES: {smarts_pattern}")

            # Get matches for the scaffold substructure
            matches = mol.GetSubstructMatches(pattern)
            
            if not matches:
                raise ValueError(f"not matched scaffold SMILES: {smarts_pattern}")
            
            # Get the atom indices that should NOT be masked (part of scaffold)
            scaffold_tokens = set(atom_idx for match in matches for atom_idx in match)
            
            if task == 'replace':
                # Standard replacement masking
                tokens = modified_ids[0].tolist()
                for j in range(1, len(tokens) - 1):  # Skip CLS and SEP tokens
                    if tokens[j] != 0 and j not in scaffold_tokens:
                        if np.random.random() < self._mutation_parameter:
                            modified_ids[0][j] = self._tokenizer.mask_token_id

            # # Apply masking, avoiding scaffold tokens
            # tokens = modified_ids[0].tolist()
            # for j in range(1, len(tokens) - 1):  # Skip CLS and SEP tokens
            #     if tokens[j] != 0 and j not in scaffold_tokens:
            #         if np.random.random() < self._mutation_parameter:
            #             modified_ids[0][j] = self._tokenizer.mask_token_id
            
            # Append results
            all_masked_ids.append(modified_ids[0])
            all_updated_masks.append(single_mask[0])
        
        # Create labels for masked language modeling
        for i in range(len(all_masked_ids)):
            # Create labels for masked language modeling
            label = torch.where(all_masked_ids[i] == self._tokenizer.mask_token_id, batch_ids[i], torch.tensor(-100))
            all_labels.append(label)
        
        # Now we need to pad all tensors to the same length
        max_length = max(ids.size(0) for ids in all_masked_ids)
        
        # Pad each tensor to the maximum length
        padded_ids = []
        padded_masks = []
        padded_labels = []
        
        for i in range(len(all_masked_ids)):
            ids = all_masked_ids[i]
            mask = all_updated_masks[i]
            label = all_labels[i]
            
            # Calculate padding needed
            pad_length = max_length - ids.size(0)
            
            if pad_length > 0:
                # Pad the tensors
                ids = torch.nn.functional.pad(ids, (0, pad_length), 'constant', 0)
                mask = torch.nn.functional.pad(mask, (0, pad_length), 'constant', 0)
                label = torch.nn.functional.pad(label, (0, pad_length), 'constant', -100)
            
            padded_ids.append(ids)
            padded_masks.append(mask)
            padded_labels.append(label)
        
        # Stack the padded tensors
        return torch.stack(padded_ids), torch.stack(padded_masks), torch.stack(padded_labels)
    
    def generate_masks_per_molecule(self, batch_ids, batch_mask, task='replace', use_scaffold=False, scaffold_handler=None):
        """Generate masks with different tasks for each molecule in the batch.
        
        Args:
            batch_ids (tensor): token ids for molecules sequences
            batch_mask (tensor): attention mask for molecule sequences
            task (str or None): If provided, use this task for all molecules.
                            If None, randomly select a task for each molecule.
        
        Returns:
            tensor with randomly masked token ids
            tensor with updated attention mask
            tensor with labels for masked tokens
        """

        """Generate masks with different tasks for each molecule in the batch."""
        # If scaffolds are enabled and handler is provided, use scaffold-aware masking
        if use_scaffold and scaffold_handler is not None:
            return self.generate_scaffold_masks(batch_ids, batch_mask, use_scaffold, scaffold_handler, task)
        
        # Create new tensors to store results
        all_masked_ids = []
        all_updated_masks = []
        all_labels = []
        
        # Process each molecule separately
        for i in range(len(batch_ids)):
            # Choose a task for this molecule
            molecule_task = task
            if molecule_task is None or molecule_task == 'random':
                molecule_task = np.random.choice(['replace', 'insert', 'delete'])
            
            # Create single-molecule tensors
            single_ids = batch_ids[i:i+1]
            single_mask = batch_mask[i:i+1]
            
            # Apply the standard generate_masks to just this molecule
            modified_ids, modified_mask, molecule_labels = self.generate_masks(
                single_ids, single_mask, molecule_task
            )
            
            # Append the results
            all_masked_ids.append(modified_ids[0])
            all_updated_masks.append(modified_mask[0])
            if molecule_labels is not None:
                all_labels.append(molecule_labels[0])
            else:
                # Create a default label tensor of the right size
                default_label = torch.ones_like(modified_ids[0]) * -100
                all_labels.append(default_label)
        
        # Now we need to pad all tensors to the same length
        max_length = max(ids.size(0) for ids in all_masked_ids)
        
        # Pad each tensor to the maximum length
        padded_ids = []
        padded_masks = []
        padded_labels = []
        
        for i in range(len(all_masked_ids)):
            ids = all_masked_ids[i]
            mask = all_updated_masks[i]
            label = all_labels[i]
            
            # Calculate padding needed
            pad_length = max_length - ids.size(0)
            
            if pad_length > 0:
                # Pad the tensors
                ids = torch.nn.functional.pad(ids, (0, pad_length), 'constant', 0)
                mask = torch.nn.functional.pad(mask, (0, pad_length), 'constant', 0)
                label = torch.nn.functional.pad(label, (0, pad_length), 'constant', -100)
            
            padded_ids.append(ids)
            padded_masks.append(mask)
            padded_labels.append(label)
        
        # Stack the padded tensors
        return torch.stack(padded_ids), torch.stack(padded_masks), torch.stack(padded_labels)
    
    def _generate_scaffold_distributed_masks(self, smiles_batch, scaffold_handler):
        """Generate masks using scaffold_handler's partition sampling method.
        
        Args:
            smiles_batch (List[str]): List of SMILES strings
            scaffold_handler: ScaffoldHandler object with distribution method
            
        Returns:
            Tuple of (batch_ids, batch_mask) with masked tokens
        """
        # Generate masked molecules using scaffold_handler's method
        masked_molecules = scaffold_handler.generate_masked_molecules(len(smiles_batch))
        
        # Tokenize the masked molecules
        tokenized = self._tokenizer(masked_molecules, padding=True, return_tensors='pt')
        
        return tokenized['input_ids'], tokenized['attention_mask']

    def evaluate_generator(self, smiles_batch, use_scaffold=False, scaffold_handler=None, mask_mode="replace"):
        """Generate text sequences from a batch of smiles.
        
        Args:
            smiles_batch (List[str]): List of smiles strings for molecules
            use_scaffold (bool): Whether to enforce scaffold constraints
            scaffold_handler: ScaffoldHandler object for scaffold validation
            mask_mode (str): Masking mode - "replace", "sample_partition", or "pure_random_mask"

        Returns:
            List[str] with generated molecules
        """
        import rdkit.Chem as Chem
        
        self._gen.eval()
        masked_sequences = []
        
        # Set up scaffold validation if enabled
        pattern_mol = None
        if use_scaffold and scaffold_handler is not None:
            scaffold_smarts = scaffold_handler.fixed_substructure.replace('#', '*')
            pattern_mol = Chem.MolFromSmarts(scaffold_smarts)
        
        with torch.no_grad():
            # tokenize batch
            batch = self._tokenizer(smiles_batch, padding=True, return_tensors='pt')

            # Choose masking method based on mask_mode
            if use_scaffold and mask_mode == "sample_partition" and scaffold_handler is not None:
                # Generate masks using scaffold_handler's distribution method
                batch_ids, batch_mask = self._generate_scaffold_distributed_masks(
                    smiles_batch, 
                    scaffold_handler
                )
            elif use_scaffold and scaffold_handler is not None:
                # Use existing scaffold-aware masking
                batch_ids, batch_mask, _ = self.generate_masks_per_molecule(
                    batch['input_ids'], 
                    batch['attention_mask'], 
                    'replace',
                    use_scaffold=True,
                    scaffold_handler=scaffold_handler
                )
            elif not use_scaffold and mask_mode == "pure_random_mask":
                # Generate completely random masked molecules
                masked_mols = util.generate_masked_molecules_no_scaffold(len(smiles_batch))
                batch = self._tokenizer(masked_mols, padding=True, return_tensors='pt')
                batch_ids = batch['input_ids']
                batch_mask = batch['attention_mask']
            else:
                # Use standard masking
                if mask_mode == 'random':
                    mask_mode = np.random.choice(['replace', 'insert', 'delete'])
                batch_ids, batch_mask, _ = self.generate_masks(
                    batch['input_ids'], 
                    batch['attention_mask'], 
                    mask_mode
                )
                
            # Rest of the method remains the same...
            batch_ids = batch_ids.to(self._device)
            batch_mask = batch_mask.to(self._device)
            
            # generate token probabilities for masked inputs
            fake = self._gen(input_ids=batch_ids, attention_mask=batch_mask, hard=False).detach().cpu()
            batch_ids = batch_ids.detach().cpu()

            results_all = []
            results = []
            for i in range(fake.size(0)):
                # masked sequence
                input_ids = batch_ids[i]
                masked_sequences.append(self._tokenizer.decode(input_ids))

                # find probablities at locations with a mask token
                masked_index = torch.nonzero(input_ids == self._tokenizer.mask_token_id, as_tuple=False).flatten()
                probs = fake[i, masked_index, :]

                # find topk predictions for each mask token
                values, predictions = probs.topk(self._top_k)

                possible_indices = torch.zeros(len(predictions), dtype=torch.long)
                for k in range(self._top_k):
                    indices = None
                    if k == 0:
                        # take top predictions
                        indices = predictions[:,0]
                    else:
                        # find next best prediction
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

                    # fill in masks with predictions
                    new_ids = input_ids.clone()
                    new_ids[masked_index] = indices
                    smiles = self._tokenizer.decode(new_ids, skip_special_tokens=True).replace(' ','').replace('##','')
                    
                    results_all.append(smiles)
                    # Validate scaffold if enabled
                    if use_scaffold and pattern_mol is not None:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None or not mol.HasSubstructMatch(pattern_mol):
                            continue  # Skip molecules without scaffold
                    
                    results.append(smiles)

            return results, results_all, masked_sequences
    
    def save(self, generator_file, discriminator_file):
        """Save generator and discriminator."""
        if self._disc is None:
            raise AttributeError('Discriminator does not exist')
        torch.save(self._gen.state_dict(), generator_file)
        torch.save(self._disc.state_dict(), discriminator_file)







