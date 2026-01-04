import torch
import torch.nn.functional as F
import numpy as np
import rdkit.Chem as Chem
from .MoleculeValueNetwork import MoleculeValueNetwork


class PPOTrainer:
    """PPO trainer for molecular generation optimization."""
    
    def __init__(self, gan_operators, scoring_operator, device, lr=0.00005):
        """
        Initialize PPO trainer.
        
        Args:
            gan_operators: List of GAN operators
            scoring_operator: Scoring operator for molecule evaluation
            device: Device to run training on
            lr: Learning rate for optimizer
        """
        self.gan_operators = gan_operators
        self.scoring_operator = scoring_operator
        self.device = device
        self.lr = lr
        
        # Use the first GAN operator for PPO training
        self.gan = self.gan_operators[0]
        
        # Initialize value network
        self.value_head = MoleculeValueNetwork(
            hidden_size=self.gan._gen.embedding.config.hidden_size,
            device=device,
            dropout_rate=0.1
        ).to(device)
        
        # Initialize optimizer for both generator and value network
        self.optimizer = torch.optim.Adam([
            {'params': self.gan._gen.parameters(), 'lr': lr},
            {'params': self.value_head.parameters(), 'lr': lr * 2}
        ])
    
    def train_ppo(self, dataloader, ppo_epochs=4, clip_ratio=0.2, 
                  entropy_coef=0.01, value_coef=0.5, reward_scale=1.0, 
                  invalid_penalty=-0.1, batch_size=32, use_scaffold=False, 
                  scaffold_handler=None, mask_mode="replace"):
        """
        Train GANs with PPO algorithm to optimize molecular properties.
        
        Args:
            dataloader: Dataloader to iterate through dataset
            ppo_epochs: Number of PPO epochs per batch
            clip_ratio: PPO clipping parameter
            entropy_coef: Coefficient for entropy term in loss
            value_coef: Coefficient for value function loss
            reward_scale: Scaling factor for rewards
            invalid_penalty: Penalty for invalid molecules
            batch_size: Batch size for training
            use_scaffold: Whether to use scaffold constraints
            scaffold_handler: ScaffoldHandler object
            mask_mode: Masking mode for generation
            
        Returns:
            tuple with training metrics (ppo_loss, avg_reward, valid_rate)
        """
        # Set models to training mode
        self.gan._gen.train()
        self.value_head.train()
        
        # Initialize metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        valid_molecules = 0
        total_molecules = 0
        all_rewards = []
        batch_count = 0
        
        # Process each batch of molecules
        for batch_smiles in dataloader:
            batch_count += 1
            
            # Generate molecules with masked tokens and track probabilities
            molecules_info = self._generate_molecules_with_probs(
                batch_smiles, use_scaffold=use_scaffold, 
                scaffold_handler=scaffold_handler, mask_mode=mask_mode
            )
            
            # Calculate rewards for molecules
            molecules_info = self._calculate_molecule_rewards(
                molecules_info, invalid_penalty, reward_scale, 
                use_scaffold, scaffold_handler
            )
            
            # Track valid molecules and rewards
            batch_rewards = [m['reward'] for m in molecules_info if m.get('valid', False)]
            if batch_rewards:
                all_rewards.extend(batch_rewards)
            
            valid_molecules += sum(1 for m in molecules_info if m.get('valid', False))
            total_molecules += len(molecules_info)
            
            # Perform PPO updates for this batch
            for _ in range(ppo_epochs):
                # Shuffle molecules for PPO updates
                indices = torch.randperm(len(molecules_info))
                
                # Process in mini-batches
                for i in range(0, len(indices), batch_size):
                    mini_batch_indices = indices[i:i+batch_size]
                    mini_batch = [molecules_info[idx] for idx in mini_batch_indices]
                    
                    # Skip if mini-batch is empty
                    if not mini_batch:
                        continue
                    
                    # Process mini-batch and get losses
                    policy_loss, value_loss, entropy_loss = self._process_mini_batch(
                        mini_batch, clip_ratio, value_coef, entropy_coef
                    )
                    
                    # Skip if no valid molecules in mini-batch
                    if policy_loss is None:
                        continue
                    
                    # Combined loss
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
                    
                    # Update models
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Track metrics
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy_loss.item()
        
        # Calculate average metrics
        if batch_count > 0:
            avg_policy_loss = total_policy_loss / batch_count
            avg_value_loss = total_value_loss / batch_count
            avg_entropy = total_entropy / batch_count
        else:
            avg_policy_loss = 0
            avg_value_loss = 0
            avg_entropy = 0
        
        # Calculate final statistics
        valid_rate = valid_molecules / max(1, total_molecules)
        avg_reward = sum(all_rewards) / max(1, len(all_rewards)) if all_rewards else 0
        
        # Return combined loss and other metrics
        return avg_policy_loss + avg_value_loss, avg_reward, valid_rate
    
    def _generate_molecules_with_probs(self, smiles_batch, use_scaffold=False, 
                                       scaffold_handler=None, mask_mode="replace"):
        """
        Generate molecules with probability tracking.
        
        Args:
            smiles_batch: List of SMILES strings
            use_scaffold: Whether to use scaffold constraints
            scaffold_handler: ScaffoldHandler object
            mask_mode: Masking mode for generation
            
        Returns:
            List of dictionaries with molecule information
        """
        if isinstance(smiles_batch, np.ndarray):
            smiles_batch = smiles_batch.tolist()
        
        self.gan._gen.eval()
        self.value_head.eval()
        all_results = []
        
        with torch.no_grad():
            # Tokenize batch
            batch = self.gan._tokenizer(smiles_batch, padding=True, return_tensors='pt')
            
            # Generate masks based on mode and scaffold settings
            if mask_mode == "sample_partition" and scaffold_handler is not None:
                # Generate masks using scaffold_handler's distribution method
                batch_ids, batch_mask = self.gan._generate_scaffold_distributed_masks(
                    smiles_batch, scaffold_handler
                )
            else:
                # Use standard per-molecule masking (with or without scaffold awareness)
                batch_ids, batch_mask, _ = self.gan.generate_masks_per_molecule(
                    batch['input_ids'], 
                    batch['attention_mask'],
                    mask_mode,
                    use_scaffold=use_scaffold,
                    scaffold_handler=scaffold_handler
                )
            
            # Store original masked inputs for reference
            masked_inputs = []
            for i in range(batch_ids.size(0)):
                masked_inputs.append({
                    'input_ids': batch_ids[i].clone(),
                    'attention_mask': batch_mask[i].clone(),
                    'masked_smiles': self.gan._tokenizer.decode(batch_ids[i])
                })
            
            # Move batch to device
            batch_ids = batch_ids.to(self.device)
            batch_mask = batch_mask.to(self.device)
            
            # Run the model once to get both hidden states and logits
            model_outputs = self.gan._gen.embedding(
                input_ids=batch_ids, 
                attention_mask=batch_mask,
                output_hidden_states=True
            )
            # Get logits for token prediction
            logits = model_outputs.logits
            # Get the hidden states from the last layer for value estimation
            hidden_states = model_outputs.hidden_states[-1]

            # Feed the hidden states to the value head
            values = self.value_head(hidden_states, batch_mask)
            
            # Process each molecule in the batch
            for i in range(batch_ids.size(0)):
                molecule_info = {
                    'masked_input': masked_inputs[i],
                    'value': values[i].item()
                }
                
                # Find positions with mask tokens
                input_ids = batch_ids[i]
                masked_index = torch.nonzero(
                    input_ids == self.gan._tokenizer.mask_token_id, as_tuple=False
                ).flatten()
                
                if len(masked_index) == 0:
                    # Skip molecules with no masks
                    continue
                
                # Get logits for masked positions
                mask_logits = logits[i, masked_index, :]
                
                # Create distribution for sampling
                probs = torch.nn.functional.softmax(mask_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                # Sample from distribution
                sampled_tokens = dist.sample()
                molecule_info['sampled_token_ids'] = sampled_tokens
                
                # Calculate log probabilities
                log_probs = dist.log_prob(sampled_tokens)
                entropy = dist.entropy()
                molecule_info['log_probs'] = log_probs
                molecule_info['entropy'] = entropy.mean().item()
                
                # Create new token sequence with sampled tokens
                new_ids = input_ids.clone()
                new_ids[masked_index] = sampled_tokens
                
                # Decode to SMILES
                smiles = self.gan._tokenizer.decode(
                    new_ids, skip_special_tokens=True
                ).replace(' ', '').replace('##', '')
                molecule_info['smiles'] = smiles
                
                all_results.append(molecule_info)
        
        return all_results
    
    def _calculate_molecule_rewards(self, molecules_info, invalid_penalty, 
                                    reward_scale, use_scaffold=False, scaffold_handler=None):
        """
        Calculate rewards for generated molecules.
        
        Args:
            molecules_info: List of dictionaries with molecule information
            invalid_penalty: Penalty for invalid molecules
            reward_scale: Scaling factor for rewards
            use_scaffold: Whether to enforce scaffold constraints
            scaffold_handler: ScaffoldHandler object for scaffold validation
            
        Returns:
            Updated molecules_info with rewards
        """
        # Extract SMILES strings
        smiles_list = [mol_info['smiles'] for mol_info in molecules_info]
        
        # Set up scaffold validation if enabled
        pattern_mol = None
        if use_scaffold and scaffold_handler is not None:
            scaffold_smarts = scaffold_handler.fixed_substructure.replace('#', '*')
            pattern_mol = Chem.MolFromSmarts(scaffold_smarts)
        
        # Prepare molecules for scoring
        rdkit_mols = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            mol = self.scoring_operator.prepare_data_for_scoring(smiles)
            
            # Validate molecule is valid and contains scaffold if required
            if mol is not None:
                scaffold_valid = True
                if use_scaffold and pattern_mol is not None:
                    scaffold_valid = mol.HasSubstructMatch(pattern_mol)
                
                if scaffold_valid:
                    rdkit_mols.append(mol)
                    valid_indices.append(i)
                    molecules_info[i]['valid'] = True
                else:
                    molecules_info[i]['valid'] = True  # not contain scaffold, but still valid
                    molecules_info[i]['reward'] = invalid_penalty
                    molecules_info[i]['scores'] = {}
                    # Optionally add a flag indicating scaffold failure
                    molecules_info[i]['scaffold_failure'] = True
            else:
                molecules_info[i]['valid'] = False
                molecules_info[i]['reward'] = invalid_penalty
                molecules_info[i]['scores'] = {}
        
        # Score valid molecules
        if rdkit_mols:
            scores = self.scoring_operator.generate_scores(rdkit_mols)
            
            # Assign scores and rewards to valid molecules
            for idx, orig_idx in enumerate(valid_indices):
                fitness = scores[self.scoring_operator.fitness_column_name][idx]
                molecules_info[orig_idx]['reward'] = fitness * reward_scale
                
                # Store all calculated scores
                mol_scores = {}
                for score_name in self.scoring_operator.column_names:
                    if score_name != self.scoring_operator.data_column_name:
                        mol_scores[score_name] = scores[score_name][idx]
                molecules_info[orig_idx]['scores'] = mol_scores
        
        return molecules_info
    
    def _process_mini_batch(self, mini_batch, clip_ratio, value_coef, entropy_coef):
        """
        Process a mini-batch for PPO updates.
        
        Args:
            mini_batch: List of molecule info dictionaries
            clip_ratio: PPO clipping parameter
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy term
            
        Returns:
            tuple of (policy_loss, value_loss, entropy_loss) or (None, None, None) if invalid
        """
        # Initialize losses for this mini-batch
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Process each molecule
        for mol_info in mini_batch:
            # Skip if no log_probs (means no masks)
            if 'log_probs' not in mol_info:
                continue
            
            # Get data for this molecule
            masked_input = mol_info['masked_input']
            old_log_probs = mol_info.get('log_probs', [])
            old_value = mol_info.get('value', 0)
            reward = mol_info.get('reward', -0.1)
            
            # Skip if no log probs
            if len(old_log_probs) == 0:
                continue
            
            # Convert to tensors
            old_log_probs = torch.stack([lp for lp in old_log_probs]).to(self.device)
            reward_tensor = torch.tensor(reward, device=self.device, dtype=torch.float32)
            
            # Calculate advantage
            advantage = (reward_tensor - old_value).detach()
            
            # Forward pass through generator
            input_ids = masked_input['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = masked_input['attention_mask'].unsqueeze(0).to(self.device)
            
            # Run the model to get outputs
            model_outputs = self.gan._gen.embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits = model_outputs.logits
            hidden_states = model_outputs.hidden_states[-1]
            
            # Get new value prediction
            new_value = self.value_head(hidden_states, attention_mask).squeeze()
            
            # Find masked positions
            masked_index = torch.nonzero(
                input_ids[0] == self.gan._tokenizer.mask_token_id, as_tuple=False
            ).flatten()
            
            if len(masked_index) == 0:
                continue
            
            # Get new probabilities for each masked position
            new_mask_logits = logits[0, masked_index, :]
            new_probs = F.softmax(new_mask_logits, dim=-1)
            
            # Calculate new log probs and entropy
            new_log_probs = []
            entropy = 0
            for idx, token_id in enumerate(mol_info['sampled_token_ids']):
                dist = torch.distributions.Categorical(new_probs[idx])
                new_log_probs.append(dist.log_prob(token_id))
                entropy += dist.entropy()
            
            new_log_probs = torch.stack(new_log_probs)
            entropy = entropy / len(new_log_probs)
            
            # Calculate PPO policy loss
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(new_value, reward_tensor)
            
            # Collect losses
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy.mean())
        
        # Skip if no valid molecules in mini-batch
        if not policy_losses:
            return None, None, None
        
        # Compute average losses
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()
        
        return policy_loss, value_loss, entropy_loss 