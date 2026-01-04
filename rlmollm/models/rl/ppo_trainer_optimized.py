import torch
import torch.nn.functional as F
import numpy as np
import rdkit.Chem as Chem
from .MoleculeValueNetwork import MoleculeValueNetwork


class PPOTrainerOptimized:
    """Optimized PPO trainer for molecular generation optimization with vectorization."""
    
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
        OPTIMIZED VERSION with vectorization.
        
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
            
            # OPTIMIZATION 1: Batch molecule generation with vectorization
            molecules_info = self._generate_molecules_with_probs_vectorized(
                batch_smiles, use_scaffold=use_scaffold, 
                scaffold_handler=scaffold_handler, mask_mode=mask_mode
            )
            
            # OPTIMIZATION 2: Batch reward calculation
            molecules_info = self._calculate_molecule_rewards_vectorized(
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
                
                # OPTIMIZATION 3: Vectorized mini-batch processing
                for i in range(0, len(indices), batch_size):
                    mini_batch_indices = indices[i:i+batch_size]
                    mini_batch = [molecules_info[idx] for idx in mini_batch_indices]
                    
                    # Skip if mini-batch is empty
                    if not mini_batch:
                        continue
                    
                    # Process mini-batch and get losses (vectorized)
                    policy_loss, value_loss, entropy_loss = self._process_mini_batch_vectorized(
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
    
    def _generate_molecules_with_probs_vectorized(self, smiles_batch, use_scaffold=False, 
                                                  scaffold_handler=None, mask_mode="replace"):
        """
        OPTIMIZED: Generate molecules with probability tracking using vectorization.
        DETERMINISTIC: Processes molecules in the same order as original for consistent results.
        
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
        
        with torch.no_grad():
            # Tokenize entire batch at once
            batch = self.gan._tokenizer(smiles_batch, padding=True, return_tensors='pt')
            
            # Generate masks based on mode and scaffold settings
            if mask_mode == "sample_partition" and scaffold_handler is not None:
                batch_ids, batch_mask = self.gan._generate_scaffold_distributed_masks(
                    smiles_batch, scaffold_handler
                )
            else:
                batch_ids, batch_mask, _ = self.gan.generate_masks_per_molecule(
                    batch['input_ids'], 
                    batch['attention_mask'],
                    mask_mode,
                    use_scaffold=use_scaffold,
                    scaffold_handler=scaffold_handler
                )
            
            # Move batch to device
            batch_ids = batch_ids.to(self.device)
            batch_mask = batch_mask.to(self.device)
            
            # OPTIMIZATION: Single forward pass for entire batch
            model_outputs = self.gan._gen.embedding(
                input_ids=batch_ids, 
                attention_mask=batch_mask,
                output_hidden_states=True
            )
            logits = model_outputs.logits  # [batch_size, seq_len, vocab_size]
            hidden_states = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # OPTIMIZATION: Batch value prediction
            values = self.value_head(hidden_states, batch_mask)  # [batch_size]
            
            # OPTIMIZED: Process molecules in original order to maintain determinism
            all_results = []
            mask_token_id = self.gan._tokenizer.mask_token_id
            
            # Collect all mask logits and molecule info in original order
            all_mask_logits = []
            molecule_mask_info = []
            
            for i in range(batch_ids.size(0)):
                # Find masked positions for this molecule (same as original PPO)
                input_ids = batch_ids[i]
                masked_indices = torch.nonzero(
                    input_ids == mask_token_id, as_tuple=False
                ).flatten()
                
                if len(masked_indices) == 0:
                    continue
                
                # Get logits for this molecule's masks
                mol_mask_logits = logits[i, masked_indices, :]  # [num_masks, vocab_size]
                all_mask_logits.append(mol_mask_logits)
                
                # Store molecule info for later reconstruction
                molecule_mask_info.append({
                    'molecule_idx': i,
                    'masked_indices': masked_indices,
                    'num_masks': len(masked_indices)
                })
            
            if len(all_mask_logits) == 0:
                return all_results  # No masks to process
            
            # VECTORIZED: Concatenate all mask logits and sample at once
            batch_mask_logits = torch.cat(all_mask_logits, dim=0)  # [total_masks, vocab_size]
            probs = F.softmax(batch_mask_logits, dim=-1)  # [total_masks, vocab_size]
            dist = torch.distributions.Categorical(probs)
            sampled_tokens = dist.sample()  # [total_masks]
            log_probs = dist.log_prob(sampled_tokens)  # [total_masks]
            entropies = dist.entropy()  # [total_masks]
            
            # Reconstruct molecules with sampled tokens (maintaining original order)
            token_idx = 0
            for mol_info in molecule_mask_info:
                i = mol_info['molecule_idx']
                masked_indices = mol_info['masked_indices']
                num_masks = mol_info['num_masks']
                
                # Get tokens and probabilities for this molecule
                mol_sampled_tokens = sampled_tokens[token_idx:token_idx + num_masks]
                mol_log_probs = log_probs[token_idx:token_idx + num_masks]
                mol_entropy = entropies[token_idx:token_idx + num_masks].mean()
                token_idx += num_masks
                
                # Create new token sequence with sampled tokens (same as original PPO)
                new_ids = batch_ids[i].clone()
                new_ids[masked_indices] = mol_sampled_tokens
                
                # Decode to SMILES (same as original PPO)
                smiles = self.gan._tokenizer.decode(
                    new_ids, skip_special_tokens=True
                ).replace(' ', '').replace('##', '')
                
                # Store molecule info (same format as original)
                molecule_info = {
                    'masked_input': {
                        'input_ids': batch_ids[i].clone(),
                        'attention_mask': batch_mask[i].clone(),
                        'masked_smiles': self.gan._tokenizer.decode(batch_ids[i], skip_special_tokens=True)
                    },
                    'value': values[i].item(),
                    'sampled_token_ids': mol_sampled_tokens,
                    'log_probs': mol_log_probs,
                    'entropy': mol_entropy.item(),
                    'smiles': smiles
                }
                
                all_results.append(molecule_info)
        
        return all_results
    
    def _calculate_molecule_rewards_vectorized(self, molecules_info, invalid_penalty, 
                                               reward_scale, use_scaffold=False, scaffold_handler=None):
        """
        OPTIMIZED: Calculate rewards for generated molecules using batch processing.
        
        Args:
            molecules_info: List of dictionaries with molecule information
            invalid_penalty: Penalty for invalid molecules
            reward_scale: Scaling factor for rewards
            use_scaffold: Whether to enforce scaffold constraints
            scaffold_handler: ScaffoldHandler object for scaffold validation
            
        Returns:
            Updated molecules_info with rewards
        """
        # OPTIMIZATION: Batch molecule preparation
        smiles_list = [mol_info['smiles'] for mol_info in molecules_info]
        
        # Set up scaffold validation if enabled
        pattern_mol = None
        if use_scaffold and scaffold_handler is not None:
            scaffold_smarts = scaffold_handler.fixed_substructure.replace('#', '*')
            pattern_mol = Chem.MolFromSmarts(scaffold_smarts)
        
        # OPTIMIZATION: Vectorized molecule validation
        rdkit_mols = []
        valid_indices = []
        
        # Batch prepare molecules (this could be further optimized with multiprocessing)
        for i, smiles in enumerate(smiles_list):
            mol = self.scoring_operator.prepare_data_for_scoring(smiles)
            
            if mol is not None:
                scaffold_valid = True
                if use_scaffold and pattern_mol is not None:
                    scaffold_valid = mol.HasSubstructMatch(pattern_mol)
                
                if scaffold_valid:
                    rdkit_mols.append(mol)
                    valid_indices.append(i)
                    molecules_info[i]['valid'] = True
                else:
                    molecules_info[i]['valid'] = True
                    molecules_info[i]['reward'] = invalid_penalty
                    molecules_info[i]['scores'] = {}
                    molecules_info[i]['scaffold_failure'] = True
            else:
                molecules_info[i]['valid'] = False
                molecules_info[i]['reward'] = invalid_penalty
                molecules_info[i]['scores'] = {}
        
        # OPTIMIZATION: Batch scoring (single call instead of individual scoring)
        if rdkit_mols:
            scores = self.scoring_operator.generate_scores(rdkit_mols)
            
            # Vectorized score assignment
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
    
    def _process_mini_batch_vectorized(self, mini_batch, clip_ratio, value_coef, entropy_coef):
        """
        OPTIMIZED: Process a mini-batch for PPO updates using vectorization.
        
        Args:
            mini_batch: List of molecule info dictionaries
            clip_ratio: PPO clipping parameter
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy term
            
        Returns:
            tuple of (policy_loss, value_loss, entropy_loss) or (None, None, None) if invalid
        """
        # Filter valid molecules with log_probs
        valid_molecules = [mol for mol in mini_batch if 'log_probs' in mol and len(mol['log_probs']) > 0]
        
        if not valid_molecules:
            return None, None, None
        
        # OPTIMIZATION: Batch tensor preparation
        batch_size = len(valid_molecules)
        
        # Prepare batch inputs
        batch_input_ids = []
        batch_attention_masks = []
        batch_old_log_probs = []
        batch_old_values = []
        batch_rewards = []
        batch_sampled_tokens = []
        
        for mol_info in valid_molecules:
            masked_input = mol_info['masked_input']
            batch_input_ids.append(masked_input['input_ids'])
            batch_attention_masks.append(masked_input['attention_mask'])
            batch_old_log_probs.append(mol_info['log_probs'])
            batch_old_values.append(mol_info.get('value', 0))
            batch_rewards.append(mol_info.get('reward', -0.1))
            batch_sampled_tokens.append(mol_info['sampled_token_ids'])
        
        # OPTIMIZATION: Single forward pass for entire mini-batch
        batch_input_ids = torch.stack(batch_input_ids).to(self.device)  # [batch_size, seq_len]
        batch_attention_masks = torch.stack(batch_attention_masks).to(self.device)
        
        model_outputs = self.gan._gen.embedding(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_masks,
            output_hidden_states=True
        )
        logits = model_outputs.logits  # [batch_size, seq_len, vocab_size]
        hidden_states = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # OPTIMIZATION: Batch value prediction
        new_values = self.value_head(hidden_states, batch_attention_masks).squeeze()  # [batch_size]
        if new_values.dim() == 0:  # Handle single element case
            new_values = new_values.unsqueeze(0)
        
        # OPTIMIZATION: Vectorized loss calculation
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Convert to tensors
        batch_old_values = torch.tensor(batch_old_values, device=self.device, dtype=torch.float32)
        batch_rewards = torch.tensor(batch_rewards, device=self.device, dtype=torch.float32)
        
        # Calculate advantages
        advantages = (batch_rewards - batch_old_values).detach()  # [batch_size]
        
        for i, mol_info in enumerate(valid_molecules):
            # Find masked positions for this molecule
            input_ids = batch_input_ids[i]
            masked_indices = torch.nonzero(
                input_ids == self.gan._tokenizer.mask_token_id, as_tuple=False
            ).flatten()
            
            if len(masked_indices) == 0:
                continue
            
            # Get old log probs and sampled tokens
            old_log_probs = batch_old_log_probs[i].to(self.device)
            sampled_tokens = batch_sampled_tokens[i].to(self.device)
            
            # OPTIMIZATION: Vectorized probability calculation
            new_mask_logits = logits[i, masked_indices, :]  # [num_masks, vocab_size]
            new_probs = F.softmax(new_mask_logits, dim=-1)
            
            # Calculate new log probs and entropy for all positions at once
            new_log_probs = []
            entropy_sum = 0
            
            for idx, token_id in enumerate(sampled_tokens):
                dist = torch.distributions.Categorical(new_probs[idx])
                new_log_probs.append(dist.log_prob(token_id))
                entropy_sum += dist.entropy()
            
            new_log_probs = torch.stack(new_log_probs)
            entropy = entropy_sum / len(new_log_probs)
            
            # PPO policy loss calculation
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            advantage = advantages[i]
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values[i], batch_rewards[i])
            
            # Collect losses
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy)
        
        if not policy_losses:
            return None, None, None
        
        # OPTIMIZATION: Vectorized loss aggregation
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()
        
        return policy_loss, value_loss, entropy_loss
