# scaffold_handler.py
import torch
import random
import numpy as np
import re
from rlmollm.utils.sample_util import simple_diverse_sampling, beam_search_sampling

class ScaffoldHandler:
    """Class to handle scaffold-based molecule generation."""
    
    def __init__(self, config, tokenizer, device="cpu"):
        """Initialize scaffold handler.
        
        Args:
            config: Configuration with scaffold options
            tokenizer: Tokenizer for processing molecules
            device: Device for tensor operations
        """
        self.fixed_substructure = config.get("fixed_substructure", "")
        self.min_mask_per_position = config.get("min_mask_per_position", 1)
        self.max_mask_num = config.get("max_masks", 10)
        self.min_masks = config.get("min_masks", 0)  # Minimum total masks across the molecule
        self.generate_initial_molecules = config.get("generate_initial_molecules", True)
        self.tokenizer = tokenizer
        self.device = device
        
        # Process the scaffold template
        self.attachment_points = self._find_attachment_points()
        self.scaffold_template = self._prepare_scaffold_template()
        
    def _find_attachment_points(self):
        """Find the positions of attachment points (#) in the scaffold template.
        
        Returns:
            List of indices for attachment points
        """
        return [i for i, char in enumerate(self.fixed_substructure) if char == '#']
    
    def _prepare_scaffold_template(self):
        """Convert scaffold template to SMILES by replacing # with [MASK].
        
        Returns:
            Scaffold template with # replaced by [MASK]
        """
        # Replace # with [MASK] for the tokenizer
        scaffold = self.fixed_substructure.replace('#', '[MASK]')
        return scaffold
    
    def generate_masked_molecules(self, num_molecules):
        """Generate masked molecules based on the scaffold template using a simple partitioning approach.
        
        Args:
            num_molecules: Number of molecules to generate
            
        Returns:
            List of unique masked molecules as strings
        """
        masked_molecules = set()
        max_attempts = num_molecules * 5
        attempts = 0
        
        # Find all # positions in the original scaffold
        scaffold = self.fixed_substructure
        attachment_positions = [i for i, char in enumerate(scaffold) if char == '#']
        num_attachment_points = len(attachment_positions)
        
        # Calculate base masks from min_mask_per_position requirement
        base_masks = num_attachment_points * self.min_mask_per_position  # Each # gets min_mask_per_position [MASK] tokens
        
        # Calculate maximum extra masks possible
        max_extra_masks = self.max_mask_num - base_masks
        
        # print(f"Scaffold: {scaffold}")
        # print(f"Attachment positions: {attachment_positions}")
        # print(f"Number of attachment points: {num_attachment_points}")
        # print(f"Base masks: {base_masks}")
        # print(f"Min masks: {self.min_masks}")
        # print(f"Max possible extra masks: {max_extra_masks}")
        
        while len(masked_molecules) < num_molecules and attempts < max_attempts:
            attempts += 1
            
            # 1. Randomly decide how many total extra masks to add
            # Ensure we have at least min_masks in total
            min_extra_masks = max(0, self.min_masks - base_masks)
            total_extra_masks = random.randint(min_extra_masks, max_extra_masks)
            
            # 2. Distribute these extra masks among the attachment points
            extra_masks_per_position = [0] * num_attachment_points
            
            # Simple distribution: randomly assign each extra mask to an attachment point
            for _ in range(total_extra_masks):
                position_idx = random.randint(0, num_attachment_points - 1)
                extra_masks_per_position[position_idx] += 1
            
            # 3. Build the molecule with masks
            result = ""
            last_pos = 0
            
            for i, pos in enumerate(attachment_positions):
                # Add everything up to this attachment point
                result += scaffold[last_pos:pos]
                
                # Add the minimum required [MASK] tokens that replace the #
                result += "[MASK]" * self.min_mask_per_position
                
                # Add any extra masks for this attachment point
                result += "[MASK]" * extra_masks_per_position[i]
                
                # Update last position
                last_pos = pos + 1  # +1 to skip the #
            
            # Add any remaining part of the scaffold
            if last_pos < len(scaffold):
                result += scaffold[last_pos:]
            
            # Debug output
            # if attempts <= 5 or attempts % 100 == 0:
            #     print(f"Attempt {attempts}: Generated molecule: {result}")
            #     print(f"  - Total extra masks: {total_extra_masks}")
            #     print(f"  - Extra masks per position: {extra_masks_per_position}")
            #     print(f"  - Total masks: {base_masks + total_extra_masks}")
            
            # Add to set if unique
            if result not in masked_molecules:
                masked_molecules.add(result)
        
        #print(f"Generated {len(masked_molecules)} unique masked molecules")
        return list(masked_molecules)
    
    def generate_initial_population(self, gan, population_size, batch_size=10, top_k=5, use_simple_method=True, valid_only=True):
        """Generate initial population of molecules from the scaffold.
        
        Args:
            gan: GAN model for generation
            population_size: Desired population size
            batch_size: Batch size for generation
            top_k: Number of top predictions to use
            use_simple_method: If True, use simple diverse approach; if False, use advanced beam search
            valid_only: If True, only return valid molecules containing the scaffold
            
        Returns:
            List of generated SMILES strings
        """
        import rdkit.Chem as Chem
        results = []
        
        # Get the scaffold pattern (replace # with * for SMARTS matching)
        scaffold_smarts = self.fixed_substructure.replace('#', '*')
        pattern_mol = None
        
        if valid_only:
            pattern_mol = Chem.MolFromSmarts(scaffold_smarts)
            if pattern_mol is None:
                raise ValueError(f"Invalid scaffold SMARTS pattern: {scaffold_smarts}")
        
        # Generate unique masked molecules
        needed_templates = population_size 
        masked_mols = self.generate_masked_molecules(needed_templates)
        
        # Process in batches
        for batch_start in range(0, len(masked_mols), batch_size):
            # Exit if we already have enough results
            if len(results) >= population_size:
                break
                
            batch_end = min(batch_start + batch_size, len(masked_mols))
            batch_mols = masked_mols[batch_start:batch_end]
            
            # Tokenize batch
            batch = self.tokenizer(batch_mols, padding=True, return_tensors='pt')
            batch_ids = batch['input_ids'].to(self.device)
            batch_mask = batch['attention_mask'].to(self.device)
            
            # Generate token probabilities
            with torch.no_grad():
                fake = gan._gen(input_ids=batch_ids, attention_mask=batch_mask, hard=False).detach().cpu()
            
            # Process each sequence in the batch
            for i in range(fake.size(0)):
                input_ids = batch_ids[i].detach().cpu()
                
                # Find mask token positions
                masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).flatten()
                if len(masked_index) == 0:
                    continue
                    
                probs = fake[i, masked_index, :]
                
                # Get top-k predictions for each mask
                values, predictions = probs.topk(top_k)
                
                if use_simple_method:
                    # Use the simple diverse sampling approach
                    results, done = simple_diverse_sampling(
                        input_ids=input_ids,
                        masked_index=masked_index,
                        predictions=predictions,
                        tokenizer=self.tokenizer,
                        top_k=top_k,
                        results=results,
                        population_size=population_size,
                        pattern_mol=pattern_mol,
                        valid_only=valid_only
                    )
                    
                    if done:
                        return results
                else:
                    # Use the beam search sampling approach
                    results, done = beam_search_sampling(
                        input_ids=input_ids,
                        masked_index=masked_index,
                        predictions=predictions,
                        values=values,
                        tokenizer=self.tokenizer,
                        top_k=top_k,
                        results=results,
                        population_size=population_size,
                        pattern_mol=pattern_mol,
                        valid_only=valid_only
                    )
                    
                    if done:
                        return results
        
        # If we couldn't generate enough molecules with scaffold
        if valid_only:
            print(f"Warning: Could only generate {len(results)}/{population_size} valid molecules with scaffold")
        else:
            print(f"Warning: Could only generate {len(results)}/{population_size} molecules")
        return results

    def mask_attachment_points(self, smiles_str, mutation_parameter):
        """Mask tokens at the original attachment points with probability mutation_parameter.
        
        Args:
            smiles_str: SMILES string
            mutation_parameter: Probability of masking a token
            
        Returns:
            Masked SMILES string
        """
        # This implementation would need to track where the original attachment points
        # ended up in the final molecule, which is challenging
        # For a real implementation, you might want to use RDKit to identify
        # attachment points in the generated molecules
        tokens = list(smiles_str)
        for i in range(len(tokens)):
            if random.random() < mutation_parameter:
                tokens[i] = '[MASK]'
        
        return ''.join(tokens)

  