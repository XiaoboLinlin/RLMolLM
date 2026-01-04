import torch
import rdkit.Chem as Chem

def simple_diverse_sampling(
    input_ids, 
    masked_index, 
    predictions, 
    tokenizer, 
    top_k, 
    results, 
    population_size, 
    pattern_mol=None,
    valid_only=True
):
    """Simple diverse sampling approach: take kth best token for all positions.
    
    Args:
        input_ids: Original input token IDs
        masked_index: Indices of mask tokens
        predictions: Top-k predictions for each mask position
        tokenizer: Tokenizer for decoding
        top_k: Number of top predictions to use
        results: Current list of results to append to
        population_size: Maximum number of results needed
        pattern_mol: RDKit molecule pattern to match (optional)
        valid_only: If True, only return valid molecules containing the scaffold
        
    Returns:
        Tuple of (updated results, boolean indicating if population_size reached)
    """
    for k in range(top_k):
        # Make a copy of input_ids
        new_ids = input_ids.clone()
        
        # Use kth prediction for each mask
        indices = predictions[:, k]
        
        # Fill in masks with predictions
        new_ids[masked_index] = indices
        smiles = tokenizer.decode(new_ids, skip_special_tokens=True).replace(' ','').replace('##','')
        
        if valid_only:
            # Validate scaffold is present in the molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and (pattern_mol is None or mol.HasSubstructMatch(pattern_mol)):
                # Convert to canonical SMILES
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                
                # Only add unique valid molecules with scaffold
                if canonical_smiles not in results:
                    results.append(canonical_smiles)
                
                # Stop if we've generated enough molecules
                if len(results) >= population_size:
                    return results, True
        else:
            # For non-valid_only mode, still try to canonicalize if possible
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                if canonical_smiles not in results:
                    results.append(canonical_smiles)
            else:
                # Add raw SMILES if it can't be parsed
                if smiles not in results:
                    results.append(smiles)
            
            # Stop if we've generated enough molecules
            if len(results) >= population_size:
                return results, True
                
    return results, False

def beam_search_sampling(
    input_ids, 
    masked_index, 
    predictions, 
    values, 
    tokenizer, 
    top_k, 
    results, 
    population_size, 
    pattern_mol=None,
    valid_only=True
):
    """Advanced beam search approach: optimize joint probability.
    
    Args:
        input_ids: Original input token IDs
        masked_index: Indices of mask tokens
        predictions: Top-k predictions for each mask position
        values: Probability values for the predictions
        tokenizer: Tokenizer for decoding
        top_k: Number of top predictions to use
        results: Current list of results to append to
        population_size: Maximum number of results needed
        pattern_mol: RDKit molecule pattern to match (optional)
        valid_only: If True, only return valid molecules containing the scaffold
        
    Returns:
        Tuple of (updated results, boolean indicating if population_size reached)
    """
    possible_indices = torch.zeros(len(predictions), dtype=torch.long)
    for k in range(top_k):
        indices = None
        if k == 0:
            # Take top predictions
            indices = predictions[:, 0]
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
        smiles = tokenizer.decode(new_ids, skip_special_tokens=True).replace(' ','').replace('##','')
        
        if valid_only:
            # Validate scaffold is present in the molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and (pattern_mol is None or mol.HasSubstructMatch(pattern_mol)):
                # Convert to canonical SMILES
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                
                # Only add unique valid molecules with scaffold
                if canonical_smiles not in results:
                    results.append(canonical_smiles)
                
                # Stop if we've generated enough molecules
                if len(results) >= population_size:
                    return results, True
        else:
            # For non-valid_only mode, still try to canonicalize if possible
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                if canonical_smiles not in results:
                    results.append(canonical_smiles)
            else:
                # Add raw SMILES if it can't be parsed
                if smiles not in results:
                    results.append(smiles)
            
            # Stop if we've generated enough molecules
            if len(results) >= population_size:
                return results, True
                
    return results, False 