from rlmollm.scoring.molecule_scoring import MoleculeScoring
from rdkit.Chem import Descriptors
import numpy as np
import scipy.stats
from rdkit.Chem import Crippen

class LinkerScoring(MoleculeScoring):
    """Custom scoring class for linker optimization with target LogP."""

    def __init__(self, style, scoring_names, selection_names, scoring_admet_names=None, scoring_parameters=None, 
                 data_column_name='smiles', fitness_column_name='fitness', 
                 fitness_function=scipy.stats.hmean):
        super().__init__(scoring_names, scoring_admet_names, selection_names, 
                        scoring_parameters=scoring_parameters, 
                        data_column_name=data_column_name, 
                        fitness_column_name=fitness_column_name, 
                        fitness_function=fitness_function)
        
        # Default values for target LogP parameters
        self._target_logp_value = 0.3  # Default target
        self._target_logp_sigma = 1.0  # Default tolerance
        
        # Override with values from scoring_parameters if provided
        if scoring_parameters is not None:
            if 'target_logp_value' in scoring_parameters:
                self._target_logp_value = float(scoring_parameters['target_logp_value'])
            if 'target_logp_sigma' in scoring_parameters:
                self._target_logp_sigma = float(scoring_parameters['target_logp_sigma'])

        # Default values for LogP range
        self._min_logp_value = 2.0
        self._max_logp_value = 3.0
        self._logp_sigma = 0.5  # Controls how quickly score drops outside range

        # Override with values from scoring_parameters if provided
        if scoring_parameters is not None:
            if 'min_logp_value' in scoring_parameters:
                self._min_logp_value = float(scoring_parameters['min_logp_value'])
            if 'max_logp_value' in scoring_parameters:
                self._max_logp_value = float(scoring_parameters['max_logp_value'])
            if 'logp_sigma' in scoring_parameters:
                self._logp_sigma = float(scoring_parameters['logp_sigma'])

    def get_name_to_function_dict(self):
        """Add custom scoring functions."""
        # Get the base scoring functions
        output_dict = super().get_name_to_function_dict()
        
        # Add our custom LogP scoring function
        output_dict['target_logp'] = self._target_logp_score

        output_dict['range_logp'] = self._range_logp_score

        
        return output_dict
    
    def _target_logp_score(self, mol, default=0.0):
        """Score based on LogP proximity to target value."""
        try:
            # logp = Descriptors.MolLogP(mol)
            logp = Crippen.MolLogP(mol)
            # Gaussian score centered at target value
            score = np.exp(-0.5 * ((logp - self._target_logp_value)/self._target_logp_sigma)**2)
            return score
        except:
            return default
        
    def _range_logp_score(self, mol, default=0.0):
        """Score based on LogP falling within a desired range."""
        try:
            # logp = Descriptors.MolLogP(mol)
            logp = Crippen.MolLogP(mol)
            # Get range parameters from scoring_parameters
            min_logp = self._min_logp_value  # e.g., 2.0
            max_logp = self._max_logp_value  # e.g., 3.0
            
            # If LogP is within range, give full score
            if min_logp <= logp <= max_logp:
                return 1.0
            
            # If below minimum, calculate distance from minimum
            elif logp < min_logp:
                distance = min_logp - logp
                # Use sigma for smoothing the transition
                return np.exp(-0.5 * (distance/self._logp_sigma)**2)
            
            # If above maximum, calculate distance from maximum
            else:  # logp > max_logp
                distance = logp - max_logp
                # Use sigma for smoothing the transition
                return np.exp(-0.5 * (distance/self._logp_sigma)**2)
                
        except:
            return default