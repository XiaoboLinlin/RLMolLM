import rdkit
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import numpy as np
import math
import gzip
import pickle
import re
import scipy.stats
from rlmollm.scoring.scoring_interface import ScoringInterface

def smiles_to_mol(smiles):
    """Convert smiles to rdkit molecule.

    Args:
        smiles (str): smiles string representation of a molecule

    Returns:
        rdkit.Chem.rdchem.Mol

    """
    try:
        return Chem.MolFromSmiles(smiles, sanitize=True)
    except:
        return None

def mol_to_canonical_smiles(mol):
    """Convert rdkit molecule to canonical smiles string.

    Args:
        mol (rdkit.Chem.rdchem.Mol): rdkit molecule

    Returns:
        str representation of rdkit molecules (canonical)

    """
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def remap(x, x_min, x_max):
    """Translate and scale a given input.

    Args:
        x (float): original value
        x_min (float): value to subtract and lower bound for scale
        x_max (float): upper bound for scale

    Returns:
        float value translated and scaled

    """
    return (x - x_min) / (x_max - x_min)

class MoleculeScoring(ScoringInterface):
    """Class to score smiles sequences."""

    def __init__(self, scoring_names, scoring_admet_names, selection_names, scoring_parameters=None, data_column_name='smiles', fitness_column_name='fitness', fitness_function=scipy.stats.hmean):
        """Constructor for MoleculeScoring class.

        Args:
            scoring_names (List[str]): List of names for scoring functions to use
            selection_names (List[str]): List of names for selection functions to use
            scoring_parameters (Dict[str, str]): Dictionary of parameters needed for scoring functions
            data_column_name (str): Name used data column
            fitness_column_name (str): Name used for fitness column
            fitness_function (function): Function used to calculate fitness score from selection metrics

        """
        super().__init__()

        # setup scoring parameters
        if scoring_parameters is None:
            scoring_parameters = {}

        # Dictionary storing mapping from names to scoring functions
        self._name_to_function = self.get_name_to_function_dict()

        # store variables
        self._data_column_name = data_column_name
        self._fitness_column_name = fitness_column_name
        self._fitness_function = fitness_function

        # exempt brackets for data cleaning
        self._exempt_brackets = {'[C@@H]', '[C@H]', '[C@]', '[C@@]', '[nH]'}

        # check that data and fitness column names are not part of possible scoring functions
        if self.data_column_name in self._name_to_function:
            raise ValueError('Error: data column name ' + self.data_column_name + ' cannot be ' + ', '.join(self._name_to_function.keys()))

        if self.fitness_column_name in self._name_to_function:
            raise ValueError('Error: data column name ' + self.data_column_name + ' cannot be ' + ', '.join(self._name_to_function.keys()))

        # Note: SA scoring now uses RDKit's built-in sascorer directly
        # The old pickle-based sa_model approach is deprecated and removed

        # store names of functions to be used in scoring - make sure that selection names are subset
        self._scoring_names = scoring_names
        
        # for name in selection_names:
        #     if name not in self._scoring_names:
        #         self._scoring_names.append(name)

        self._scoring_admet_names = scoring_admet_names
    
        # store selection names and check that they are subset of scoring names
        self._selection_names = selection_names

        # # check whether scoring names are in self._name_to_function
        # for name in scoring_names:
        #     if name not in self._name_to_function:
        #         raise KeyError('Error: ' + name + ' not an implemented scoring function. Options are ' + ', '.join(self._name_to_function.keys()))
            
        # self._scoring_admet_names = scoring_admet_names
        # self._selection_admet_names = selection_admet_names
        # self._property_config = property_config


    def get_name_to_function_dict(self):
        """Get dictionary that maps string to scoring functions.

        Note:
            To add scoring functions, inherit from MoleculeScoring and override

        Returns:
            Dict[str, function] to relate names to scoring functions

        """
        return {
            'synth': self._synthetic_accessibility_with_default,
            'drug': self._qed_with_default,
            'logP': self._crippen_mol_logp_with_default,
            'number': self._number_with_default
        }

    def generate_scores(self, mols):
        """Generate scores for list of rdkit molecules.

        Args:
            mols (List[rdkit.Chem.rdchem.Mol]): List of rdkit molecules

        Returns:
            Dict[str, List[float]] where keys are the scoring function names and the values are the scores
            Also includes raw (unnormalized) values with suffix '_raw'
            Only includes properties that are in selection_names (used for fitness)

        """
        output = {}
        
        # Import SA scorer for clean SA score calculation
        from rdkit.Contrib.SA_Score import sascorer
        
        # Mapping for properties that need raw values
        raw_function_map = {
            'logP': lambda mol: Crippen.MolLogP(mol) if mol else -3.0,
            'synth': lambda mol: sascorer.calculateScore(mol) if mol else 10.0,
            'drug': self._qed_with_default,
            'number': self._number_with_default
        }

        # Apply scoring functions ONLY for properties in selection_names
        # This ensures only properties used for fitness appear in output
        for scoring_name in self._scoring_names:
            # Only include if it's in selection_names (used for fitness)
            if scoring_name in self._selection_names:
                output[scoring_name] = list(map(self._name_to_function[scoring_name], mols))
                # Get raw values if mapping exists
                if scoring_name in raw_function_map:
                    output[scoring_name + '_raw'] = list(map(raw_function_map[scoring_name], mols))
        
        # always return data as well
        output[self.data_column_name] = list(map(mol_to_canonical_smiles, mols))

        # return fitness
        output[self._fitness_column_name] = []
        for i in range(len(output[self.data_column_name])):
            fitness_array = [output[x][i] for x in self._selection_names]
            if len(fitness_array) > 0:
                output[self._fitness_column_name].append(self._fitness_function(fitness_array))
            else:
                output[self._fitness_column_name].append(-1.0)

        return output

    def prepare_data_for_scoring(self, smiles):
        """Prepare smiles str for scoring.

        Args:
            smiles (str): smiles string representation of a molecule

        Returns:
            rdkit.Chem.rdchem.Mol if valid smiles or None if not valid smiles

        """
        # remove any spaces
        smiles = smiles.replace(' ','')

        # only allow brackets from exempt list
        all_brackets = re.findall(r'\[.*?\]', smiles)
        for bracket in all_brackets:
            if bracket not in self._exempt_brackets:
                return None

        # check for multiple molecules or wildcards
        if ('.' in smiles) or ('*' in smiles):
            return None

        # attempt conversion to molecule
        return smiles_to_mol(smiles)

    def make_canonical(self, mol):
        """Generate canonical smiles string for molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): input molecule

        Returns: 
            str of canonical smiles (or None if not valid)

        """
        return mol_to_canonical_smiles(mol)

    @property
    def column_names(self):
        """Get list of names used in scoring dictionary.

        Returns:
            List[str] of names used in scoring dictionary (including raw values)
            Only includes properties that are in selection_names (used for fitness)

        """
        # Start with data column name
        base_names = [self._data_column_name]
        
        # Add only selection names (properties actually used for fitness)
        base_names.extend([name for name in self._scoring_names if name in self._selection_names])
        
        # Add raw value columns for selection names only
        raw_names = [name + '_raw' for name in self._scoring_names if name in self._selection_names]
        
        # Add ADMET names if present (always included if specified)
        if self._scoring_admet_names is not None:
            admet_raw = [name + '_raw' for name in self._scoring_admet_names]
            all_names = base_names + raw_names + admet_raw + list(self._scoring_admet_names) + [self._fitness_column_name]
        else:
            all_names = base_names + raw_names + [self._fitness_column_name]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in all_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        return unique_names

    @property
    def selection_names(self):
        """Get list of names used for selection.

        Returns:
            List[str] of names using for selection (i.e. fitness) scoring

        """
        return self._selection_names

    @property
    def data_column_name(self):
        """Get name of data column.

        Returns:
            str with name of data column

        """
        return self._data_column_name

    @property
    def fitness_column_name(self):
        """Get name of fitness column.

        Returns:
            str with name of data column

        """
        return self._fitness_column_name

    def _qed_with_default(self, mol, default=0.0):
        """ Generate quantitative estimation of drug-likenss

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score
            default (float): Default value if scoring fails

        Returns:
            float score value

        """
        try:
            return QED.qed(mol)
        except:
            return default

    # octanol-water partition coefficient
    def _crippen_mol_logp_with_default(self, mol, default=-3.0, norm=True):
        """Generate Crippen MolLogP.

        Note:
            For remap values see https://github.com/nicola-decao/MolGAN/blob/master/utils/molecular_metrics.py

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score
            default (float): Default raw value if scoring fails
            norm (bool): Option to normalize output

        Returns:
            float score value

        """
        try:
            score = Crippen.MolLogP(mol)
        except:
            score = default
        if norm:
            score = np.clip(remap(score, -2.12178879609, 6.0429063424), 0.0, 1.0)
        return score

    def _synthetic_accessibility_with_default(self, mol, default=10, norm=True):
        """Generate synthesizability score using RDKit's SA scorer.

        Note:
            RDKit's SA scorer returns [1-10] where 1=easy to synthesize, 10=hard
            We invert this so higher values = better (easier to synthesize)

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score
            default (float): Default raw value if scoring fails
            norm (bool): Option to normalize output

        Returns:
            float score value

        """
        try:
            # Use RDKit's built-in SA scorer
            from rdkit.Contrib.SA_Score import sascorer
            score = sascorer.calculateScore(mol)
            
            # RDKit SA score: [1-10] where 1=easy, 10=hard
            # Invert and normalize: [1-10] -> [10-1] -> [0-1]
            if norm:
                score = np.clip((10 - score) / 9.0, 0.0, 1.0)
        except:
            score = default
            if norm:
                score = 0.0  # Default to 0 if normalization requested
        return score

    def _number_with_default(self, mol, default=0):
        """Generate number of atoms.

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score
            default (float): Default raw value if scoring fails

        Returns:
            float score value

        """
        try:
            score = mol.GetNumAtoms()
        except:
            score = default
        return score


# some sample use cases
if __name__ == '__main__':
    print('Example of using molecule scoring\n', flush=True)

    # construct MoleculeScoring object
    metrics = ['drug', 'sol', 'number']
    selection_metrics = ['drug']
    metric_parameters = {}
    molecule_scoring = MoleculeScoring(metrics, selection_metrics, metric_parameters)

    # example smiles    
    smiles_examples = ['c1ccccc1', 'OCc1ccccc1', 'Brc1ccccc1C2CCCC2', 'junk']

    # prepare data for scoring
    molecule_examples = []
    for s in smiles_examples:
        mol = molecule_scoring.prepare_data_for_scoring(s)
        if mol is not None:
            molecule_examples.append(mol)

    # score data
    scores = molecule_scoring.generate_scores(molecule_examples)

    # number of samples and column names
    number_of_samples = len(scores[molecule_scoring.data_column_name])
    column_names = molecule_scoring.column_names

    # print column names
    for key in column_names:
        print(key, end='\t')
    print()

    # print data
    for i in range(number_of_samples):
        for key in column_names:
            print(scores[key][i], end='\t')
        print()



