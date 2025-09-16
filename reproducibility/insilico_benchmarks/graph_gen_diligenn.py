#!/usr/bin/env python3
"""
DILIGeNN Graph Generation and Data Processing Script

This script contains utilities for processing DILI (Drug-Induced Liver Injury) data
for the DILIGeNN model, including molecular standardization, graph generation for
graph neural network models, and data cleaning pipelines. It provides custom PyTorch
Geometric dataset classes for converting molecular SMILES to graph representations
with comprehensive node and edge features optimized for DILIGeNN's architecture.

Key components:
- Molecular standardization using RDKit for DILIGeNN preprocessing
- Custom dataset classes for DILIGeNN graph neural networks
- DILI data processing and cleaning functions for DILIGeNN training
- Graph feature extraction (node and edge features) tailored for DILIGeNN
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.MolStandardize import rdMolStandardize

import torch
from torch_geometric.data import Data, InMemoryDataset


def standardise_mol(mol):
    # Removing Hs, disconnect metal atoms, normalise the molecule, reionise the molecule
    clean_mol = rdMolStandardize.Cleanup(mol)
    # If more than one fragments, select parent (bigger) molecule
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    # Neutralise molecule
    uncharger = rdMolStandardize.Uncharger()
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    # Enumerate tautomers
    te = rdMolStandardize.TautomerEnumerator()
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    return taut_uncharged_parent_clean_mol


def standardise_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        std_mol = standardise_mol(mol)
        if std_mol is None:
            return None
        else:
            return Chem.MolToSmiles(std_mol)
    except Exception:
        return None


def select_and_standardise(
    df,
    smiles_col='smiles',
    error_msg_col='error_msg',
    std_smiles_col='standardised_smiles',
    common_elements={1, 6, 7, 8, 9, 15, 16, 17, 35, 53},
):
    error_msgs = []
    standardised_smiles = []

    for smiles in tqdm(df[smiles_col], desc='Processing molecules'):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                error_msgs.append('Invalid SMILES')
                standardised_smiles.append(None)
                continue
            invalid_element_found = False
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in common_elements:
                    error_msgs.append(
                        f'Contains uncommon element: {atom.GetAtomicNum()}'
                    )
                    invalid_element_found = True
                    standardised_smiles.append(None)
                    break
            if not invalid_element_found:
                error_msgs.append('Success')
                std_smiles = standardise_smiles(smiles)
                standardised_smiles.append(std_smiles)
        except Exception as e:
            error_msgs.append(str(e))
            standardised_smiles.append(None)

    df[error_msg_col] = error_msgs
    df[std_smiles_col] = standardised_smiles
    return df


def graph_to_mol(data):
    mol = Chem.RWMol()
    node_features = data.x
    edge_indices = data.edge_index.t().numpy()
    edge_attrs = data.edge_attr.numpy()

    atom_indices = []
    for node_feat in node_features:
        atomic_num = int(node_feat[0].item())
        atom = Chem.Atom(atomic_num)
        idx = mol.AddAtom(atom)
        atom_indices.append(idx)

    for i, (start, end) in enumerate(edge_indices):
        if start < end:  # To avoid adding duplicate bonds
            bond_type = Chem.BondType.SINGLE
            if edge_attrs[i][0] == 1:
                bond_type = Chem.BondType.SINGLE
            elif edge_attrs[i][0] == 2:
                bond_type = Chem.BondType.DOUBLE
            elif edge_attrs[i][0] == 3:
                bond_type = Chem.BondType.TRIPLE
            elif edge_attrs[i][0] == 1.5:
                bond_type = Chem.BondType.AROMATIC

            try:
                mol.AddBond(int(start), int(end), bond_type)
            except Exception as e:
                print(f'Error adding bond between atoms {start} and {end}: {e}')

    mol = mol.GetMol()
    # mol = Chem.RemoveHs(mol)  # Remove hydrogen atoms explicitly added -> if not removed, causing problems in XAI methods
    return mol


def pytorch_to_df(dataset):
    smiles_list, label_list = [], []
    for molecule in dataset:
        smiles_list.append(molecule['smiles'])
        label_list.append(molecule['y'].item())
    df = pd.DataFrame({'smiles': smiles_list, 'label': label_list})
    return df


def get_dataset_info(dataset):
    return {
        'num_graphs': len(dataset),
        'label_shape': dataset.data.y.shape,
        'num_unique_labels': len(np.unique(dataset.data.y.tolist())),
        'num_node_features': dataset.num_node_features,
        'num_edge_features': dataset.num_edge_features
        if hasattr(dataset.data, 'edge_attr')
        else 0,
    }


class MoleculeDataset_minimal(InMemoryDataset):
    def __init__(
        self,
        dataframe,
        root,
        smiles_col='smiles',
        label_col='label',
        test=False,
        transform=None,
        pre_transform=None,
    ):
        """
        Basic Molecular Graph Generation Algorithm
        """
        self.test = test
        self.dataframe = dataframe
        self.smiles_col = smiles_col
        self.label_col = label_col
        self._data = None
        self.error_indices = []
        super(MoleculeDataset_minimal, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return 'dataframe'

    @property
    def processed_file_names(self):
        return ['data_test.pt' if self.test else 'data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for index, mol in tqdm(
            self.dataframe.iterrows(), total=self.dataframe.shape[0]
        ):
            try:
                mol_obj = Chem.MolFromSmiles(mol[self.smiles_col])

                # Get node features
                node_feats = self._get_node_features(mol_obj)
                # Get edge features
                edge_feats = self._get_edge_features(mol_obj)
                # Get adjacency info
                edge_index = self._get_adjacency_info(mol_obj)
                # Get labels info
                label = self._get_labels(mol[self.label_col])

                # Create data object
                data = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    edge_attr=edge_feats,
                    y=label,
                    smiles=mol[self.smiles_col],
                )
                data_list.append(data)
            except Exception as e:
                print(f'Error processing molecule at index {index}: {e}')
                self.error_indices.append(index)

        # Save all data objects into one file
        if self.test:
            torch.save(data_list, os.path.join(self.processed_dir, 'data_test.pt'))
        else:
            torch.save(data_list, os.path.join(self.processed_dir, 'data.pt'))

    def _get_node_features(self, mol):
        """
        Return a matrix / 2D array of the shape [Number of Nodes, Node Feature size]
        with atomic number as the only node feature.
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature: Atomic number
            node_feats.append(atom.GetAtomicNum())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """
        Return a matrix / 2D array of the shape [Number of edges, Edge Feature size]
        with bond type as the only edge feature.
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())

            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([[label]])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.dataframe.shape[0]

    def get(self, idx):
        if self._data is None:
            if self.test:
                self._data = torch.load(
                    os.path.join(self.processed_dir, 'data_test.pt')
                )
            else:
                self._data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return self._data[idx]


## Customised preprocessing (Mols-To-Graph function)


class MoleculeDataset_updated(InMemoryDataset):
    def __init__(
        self,
        dataframe,
        root,
        smiles_col='smiles',
        label_col='label',
        test=False,
        transform=None,
        pre_transform=None,
    ):  # modified
        """
        Force Field Optimised Graph Generation Algorithm
        """
        self.test = test
        self.dataframe = dataframe
        self.smiles_col = smiles_col  # modified
        self.label_col = label_col  # modified
        self._data = None  # Add this line to initialize the cache
        self.error_indices = []  # bad indices: To keep track of error indices
        super(MoleculeDataset_updated, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return 'dataframe'  # placeholder

    @property
    def processed_file_names(self):
        return ['data_test.pt' if self.test else 'data.pt']

    def download(self):
        pass

    def process(self):
        # self.data = pd.read_csv(self.raw_paths[0]) # we are not reading from dir
        data_list = []
        for index, mol in tqdm(
            self.dataframe.iterrows(), total=self.dataframe.shape[0]
        ):
            try:
                mol_obj = Chem.MolFromSmiles(mol[self.smiles_col])  # modified

                # below generates 3D force field optimised molecular structures
                mol_obj = Chem.AddHs(mol_obj)
                AllChem.EmbedMolecule(
                    mol_obj,
                    randomSeed=42,
                    useRandomCoords=True,
                    maxAttempts=5000,  # Use this when Bad Conformer ID error
                )
                AllChem.MMFFOptimizeMolecule(mol_obj)
                mol_obj = Chem.RemoveHs(
                    mol_obj
                )  # We have number of Hs node features already
                AllChem.ComputeGasteigerCharges(mol_obj)  # additional feature

                ################################################################
                # Get node features
                node_feats = self._get_node_features(mol_obj)
                # Get edge features
                edge_feats = self._get_edge_features(mol_obj)
                # Get adjacency info
                edge_index = self._get_adjacency_info(mol_obj)
                # Get labels info
                label = self._get_labels(mol[self.label_col])  # modified

                # Create data object
                data = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    edge_attr=edge_feats,
                    y=label,
                    smiles=mol[self.smiles_col],  # modified
                )
                data_list.append(data)
            except Exception as e:
                print(f'Error processing molecule at index {index}: {e}')
                self.error_indices.append(
                    index
                )  # bad_indices: Track the index of the molecule that caused an error

        # Save all data objects into one file
        if self.test:
            torch.save(data_list, os.path.join(self.processed_dir, 'data_test.pt'))
        else:
            torch.save(data_list, os.path.join(self.processed_dir, 'data.pt'))

    def _get_node_features(self, mol):
        """
        Return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree -> Number of directly-bonded neighbours
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge -> charge of the atom
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization -> hybridization state i.e. sp3
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Feature 10: Gasteiger Charges
            node_feats.append(atom.GetDoubleProp('_GasteigerCharge'))
            # Feature 11: Total Valence
            node_feats.append(atom.GetTotalValence())
            # Feature 12: Explicit Valence
            node_feats.append(atom.GetExplicitValence())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """
        Return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        conf = (
            mol.GetConformer()
        )  # will be used to calculate bond length, force field optimised conformer
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Bond length
            edge_feats.append(
                np.round(
                    rdMolTransforms.GetBondLength(
                        conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    ),
                    3,
                )
            )
            # Feature 3: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([[label]])  # or just np.asarray([[label]])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.dataframe.shape[0]

    def get(self, idx):
        if self._data is None:  # Check if data is already loaded
            if self.test:
                self._data = torch.load(
                    os.path.join(self.processed_dir, 'data_test.pt')
                )
            else:
                self._data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return self._data[idx]

    # we retrieved SMILES from pubchem


# df_DILI_fail: failed to retrieve SMILES molecules, see GNN_benchmark_DILI2


def process_dili_data(df, smiles_col='smiles'):
    """Process DILI data: standardize and clean"""
    RDLogger.DisableLog('rdApp.*')

    initial_count = df.shape[0]
    print(f'Initial dataset size: {initial_count}')

    # Check for missing SMILES
    df_DILI_fail = df[df[smiles_col].isna()]
    missing_smiles_count = df_DILI_fail.shape[0]
    print(
        f'Compounds with missing SMILES: {missing_smiles_count} ({missing_smiles_count / initial_count * 100:.1f}%)'
    )

    df_DILI = df.dropna(subset=[smiles_col])
    after_smiles_filter = df_DILI.shape[0]
    print(f'After removing missing SMILES: {after_smiles_filter}')

    # Standardize SMILES
    df_DILI = select_and_standardise(
        df=df_DILI,
        smiles_col=smiles_col,
        error_msg_col='error_msg',
        std_smiles_col='smiles_std',
    )

    return df_DILI


def clean_dili_data(df, smiles_col='smiles', label_col='label'):
    """Clean DILI data by removing problematic molecules"""
    initial_count = df.shape[0]
    print(f'Standardised dataset size: {initial_count}')

    # Filter by standardization success
    df_DILI_filtered = df[df['error_msg'] == 'Success']
    after_standardization = df_DILI_filtered.shape[0]
    standardization_failures = initial_count - after_standardization
    print(
        f'Compounds failed standardization: {standardization_failures} ({standardization_failures / initial_count * 100:.1f}%)'
    )
    print(f'After standardization filtering: {after_standardization}')

    from pathlib import Path

    # Path to diligenn data folder
    HERE = Path(__file__).resolve().parent
    root = HERE / 'data' / 'diligenn'

    # Create dataset to identify error indices (conformer generation failures)
    dili_cus_prerun = MoleculeDataset_minimal(
        dataframe=df_DILI_filtered,
        smiles_col=smiles_col,
        label_col=label_col,
        root=root,
    )
    error_indices = dili_cus_prerun.error_indices  # bad_indices: Get the error indices
    conformer_failures = len(error_indices)
    print(
        f'Compounds failed conformer generation: {conformer_failures} ({conformer_failures / after_standardization * 100:.1f}%)'
    )
    print(f'Error indices: {error_indices}')

    # df_dili_filtered -> got rid of uncommon elements
    # df_dili_filtered_cleaned -> got rid of bad conformers
    df_DILI_filtered_cleaned = df_DILI_filtered.drop(
        index=error_indices
    )  # bad_indices: Drop the error indices
    final_count = df_DILI_filtered_cleaned.shape[0]

    print(f'Final cleaned dataset size: {final_count}')
    print(
        f'Total compounds filtered out: {initial_count - final_count} ({(initial_count - final_count) / initial_count * 100:.1f}%)'
    )

    return df_DILI_filtered_cleaned
