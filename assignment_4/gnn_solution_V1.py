import torch
import torch.nn as nn
import torch_geometric as torch_geom
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch.utils.data import Subset
import torch_geometric.nn as geom_nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data


def one_hot_encode(x: str, permitted_list: list[str]) -> list[int]:
    if x not in permitted_list:
        x = permitted_list[-1]
    return [
        int(boolean_value)
        for boolean_value in list(map(lambda s: x == s, permitted_list))
    ]


def get_atom_features(
    atom, use_chirality: bool = True, hydrogens_implicit: bool = True
) -> np.ndarray:
    permitted_list_of_atoms = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
        "Unknown",
    ]
    if not hydrogens_implicit:
        permitted_list_of_atoms = ["H"] + permitted_list_of_atoms

    atom_type_enc = one_hot_encode(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encode(
        int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"]
    )
    formal_charge_enc = one_hot_encode(
        int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
    )
    hybridisation_type_enc = one_hot_encode(
        str(atom.GetHybridization()),
        ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
    )
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]
    vdw_radius_scaled = [
        float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)
    ]
    covalent_radius_scaled = [
        float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)
    ]
    atom_feature_vector = (
        atom_type_enc
        + n_heavy_neighbors_enc
        + formal_charge_enc
        + hybridisation_type_enc
        + is_in_a_ring_enc
        + is_aromatic_enc
        + atomic_mass_scaled
        + vdw_radius_scaled
        + covalent_radius_scaled
    )
    if use_chirality:
        chirality_type_enc = one_hot_encode(
            str(atom.GetChiralTag()),
            [
                "CHI_UNSPECIFIED",
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
                "CHI_OTHER",
            ],
        )
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encode(
            int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]
        )
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry: bool = True) -> np.ndarray:
    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type_enc = one_hot_encode(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    if use_stereochemistry:
        stereo_type_enc = one_hot_encode(
            str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        )
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def _smile_to_graph(smile: str, y: float) -> Data:
    # Convert smile to RDKit mol object
    mol = Chem.MolFromSmiles(smile)
    # Get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2 * mol.GetNumBonds()
    unrelated_smile = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smile)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
    # Construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)

    X = torch.tensor(X, dtype=torch.float)
    # Construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim=0)
    # Construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    for k, (i, j) in enumerate(zip(rows, cols)):
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

    EF = torch.tensor(EF, dtype=torch.float)
    # Construct label tensor
    y_tensor = torch.tensor(np.array([y]), dtype=torch.float)
    # Construct Pytorch Geometric data object and append to data list
    return Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor)


def smiles_to_data_list(smiles: list[str], y: list[float]) -> list[Data]:
    """
    Args:
    smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerical labels for the SMILES strings,
    e.g for our case, the E_LUMO, or HOMO_LUMO gap values.

    Returns:
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects
    which represent labeled molecular graphs.
    """
    data_list = []
    for smile_i, y_i in tqdm(
        zip(smiles, y), desc="Extracting SMILE Graphs...", total=len(y)
    ):
        data_list.append(_smile_to_graph(smile_i, y_i))

    return data_list


class SMILESGraphDataset(InMemoryDataset):
    def __init__(self, root="./datasets/", mode="pretrain"):
        assert mode in ("pretrain", "finetune")
        self.mode = mode
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.mode}_features.csv", f"{self.mode}_labels.csv"]

    @property
    def processed_file_names(self):
        return [f"{self.mode}_smiles_graph_dataset.pt"]

    @property
    def processed_paths(self):
        return [
            os.path.join(self.processed_dir, processed_file_name)
            for processed_file_name in self.processed_file_names
        ]

    def process(self):
        features = pd.read_csv(f"{self.mode}_features.csv")
        labels = pd.read_csv(f"{self.mode}_labels.csv")
        df = pd.merge(features, labels, left_on="Id", right_on="Id", how="left")
        y_str = "lumo_energy" if self.mode == "pretrain" else "homo_lumo_gap"
        data_list = smiles_to_data_list(smiles=df["smiles"], y=df[y_str])
        data, slices = self.collate(data_list)
        torch.save(
            (data, slices),
            os.path.join(self.processed_dir, self.processed_file_names[0]),
        )


class IsaacGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        self.conv1 = geom_nn.SAGEConv(
            in_channels=-1,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
        )
        self.conv2 = geom_nn.SAGEConv(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
        )
        self.batch_norm = geom_nn.BatchNorm(in_channels=hidden_channels)
        self.graph_norm = geom_nn.GraphNorm(in_channels=hidden_channels)
        self.dropout = nn.Dropout(self.dropout)
        self.linear_1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.linear_2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, batch_index):
        x = self.conv1(x, edge_index).relu()
        x = self.graph_norm(x)
        x = self.conv2(x, edge_index).relu()
        x = self.graph_norm(x)
        x = geom_nn.global_max_pool(x, batch_index)
        x = self.dropout(x)
        x = self.linear_1(x).relu()
        x = self.dropout(x)
        x = self.linear_2(x)
        return x.squeeze(-1)


def train_one_epoch(
    model, train_dataloader, loss_fn, optimiser, n_iter, writer, n_epoch, device
):
    model.to(device)
    model.train()
    batch_losses = []
    tqdm_iter = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch: {n_epoch}",
    )
    for k, batch in tqdm_iter:
        x, y, edge_index, batch_index = batch.x, batch.y, batch.edge_index, batch.batch
        x.to(device)
        y.to(device)
        edge_index.to(device)
        batch_index.to(device)
        optimiser.zero_grad()
        output = model(x=x, edge_index=edge_index, batch_index=batch_index)
        loss = torch.sqrt(loss_fn(output, y))
        loss.backward()
        optimiser.step()
        batch_loss = loss.item()
        if k % 30 == 0:  # Add the batch loss to the progress bar every 30 batches
            tqdm_iter.set_postfix_str(f"Batch loss: {batch_loss:.4f}")
        batch_losses.append(batch_loss)

    epoch_avg_loss = np.mean(np.array(batch_losses))
    writer.add_scalar("RMSE Loss/train", epoch_avg_loss, n_iter)
    print("[RMSE Loss/train]:", epoch_avg_loss)


def get_val_perf(model, val_dataloader, loss_fn, n_iter, writer):
    """Evaluate Model on Validation Set."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        val_losses = []
        for k, batch in enumerate(val_dataloader):
            x, y, edge_index, batch_index = (
                batch.x,
                batch.y,
                batch.edge_index,
                batch.batch,
            )
            x.to(device)
            y.to(device)
            edge_index.to(device)
            batch_index.to(device)
            output = model(x=x, edge_index=edge_index, batch_index=batch_index)
            val_losses.append(np.sqrt(loss_fn(output, batch.y)))

        avg_val_loss = np.mean(np.array(val_losses))
        writer.add_scalar("RMSE Loss/val", avg_val_loss, n_iter)
        print("[RMSE Loss/val]:", avg_val_loss)


if __name__ == "__main__":
    random_state = 420
    batch_size = 2**5
    n_epochs = 100
    dropout = 0.5
    hidden_channels = 400
    num_layers = 3
    learning_rate = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = IsaacGNN(hidden_channels=hidden_channels, out_channels=1, dropout=dropout)
    model.to(device)

    dataset = SMILESGraphDataset(mode="pretrain")
    print(f"Loaded pretraining dataset: {dataset}!")

    # Train test split our pretraining data
    train_idx, test_idx, _, _ = train_test_split(
        range(len(dataset)),
        range(len(dataset)),
        test_size=0.05,
        random_state=random_state,
    )
    train_idx, val_idx, _, _ = train_test_split(
        range(len(train_idx)),
        range(len(train_idx)),
        test_size=0.1,
        random_state=random_state,
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    print(
        f"Loaded pretraining train: {len(train_loader.dataset)}, "
        + f"val: {len(val_loader.dataset)} "
        + f"and test: {len(test_loader.dataset)} dataloaders!"
    )

    writer = SummaryWriter()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    n_iter = 0
    print("Starting pretraining training...")
    for epoch in range(n_epochs):
        print("=" * 40 + f"> Epoch: {epoch} / {n_epochs} <" + "=" * 40)
        train_one_epoch(
            model=model,
            train_dataloader=train_loader,
            loss_fn=loss_fn,
            n_iter=n_iter,
            writer=writer,
            optimiser=optimiser,
            n_epoch=epoch,
            device=device,
        )
        get_val_perf(
            model=model,
            val_dataloader=val_loader,
            loss_fn=loss_fn,
            n_iter=n_iter,
            writer=writer,
            device=device,
        )

        n_iter += 1

    writer.flush()
    writer.close()
