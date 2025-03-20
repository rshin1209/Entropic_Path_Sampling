import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import mdtraj as md
import numpy as np


class XYZ2BATConverter:
    """
    A class to convert XYZ coordinates to internal coordinates (BAT) for a small molecule reaction.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        atom1: int,
        atom2: int,
        nb1: Optional[int] = None,
        nb2: Optional[int] = None,
    ):
        """
        Initializes the converter with input and output directories and reaction parameters.

        :param input_dir: Path to the input directory containing .xyz and .pdb files.
        :param output_dir: Path to the output directory where results will be saved.
        :param atom1: 1-indexed atom index for the reacting bond (required).
        :param atom2: 1-indexed atom index for the reacting bond (required).
        :param nb1: 1-indexed atom index for bond formation (optional).
        :param nb2: 1-indexed atom index for bond formation (optional).
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.atom1 = atom1 - 1  # Convert to 0-based index
        self.atom2 = atom2 - 1  # Convert to 0-based index
        self.nb1 = nb1 - 1 if nb1 is not None else None  # Convert to 0-based index
        self.nb2 = nb2 - 1 if nb2 is not None else None  # Convert to 0-based index

        self.xyz_files = sorted(self.input_dir.glob("*.xyz"))
        self.pdb_files = list(self.input_dir.glob("*.pdb"))

        if not self.xyz_files:
            logging.error(f"No .xyz files found in {self.input_dir}")
            raise FileNotFoundError(f"No .xyz files found in {self.input_dir}")

        if len(self.pdb_files) != 1:
            logging.error(f"Expected exactly one .pdb file in {self.input_dir}, found {len(self.pdb_files)}")
            raise FileNotFoundError(f"Expected exactly one .pdb file in {self.input_dir}, found {len(self.pdb_files)}")

        self.pdb_file = self.pdb_files[0]
        logging.info(f"Using topology file: {self.pdb_file.name}")

        # Load topology from PDB
        try:
            self.topology = md.load_pdb(str(self.pdb_file)).topology
            logging.info("Topology successfully loaded from PDB.")
        except Exception as e:
            logging.error(f"Failed to load PDB file {self.pdb_file}: {e}")
            raise e

        # Generate connectivity graph
        self.connectivity = self.generate_connectivity()
        if self.nb1 is not None and self.nb2 is not None:
            self.connectivity[self.nb1].append(self.nb2)
            self.connectivity[self.nb2].append(self.nb1)
            logging.info(f"Bond formation added between atoms {self.nb1 + 1} and {self.nb2 + 1}.")
        self.atom_count, self.topology_paths = self.generate_topology()

        self.save_topology()

        self.dofs_list = []  # To accumulate degrees of freedom from all trajectories

    def generate_connectivity(self) -> Dict[int, List[int]]:
        """
        Generates a connectivity graph from the PDB topology bonds.

        :return: A dictionary representing the connectivity graph.
        """
        graph = {}
        for bond in self.topology.bonds:
            index1, index2 = bond.atom1.index, bond.atom2.index
            graph.setdefault(index1, []).append(index2)
            graph.setdefault(index2, []).append(index1)
        logging.debug("Connectivity graph generated from PDB.")
        return graph

    def generate_topology(self) -> Tuple[int, List[Tuple[int, ...]]]:
        """
        Generates the topology from the connectivity graph using a custom DFS approach.

        :return: A tuple containing the number of atoms and a list of unique paths.
        """
        unique_paths: Set[Tuple[int, ...]] = set()
        max_path_length = 4  # Maximum path length to consider

        def dfs(current_path: List[int]):
            if len(current_path) > max_path_length:
                return
            if len(current_path) >= 2:
                # Add path in canonical form to avoid duplicates
                path = tuple(current_path)
                reversed_path = tuple(reversed(current_path))
                if reversed_path not in unique_paths:
                    unique_paths.add(path)
            last_node = current_path[-1]
            for neighbor in self.connectivity.get(last_node, []):
                if neighbor not in current_path:
                    current_path.append(neighbor)
                    dfs(current_path)
                    current_path.pop()

        for node in self.connectivity:
            dfs([node])

        sorted_unique_paths = sorted(unique_paths, key=lambda x: (len(x), x))
        atom_count = len(self.connectivity)
        logging.debug(f"Topology generated with {atom_count} atoms and {len(sorted_unique_paths)} paths.")
        return atom_count, sorted_unique_paths

    def save_topology(self) -> None:
        """
        Saves the topology to a text file, including bond formations if provided.
        """
        output_path = self.output_dir / "topology.txt"
        with output_path.open("w") as file:
            file.write(f"{self.atom_count}\n")
            for path in self.topology_paths:
                # Convert to 1-based indexing
                file.write(" ".join(str(atom + 1) for atom in path) + "\n")
        logging.info(f"Topology saved to {output_path}")

    @staticmethod
    def midpoint(xyz: np.ndarray) -> np.ndarray:
        """
        Applies the midpoint algorithm to smooth the trajectory coordinates.

        :param xyz: Numpy array of shape (n_frames, n_atoms, 3).
        :return: Smoothed coordinates as a numpy array.
        """
        acc = xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]
        vel = xyz[2:] - xyz[1:-1] - 0.5 * acc

        new_xyz = []
        for i in range(1, len(xyz) - 1):
            new_xyz.append(xyz[i])
            new_position = xyz[i] + vel[i - 1] * 0.5 + acc[i - 1] * 0.125
            new_xyz.append(new_position)

        # Handle the first and last frames
        first_mid = xyz[1] - vel[0] * 0.5 + acc[0] * 0.125
        new_xyz.insert(0, first_mid)
        new_xyz.insert(0, xyz[0])
        new_xyz.append(xyz[-1])

        return np.array(new_xyz)

    def midpoint_multiple(self, xyz: np.ndarray, num_mult: int) -> np.ndarray:
        """
        Applies the midpoint algorithm multiple times.

        :param xyz: Numpy array of shape (n_frames, n_atoms, 3).
        :param num_mult: Number of times to apply the midpoint algorithm.
        :return: Smoothed coordinates as a numpy array.
        """
        for _ in range(num_mult):
            xyz = self.midpoint(xyz)
        logging.debug(f"Midpoint algorithm applied {num_mult} times.")
        return xyz

    def process_xyz_file(self, xyz_file: Path, num_mult: int) -> np.ndarray:
        """
        Processes a single XYZ file to compute degrees of freedom.

        :param xyz_file: Path to the .xyz file.
        :param num_mult: Number of times to apply the midpoint algorithm.
        :return: Degrees of freedom as a numpy array.
        """
        try:
            trajectory = md.load_xyz(str(xyz_file), top=self.pdb_file)
            logging.info(f"Loaded trajectory from {xyz_file.name}")
        except Exception as e:
            logging.error(f"Failed to load {xyz_file}: {e}")
            raise e

        if num_mult > 0:
            trajectory.xyz = self.midpoint_multiple(trajectory.xyz, num_mult)
            logging.debug("Trajectory coordinates smoothed using midpoint algorithm.")

        # Compute reaction coordinate (distance between atom1 and atom2)
        reaction_coord = md.compute_distances(trajectory, [[self.atom1, self.atom2]]) * 10.0  # nm to angstrom
        logging.debug("Reaction coordinate computed.")

        # Compute internal coordinates
        bond_list = [path for path in self.topology_paths if len(path) == 2]
        angle_list = [path for path in self.topology_paths if len(path) == 3]
        torsion_list = [path for path in self.topology_paths if len(path) == 4]

        bonds = md.compute_distances(trajectory, bond_list) * 10.0  # nm to angstrom
        angles = md.compute_angles(trajectory, angle_list)
        dihedrals = md.compute_dihedrals(trajectory, torsion_list)

        # Concatenate reaction coordinate with degrees of freedom
        dofs = np.hstack((reaction_coord, bonds, angles, dihedrals))
        logging.debug(f"Degrees of freedom computed for {xyz_file.name}.")

        return dofs

    def run(self, num_mult: int) -> None:
        """
        Executes the conversion process for all .xyz files in the input directory.

        :param num_mult: Number of times to apply the midpoint algorithm.
        """
        for idx, xyz_file in enumerate(self.xyz_files, start=1):
            logging.info(f"Processing file {idx}/{len(self.xyz_files)}: {xyz_file.name}")
            try:
                dofs = self.process_xyz_file(xyz_file, num_mult)
                self.dofs_list.append(dofs)
            except Exception as e:
                logging.error(f"Error processing {xyz_file.name}: {e}")
                continue

        if self.dofs_list:
            all_dofs = np.vstack(self.dofs_list)
            output_file = self.output_dir / "dofs.npy"
            np.save(output_file, all_dofs)
            logging.info(f"All degrees of freedom saved to {output_file}")
        else:
            logging.warning("No degrees of freedom were processed. 'dofs.npy' not created.")


def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert XYZ coordinates to internal coordinates (BAT) for reactions.")
    parser.add_argument(
        "--reaction",
        type=str,
        required=True,
        help="Name of the reaction. The script expects ./dataset/{reaction} as input and ./output/{reaction} as output."
    )
    parser.add_argument(
        "--atom1",
        type=int,
        required=True,
        help="1-indexed atom index for the reacting bond (required)."
    )
    parser.add_argument(
        "--atom2",
        type=int,
        required=True,
        help="1-indexed atom index for the reacting bond (required)."
    )
    parser.add_argument(
        "--nb1",
        type=int,
        default=None,
        help="1-indexed atom index for bond formation (optional)."
    )
    parser.add_argument(
        "--nb2",
        type=int,
        default=None,
        help="1-indexed atom index for bond formation (optional)."
    )
    parser.add_argument(
        "--num_mult",
        type=int,
        default=0,
        help="Number of times to apply the midpoint algorithm for smoothing."
    )
    return parser.parse_args()


def main():
    """
    The main entry point of the script.
    """
    setup_logging()
    args = parse_arguments()

    input_dir = Path(f"./dataset/{args.reaction}")
    output_dir = Path(f"./output/{args.reaction}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    converter = XYZ2BATConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        atom1=args.atom1,
        atom2=args.atom2,
        nb1=args.nb1,
        nb2=args.nb2,
    )
    converter.run(num_mult=args.num_mult)


if __name__ == "__main__":
    main()
