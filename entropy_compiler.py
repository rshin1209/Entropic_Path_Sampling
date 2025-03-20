import os
import sys
import math
import argparse
import logging
from pathlib import Path
from typing import List, Set, Tuple, Optional

from tqdm import tqdm, trange  # Correctly import tqdm and trange
import numpy as np
from numba import njit, prange

# Constant: Boltzmann constant in kcal/(mol*K)
kB = 1.987204259e-3

DOF_TYPE_MAPPING = {
    2: 0,  # bond
    3: 1,  # angle
    4: 2   # torsion
}

class EntropyProcessor:
    """
    A class to process entropy calculations based on mutual information matrices.
    """
    def __init__(
        self,
        reaction: str,
        temperature: float,
        atoms: Optional[str],
        log_level: str = "INFO"
    ):
        """
        Initializes the EntropyProcessor with necessary parameters.

        :param reaction: Name of the reaction.
        :param temperature: Temperature in Kelvin.
        :param atoms: String specifying atom indices (e.g., '1,2,5-10').
        :param log_level: Logging level.
        """
        self.reaction = reaction
        self.temperature = temperature
        self.atoms_input = atoms
        self.log_level = log_level

        self.data_dir = Path("./output") / self.reaction
        self.topology_path = self.data_dir / "topology.txt"
        self.em_files = sorted(self.data_dir.glob("em_*.npy"))

        self.setup_logging()
        self.validate_paths()
        self.allowed_atoms = self.parse_atoms(self.atoms_input)
        self.dof_list, self.atom_count = self.load_topology()
        self.jtype_list = self.compute_jtype_list()
    
    def setup_logging(self) -> None:
        """
        Configures the logging settings.
        """
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {self.log_level}")
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.data_dir / "entropy_log.txt", mode='w')
            ]
        )
        logging.info(f"Logging initialized at {self.log_level} level.")

    def validate_paths(self) -> None:
        """
        Validates the existence of required files and directories.
        """
        if not self.data_dir.exists():
            logging.error(f"Data directory does not exist: {self.data_dir}")
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        if not self.topology_path.exists():
            logging.error(f"topology.txt not found in {self.data_dir}")
            raise FileNotFoundError(f"topology.txt not found in {self.data_dir}")

        if not self.em_files:
            logging.error(f"No 'em_*.npy' files found in {self.data_dir}")
            raise FileNotFoundError(f"No 'em_*.npy' files found in {self.data_dir}")

        logging.info(f"Found {len(self.em_files)} 'em_*.npy' files for processing.")

    @staticmethod
    def parse_atom_range(atom_string: Optional[str]) -> Set[int]:
        """
        Parses a string of atom indices and ranges into a set of integers.

        :param atom_string: String specifying atom indices (e.g., '1,2,5-10').
        :return: Set of 1-based atom indices.
        """
        if not atom_string:
            return set()  # Empty set signifies all atoms
        result = set()
        parts = atom_string.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                start, end = int(start), int(end)
                if start > end:
                    start, end = end, start
                result.update(range(start, end + 1))
            else:
                result.add(int(part))
        logging.info(f"Parsed atom indices: {sorted(result)}")
        return result

    def parse_atoms(self, atom_string: Optional[str]) -> Set[int]:
        """
        Determines the set of atom indices to include based on user input.

        :param atom_string: String specifying atom indices (e.g., '1,2,5-10').
        :return: Set of 1-based atom indices to include.
        """
        user_atoms = self.parse_atom_range(atom_string)
        if not user_atoms:
            # If no atoms specified, include all atoms
            logging.info("No specific atom indices provided. Including all atoms.")
            return set()
        else:
            return user_atoms

    def load_topology(self) -> Tuple[List[List[str]], int]:
        """
        Loads the topology information from 'topology.txt'.

        :return: A tuple containing:
                 - A list of DOFs, each represented as a list of atom indices.
                 - Total number of atoms.
        """
        with self.topology_path.open("r") as file:
            lines = [line.strip() for line in file if line.strip()]
        
        if not lines:
            logging.error("topology.txt is empty.")
            raise ValueError("topology.txt is empty.")

        atom_count = int(lines[0])
        dof_lines = lines[1:]
        logging.info(f"Loaded topology with {len(dof_lines)} DOFs and {atom_count} atoms.")

        # Filter DOFs based on allowed atoms
        if self.allowed_atoms:
            kept_indices = []
            kept_dof_types = []
            for i, line in enumerate(dof_lines):
                tokens = line.split()
                atoms_in_dof = list(map(int, tokens))
                if all(atom in self.allowed_atoms for atom in atoms_in_dof):
                    kept_indices.append(i)
                    dof_type = self.classify_dof(tokens)
                    kept_dof_types.append(dof_type)
            logging.info(f"Original DOFs: {len(dof_lines)}; Kept DOFs: {len(kept_indices)} based on specified atoms.")
            self.kept_indices = kept_indices
            self.kept_dof_types = kept_dof_types
        else:
            # If no atoms specified, keep all DOFs
            kept_dof_types = [self.classify_dof(line.split()) for line in dof_lines]
            logging.info(f"Keeping all {len(dof_lines)} DOFs.")
            self.kept_indices = list(range(len(dof_lines)))
            self.kept_dof_types = kept_dof_types

        # Load the DOFs based on kept_indices
        kept_dof_list = [dof_lines[i].split() for i in self.kept_indices]
        return kept_dof_list, atom_count

    @staticmethod
    def classify_dof(line_tokens: List[str]) -> int:
        """
        Determines the type of DOF based on the number of atoms involved.

        :param line_tokens: List of atom indices as strings.
        :return: Integer representing DOF type (0: bond, 1: angle, 2: torsion).
        """
        n = len(line_tokens)
        return DOF_TYPE_MAPPING.get(n, 0)  # Default to 'bond' if unknown

    def compute_jtype_list(self) -> np.ndarray:
        """
        Creates a numpy array of DOF types.

        :return: Numpy array of DOF types.
        """
        jtype_list = np.array(self.kept_dof_types, dtype=np.int32)
        logging.info(f"Computed DOF types for {len(jtype_list)} DOFs.")
        return jtype_list

    def process_all_em_files(self) -> List[Tuple[float, float, float]]:
        """
        Processes all 'em_*.npy' files to compute entropy.

        :return: List of tuples containing (upper, lower, entropy).
        """
        results = []
        for em_file in tqdm(self.em_files, desc="Processing 'em_*.npy' files"):
            # Debugging: Verify the type of em_file
            if not isinstance(em_file, Path):
                logging.error(f"Expected Path object, got {type(em_file)}. Skipping.")
                continue

            # Parse upper and lower bond values from filename
            filename = em_file.name
            try:
                parts = filename.rstrip('.npy').split('_')
                if len(parts) < 3:
                    raise ValueError("Filename does not contain enough parts.")
                upper = float(parts[1])
                lower = float(parts[2])
            except (IndexError, ValueError) as e:
                logging.error(f"Filename {filename} does not match pattern 'em_upper_lower.npy'. Skipping.")
                continue

            # Load MI matrix
            try:
                em_matrix = np.load(em_file)
            except Exception as e:
                logging.error(f"Failed to load {filename}: {e}. Skipping.")
                continue

            if em_matrix.shape[0] != em_matrix.shape[1]:
                logging.error(f"Matrix in {filename} is not square. Skipping.")
                continue
            if em_matrix.shape[0] != len(self.kept_indices):
                logging.error(f"Matrix dimension in {filename} ({em_matrix.shape[0]}) does not match number of kept DOFs ({len(self.kept_indices)}). Skipping.")
                continue

            # Compute entropy
            try:
                entropy = self.compute_entropy(em_matrix)
            except Exception as e:
                logging.error(f"Failed to compute entropy for {filename}: {e}. Skipping.")
                continue

            results.append((upper, lower, entropy))
            logging.info(f"Computed entropy for {filename}: {entropy:.4f} kcal/mol")

        return results

    def compute_entropy(self, em_matrix: np.ndarray) -> float:
        """
        Computes the final entropy based on the MI matrix.

        :param em_matrix: Mutual Information (MI) matrix.
        :return: Final entropy value in kcal/mol.
        """
        # Sum diagonal entropies for each DOF type
        S1D_bond = 0.0
        S1D_angle = 0.0
        S1D_torsion = 0.0
        for i, dof_type in enumerate(self.kept_dof_types):
            eii = em_matrix[i, i]
            if dof_type == 0:
                S1D_bond += eii
            elif dof_type == 1:
                S1D_angle += eii
            else:
                S1D_torsion += eii

        # Determine the number of atoms
        if self.allowed_atoms:
            atom_num = len(self.allowed_atoms)
        else:
            atom_num = self.atom_count  # Total number of atoms from topology

        logging.debug(f"Atom count for entropy refactor: {atom_num}")

        # Refactor S1D
        if atom_num > 0:
            if self.count_dof_type(0) > 0 and (atom_num - 1) > 0:
                S1D_bond *= (atom_num - 1) / self.count_dof_type(0)
            if self.count_dof_type(1) > 0 and (atom_num - 2) > 0:
                S1D_angle *= (atom_num - 2) / self.count_dof_type(1)
            if self.count_dof_type(2) > 0 and (atom_num - 3) > 0:
                S1D_torsion *= (atom_num - 3) / self.count_dof_type(2)
        else:
            logging.warning("Atom count is undefined. Skipping S1D refactor.")

        # Compute MIST entropy
        MIST_bond = 0.0
        MIST_angle = 0.0
        MIST_torsion = 0.0
        for i, dof_type in enumerate(self.kept_dof_types):
            row = em_matrix[i, i:].copy()
            max_off = np.max(row)
            if dof_type == 0:
                MIST_bond += max_off
            elif dof_type == 1:
                MIST_angle += max_off
            else:
                MIST_torsion += max_off

        # Refactor MIST
        if atom_num > 0:
            if self.count_dof_type(0) > 0 and (atom_num - 1) > 0:
                MIST_bond *= (atom_num - 1) / self.count_dof_type(0)
            if self.count_dof_type(1) > 0 and (atom_num - 2) > 0:
                MIST_angle *= (atom_num - 2) / self.count_dof_type(1)
            if self.count_dof_type(2) > 0 and (atom_num - 3) > 0:
                MIST_torsion *= (atom_num - 3) / self.count_dof_type(2)
        else:
            logging.warning("Atom count is undefined. Skipping MIST refactor.")

        # Final entropy
        total_S1D = S1D_bond + S1D_angle + S1D_torsion
        total_MIST = MIST_bond + MIST_angle + MIST_torsion
        final_entropy = -self.temperature * kB * (total_S1D - total_MIST)

        logging.debug(f"S1D: {total_S1D}, MIST: {total_MIST}, Final Entropy: {final_entropy}")

        return final_entropy

    def count_dof_type(self, dof_type: int) -> int:
        """
        Counts the number of DOFs of a specific type.

        :param dof_type: DOF type (0: bond, 1: angle, 2: torsion).
        :return: Count of DOFs of the specified type.
        """
        return sum(1 for dt in self.kept_dof_types if dt == dof_type)

    def sort_results(self, results: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """
        Sorts the results in decreasing order of the upper bond component and computes changes in entropy.

        :param results: List of tuples containing (upper, lower, entropy).
        :return: Sorted list of tuples containing (upper, lower, entropy, delta_entropy).
        """
        # Sort by upper bond in decreasing order
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        
        # Compute change in entropy relative to the first point
        if not sorted_results:
            logging.warning("No results to sort.")
            return []
        
        reference_entropy = sorted_results[0][2]
        sorted_with_delta = []
        for idx, (upper, lower, entropy) in enumerate(sorted_results):
            if idx == 0:
                delta = 0.0
            else:
                delta = entropy - reference_entropy
            sorted_with_delta.append((upper, lower, entropy, delta))
        
        return sorted_with_delta

    def log_results(self, sorted_results: List[Tuple[float, float, float, float]]) -> None:
        """
        Logs the entropy results in the specified format.

        :param sorted_results: Sorted list of tuples containing (upper, lower, entropy, delta_entropy).
        """
        header = "Upper\tLower\tMIST Entropy\tChange in MIST Entropy"
        logging.info(header)
        print(header)
        for upper, lower, entropy, delta in sorted_results:
            log_line = f"{upper:.4f}\t{lower:.4f}\t{entropy:.4f}\t{delta:.4f}"
            logging.info(log_line)
            print(log_line)

    def run(self) -> None:
        """
        Executes the entropy processing workflow.
        """
        # Process all 'em_*.npy' files and compute entropy
        results = self.process_all_em_files()

        if not results:
            logging.warning("No entropy results were computed. Exiting.")
            return

        # Sort results and compute delta entropy
        sorted_results = self.sort_results(results)

        # Log the sorted results
        self.log_results(sorted_results)

        logging.info("Entropy processing completed.")

def main():
    parser = argparse.ArgumentParser(
        description="Compute Entropy from 'em_*.npy' MI matrices based on specified atom indices."
    )
    parser.add_argument(
        "--reaction",
        type=str,
        required=True,
        help="Name of the reaction. Expects './output/{reaction}/dofs.npy' and './output/{reaction}/topology.txt'."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Temperature in Kelvin."
    )
    parser.add_argument(
        "--atoms",
        type=str,
        default=None,
        help="Atom indices (1-based) to include, e.g., '1,2,5-10'. If omitted, all atoms are considered."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    args = parser.parse_args()

    processor = EntropyProcessor(
        reaction=args.reaction,
        temperature=args.temperature,
        atoms=args.atoms,
        log_level=args.log_level
    )
    processor.run()

if __name__ == "__main__":
    main()
