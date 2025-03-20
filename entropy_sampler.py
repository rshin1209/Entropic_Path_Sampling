import sys
import math
import argparse
import logging
from pathlib import Path
from typing import List
import numpy as np
from tqdm import trange
from numba import njit, prange

DOF_TYPE_MAPPING = {
    "bond": 0,
    "angle": 1,
    "torsion": 2
}

def setup_logging(log_level: str = "INFO") -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform Entropic Path Sampling.")
    
    # 1. Renamed --system_name to --reaction
    parser.add_argument(
        "--reaction",
        type=str,
        required=True,
        help="Name of the reaction system."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory containing the data files."
    )
    parser.add_argument(
        "--bondmax",
        type=float,
        required=True,
        help="Maximum value of the reacting bond."
    )
    parser.add_argument(
        "--bondmin",
        type=float,
        required=True,
        help="Minimum value of the reacting bond."
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        required=True,
        help="Number of ensembles to divide the reaction coordinate into."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    return parser.parse_args()

def determine_jtype(dof: List[str]) -> int:
    """
    Determine the type of the degree of freedom based on its definition.
    """
    num_atoms = len(dof)
    jtype_map = {2: "bond", 3: "angle", 4: "torsion"}
    jtype_str = jtype_map.get(num_atoms, "bond")
    if jtype_str == "bond" and num_atoms not in jtype_map:
        logging.warning(f"Unknown DOF type for DOF: {dof}. Defaulting to 'bond'.")
    return DOF_TYPE_MAPPING.get(jtype_str, 0)  # Default to 'bond' if unknown

@njit
def get_jacobian_1D(jtype: int, bin_index: int, min_val: float, dx: float) -> float:
    """
    Jacobian for 1D histogram bin center, used in 1D entropy.
    """
    if jtype == 0:  # bond
        x_center = min_val + dx * (bin_index + 0.5)
        return x_center**2
    elif jtype == 1:  # angle
        x_center = min_val + dx * (bin_index + 0.5)
        return math.sin(x_center)
    elif jtype == 2:  # torsion
        return 1.0
    else:
        return 1.0

@njit
def get_jacobian_2D(
    i: int, j: int,
    xjtype: int, yjtype: int,
    min_x: float, dx: float,
    min_y: float, dy: float
) -> float:
    """
    Jacobian for 2D histogram bin centers, used in H(X,Y).
    """
    x_center = min_x + dx * (i + 0.5)
    y_center = min_y + dy * (j + 0.5)

    # (bond=0, angle=1, torsion=2)
    if xjtype == 0 and yjtype == 0:  # bond-bond
        return (x_center**2) * (y_center**2)
    elif xjtype == 0 and yjtype == 1:  # bond-angle
        return (x_center**2) * math.sin(y_center)
    elif xjtype == 0 and yjtype == 2:  # bond-torsion
        return x_center**2
    elif xjtype == 1 and yjtype == 1:  # angle-angle
        return math.sin(x_center) * math.sin(y_center)
    elif xjtype == 1 and yjtype == 2:  # angle-torsion
        return math.sin(x_center)
    elif xjtype == 2 and yjtype == 2:  # torsion-torsion
        return 1.0
    else:
        return 1.0

@njit
def compute_entropy_1D_numba(dof: np.ndarray, jtype: int, bins: int, min_val: float, dx: float) -> float:
    """
    Compute 1D entropy for a given DOF using a histogram approach with Jacobian corrections.
    """
    counts = np.zeros(bins, dtype=np.int64)
    n_samples = dof.shape[0]
    for idx in range(n_samples):
        x = dof[idx]
        bin_idx = int((x - min_val) / dx)
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= bins:
            bin_idx = bins - 1
        counts[bin_idx] += 1

    sample_size = np.sum(counts)
    if sample_size == 0:
        return 0.0

    prob_density = counts / sample_size
    entropy_sum = 0.0
    for i in prange(bins):
        p = prob_density[i]
        if p > 0.0:
            jacobian = get_jacobian_1D(jtype, i, min_val, dx)
            # - ∫ p log( p / (jacobian * dx) )
            entropy_sum += -p * math.log(p / (jacobian * dx))

    # Miller–Maddow bias correction
    n_non_zero = 0
    for i in range(bins):
        if counts[i] > 0:
            n_non_zero += 1
    if sample_size > 0:
        entropy_sum += (n_non_zero - 1) / (2.0 * sample_size)

    return entropy_sum

@njit
def compute_joint_entropy(dof1, dof2, xjtype, yjtype,
                          bins, min_x, dx, min_y, dy) -> float:
    """
    Compute H(X,Y) with a 2D histogram and Jacobian corrections.
    """
    H_XY = np.zeros((bins, bins), dtype=np.float64)
    n_samples = dof1.shape[0]

    # Populate histogram
    for idx in range(n_samples):
        x = dof1[idx]
        y = dof2[idx]
        bin_x = int((x - min_x) / dx)
        bin_y = int((y - min_y) / dy)
        if bin_x < 0:
            bin_x = 0
        elif bin_x >= bins:
            bin_x = bins - 1
        if bin_y < 0:
            bin_y = 0
        elif bin_y >= bins:
            bin_y = bins - 1
        H_XY[bin_x, bin_y] += 1.0

    total_count = np.sum(H_XY)
    if total_count == 0:
        return 0.0

    H_XY /= total_count  # Convert counts to probability
    entropy_sum = 0.0

    for i in prange(bins):
        for j in range(bins):
            p = H_XY[i, j]
            if p > 0.0:
                jacobian = get_jacobian_2D(i, j, xjtype, yjtype, min_x, dx, min_y, dy)
                # - ∫ p log( p / (jacobian * dx * dy) )
                entropy_sum += -p * math.log(p / (jacobian * dx * dy))

    # Miller–Maddow bias correction
    n_non_zero = 0
    for i in range(bins):
        for j in range(bins):
            if H_XY[i, j] > 0.0:
                n_non_zero += 1
    if total_count > 0:
        entropy_sum += (n_non_zero - 1) / (2.0 * total_count)

    return entropy_sum

@njit
def compute_mutual_information(
    dof1, dof2, H_X, H_Y,
    xjtype, yjtype, bins,
    min_x, dx, min_y, dy
) -> float:
    """
    Compute MI(X; Y) = H_X + H_Y - H_XY.
    """
    H_XY = compute_joint_entropy(dof1, dof2, xjtype, yjtype,
                                 bins, min_x, dx, min_y, dy)
    return H_X + H_Y - H_XY

@njit
def precompute_min_max_dx(dofs: np.ndarray, bins: int):
    """
    Precompute min, max, and dx for each row (DOF) to avoid repeated np.min/np.max in a loop.
    """
    n_dofs = dofs.shape[0]
    min_vals = np.zeros(n_dofs, dtype=np.float64)
    max_vals = np.zeros(n_dofs, dtype=np.float64)
    dx_vals = np.zeros(n_dofs, dtype=np.float64)

    for i in range(n_dofs):
        min_vals[i] = np.min(dofs[i])
        max_vals[i] = np.max(dofs[i])
        rng = max_vals[i] - min_vals[i]
        dx_vals[i] = rng / bins if bins > 0 else 1.0

    return min_vals, max_vals, dx_vals

def load_topology(output_dir: Path) -> List[List[str]]:
    topology_file = output_dir / "topology.txt"
    if not topology_file.exists():
        logging.error(f"Topology file not found at {topology_file}")
        raise FileNotFoundError(f"Topology file not found at {topology_file}")

    with topology_file.open("r") as file:
        lines = file.readlines()

    # Assuming the first line is atom count and the rest are DOFs
    dof_list = [line.strip().split() for line in lines[1:] if line.strip()]
    logging.info(f"Loaded topology with {len(dof_list)} DOFs.")
    return dof_list

def compute_jtype_list(dof_list: List[List[str]]) -> np.ndarray:
    jtype_list = np.array([determine_jtype(dof) for dof in dof_list], dtype=np.int32)
    logging.info(f"Computed jtype_list with {len(jtype_list)} entries.")
    return jtype_list

def load_dofs(output_dir: Path) -> np.ndarray:
    """
    Loads the 'dofs.npy' file from the given directory.
    The file should have shape (#snapshots, #DoFs + 1).
    """
    dofs_file = output_dir / "dofs.npy"
    if not dofs_file.exists():
        logging.error(f"'dofs.npy' file not found in {output_dir}")
        raise FileNotFoundError(f"'dofs.npy' file not found in {output_dir}")

    dofs = np.load(dofs_file).astype(np.float32)
    logging.info(f"Loaded dofs.npy with shape: {dofs.shape}")
    return dofs

@njit
def compute_entropy_1D(dofs: np.ndarray, jtype_list: np.ndarray, bins: int) -> np.ndarray:
    """
    Compute 1D entropy for all DOFs (return an array of length num_dofs).
    """
    num_dofs = dofs.shape[0]
    entropy1d = np.zeros(num_dofs, dtype=np.float64)
    for i in range(num_dofs):
        jtype = jtype_list[i]
        data = dofs[i]
        min_val = np.min(data)
        max_val = np.max(data)
        dx = (max_val - min_val) / bins if bins > 0 else 1.0
        entropy_val = compute_entropy_1D_numba(data, jtype, bins, min_val, dx)
        entropy1d[i] = entropy_val
    return entropy1d

def compute_entropy_matrix(dofs_subset: np.ndarray, jtype_list: np.ndarray, bins: int) -> np.ndarray:
    """
    Compute the mutual information entropy matrix for a subset of DOFs.
    """
    if dofs_subset.shape[0] == 0:
        logging.warning("No snapshots in this ensemble. Returning zero matrix.")
        return np.zeros((len(jtype_list), len(jtype_list)), dtype=np.float64)

    entropy1d = compute_entropy_1D(dofs_subset, jtype_list, bins)

    # Prepare the output array
    num_dofs = dofs_subset.shape[0]
    em = np.zeros((num_dofs, num_dofs), dtype=np.float64)

    # Place 1D entropies on the diagonal
    for i in range(num_dofs):
        em[i, i] = entropy1d[i]

    # Precompute min/max/dx for each DOF
    dofs64 = dofs_subset.astype(np.float64)
    min_vals, max_vals, dx_vals = precompute_min_max_dx(dofs64, bins)

    # Compute pairwise mutual information
    compute_mi_for_i_parallel(num_dofs, dofs64, jtype_list, entropy1d, bins, min_vals, dx_vals, em)

    return em

@njit(parallel=True)
def compute_mi_for_i_parallel(
    num_dofs: int,
    dofs: np.ndarray,
    jtype_list: np.ndarray,
    entropy1d: np.ndarray,
    bins: int,
    min_vals: np.ndarray,
    dx_vals: np.ndarray,
    em: np.ndarray
):
    """
    Parallel computation of mutual information for all DOFs.
    """
    for i in prange(num_dofs-1):
        for j in prange(i+1, num_dofs):
            dof1 = dofs[i]
            dof2 = dofs[j]
            jtype_i = jtype_list[i]
            jtype_j = jtype_list[j]
            H_X = entropy1d[i]
            H_Y = entropy1d[j]
            min_x = min_vals[i]
            dx = dx_vals[i]
            min_y = min_vals[j]
            dy = dx_vals[j]

            mi = compute_mutual_information(
                dof1, dof2,
                H_X, H_Y,
                jtype_i, jtype_j,
                bins,
                min_x, dx, min_y, dy
            )
            em[i, j] = mi
            em[j, i] = mi

def save_entropy_matrix(entropy_matrix: np.ndarray, output_dir: Path, bond_max: float, bond_min: float) -> None:
    filename = f"em_{bond_max:.3f}_{bond_min:.3f}.npy"
    entropy_file = output_dir / filename
    np.save(entropy_file, entropy_matrix)
    logging.info(f"Entropy matrix saved to {entropy_file}")

def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_level)

    # Determine the data directory
    if args.output_dir is not None:
        # Use user-provided output directory
        data_dir = Path(args.output_dir)
    else:
        # Default to './output/{reaction}'
        data_dir = Path("./output") / args.reaction

    if not data_dir.exists():
        logging.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    logging.info(f"Data directory: {data_dir}")

    # Load topology & DOFs from data_dir
    dof_list = load_topology(data_dir)
    jtype_list = compute_jtype_list(dof_list)
    dofs = load_dofs(data_dir)

    # Separate reaction coordinate and DOFs
    reaction_coordinate = dofs[:, 0]
    dofs_only = dofs[:, 1:]
    logging.info(f"Separated reaction coordinate and DOFs. DOFs shape: {dofs_only.shape}")

    # Define reaction coordinate ensembles
    x = np.linspace(args.bondmax, args.bondmin, args.ensemble + 1)
    logging.info(f"Defined reaction coordinate bins: {x}")

    # Precompute jtype_list for DOFs
    num_dofs = dofs_only.shape[1]
    jtype_list_subset = jtype_list[:num_dofs]
    logging.info(f"Number of DOFs for entropy calculation: {num_dofs}")

    # Precompile some Numba functions (optional, for speed)
    _ = compute_entropy_1D_numba(np.zeros(10, dtype=np.float32), 0, 10, 0.0, 1.0)
    _ = compute_mutual_information(
        np.zeros(10, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        1.0, 1.0,
        0, 0,
        10,
        0.0, 1.0,
        0.0, 1.0
    )

    # Iterate over each ensemble and compute entropy matrices
    for i in trange(args.ensemble, desc="Processing Ensembles"):
        bond_upper = x[i]
        bond_lower = x[i + 1]
        logging.info(f"Processing ensemble {i+1}/{args.ensemble}: {bond_upper} > bond > {bond_lower}")

        # Select snapshots within the current ensemble
        mask = (reaction_coordinate > bond_lower) & (reaction_coordinate <= bond_upper)
        dofs_subset = dofs_only[mask]
        logging.info(f"Selected {dofs_subset.shape[0]} snapshots for this ensemble.")

        # Compute entropy matrix for the subset
        entropy_matrix = compute_entropy_matrix(dofs_subset.T, jtype_list_subset, bins=50)

        # Save the entropy matrix with bondmax and bondmin in the filename
        save_entropy_matrix(entropy_matrix, data_dir, bond_upper, bond_lower)

    logging.info("Entropic path sampling completed successfully.")

if __name__ == "__main__":
    main()
