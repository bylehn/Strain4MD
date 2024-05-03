# %%
import MDAnalysis as mda
import jax.numpy as jnp
from jax import device_put, jit, vmap
from jax.scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import time

# %%
# Define the parameters
R = 10
stride = 10
residue_numbers = list(range(307, 398))
#residue_numbers = [29]
reference = '../examples/cript_wt_b1us.pdb'
deformed = '../examples/cript_g330t_b1us.pdb'
traj_ref = '../examples/cript_wt_b1us.xtc'
traj_deformed = '../examples/cript_g330t_b1us.xtc'
out = '../examples/cript_wt_g330t'
protein_ca = '(name CA and resid 307-398)'


# %%
# Define all the functions
@jit
def compute_strain_tensor(Am, Bm):
    D = jnp.linalg.inv(Am.transpose() @ Am)
    C = Bm @ Bm.transpose() - Am @ Am.transpose()
    Q = 0.5 * (D @ Am.transpose() @ C @ Am @ D)
    return Q

@jit
def compute_principal_strains_and_shear(Q):
    eigenvalues, _ = eigh(Q)
    shear = jnp.trace(Q @ Q) - (1/3) * (jnp.trace(Q))**2
    return shear, eigenvalues


def initialize(reference, deformed, traj_ref, traj_deformed, residue_numbers, protein_ca, R, stride):
    """Initialize the reference and deformed selections."""
    ref = mda.Universe(reference, traj_ref)
    defm = mda.Universe(deformed, traj_deformed)
    selections = []

    # Iterate over residue numbers to create selection strings
    for resid in residue_numbers:
        selection_str = f"({protein_ca} and around {R} (resid {resid} and name CA))"
        center_str = f"resid {resid} and name CA"

        ref_selection = ref.select_atoms(selection_str)
        ref_center = ref.select_atoms(center_str)

         # Get residue numbers from the reference selection
        ref_resids = ref_selection.resids
        defm_selection_str = f"(name CA and resid {' '.join(map(str, ref_resids))})"
        defm_center_str = f"resid {resid} and name CA" 

        defm_selection = defm.select_atoms(defm_selection_str)
        defm_center = defm.select_atoms(defm_center_str)

        selections.append(((ref_selection, ref_center), (defm_selection, defm_center)))

    return ref, defm, selections

def process_frame(ref, defm, selections, frame):
    """Process frames and calculate strains."""
    shear_strains = []
    principal_strains = []

    frame_shear = []
    frame_principal = []

    ref.trajectory[frame]
    defm.trajectory[frame]
    # if you want to add pairwise comparison between frames, add another loop here with 
    # for frame in zip(ref.trajectory, defm.trajectory) or something like that:
    for ((ref_sel, ref_center), (defm_sel, defm_center)) in selections:
        A = ref_sel.positions - ref_center.positions[0]
        B = defm_sel.positions - defm_center.positions[0]

        Q = compute_strain_tensor(device_put(A), device_put(B))
        shear, principal = compute_principal_strains_and_shear(Q)
        frame_shear.append(shear.tolist())
        frame_principal.append(principal.tolist())
    shear_strains.append(frame_shear)
    principal_strains.append(frame_principal)

    return np.array(shear_strains), np.array(principal_strains)

def write_files(shear_strains, principal_strains, avg_shear_strains, avg_principal_strains):
    # Calculate averages


    # Write average shear strains to file
    with open('avg_shear_strains.txt', 'w') as f_shear:
        for i, avg_shear in enumerate(avg_shear_strains):
            f_shear.write(f'Residue {i+1}: {avg_shear:.4f}\n')

    # Write average principal strains for each component
    for component in range(3):  # Assuming 3 principal components
        with open(f'avg_principal_{component+1}.txt', 'w') as f_principal:
            for i, principal in enumerate(avg_principal_strains):
                f_principal.write(f'Residue {i+1}: {principal[component]:.4f}\n')

    # Write raw shear strains
    with open('raw_shear_strains.txt', 'w') as f_raw_shear:
        for frame in shear_strains:
            f_raw_shear.write(' '.join(map(lambda x: f'{x:.4f}', frame)) + '\n')

    # Write raw principal strains for each component
    for component in range(3):
        with open(f'raw_principal_{component+1}.txt', 'w') as f_raw_principal:
            for frame in principal_strains:
                f_raw_principal.write(' '.join(map(lambda x: f'{x[component]:.4f}', frame)) + '\n')

# %%
def write_pdb_with_strains(deformed_pdb, avg_shear_strains, avg_principal_strains, output_filename):
    # Load the deformed PDB structure
    u = mda.Universe(deformed_pdb)
    residue_selection_string = "resid " + " ".join(map(str, residue_numbers))
    selected_residues = u.select_atoms(residue_selection_string)
    # Initialize a PDB writer for multiple frames
    with mda.Writer(output_filename, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
        # Frame 1: Average Shear Strains
        for residue in u.residues:
            if residue.resid in selected_residues.resids:
                ca_atom = residue.atoms.select_atoms('name CA')
                if ca_atom:  # Check if CA atom exists
                    ca_atom.tempfactors = 100 * avg_shear_strains[residue.resid - 307]  # Adjust index based on starting residue
        PDB.write(u.atoms)  # Write first frame

        # Next 3 Frames: Principal Strains
        for component in range(3):  # Assuming 3 principal components
            for residue in u.residues:
                if residue.resid in selected_residues.resids:
                    ca_atom = residue.atoms.select_atoms('name CA')
                    if ca_atom:  # Check if CA atom exists
                        # Assign principal strain component to B-factor
                        ca_atom.tempfactors = 100 * avg_principal_strains[residue.resid - 307, component]
            PDB.write(u.atoms)  # Write each principal component frame

# %%
def main(reference, deformed, traj_ref, traj_defm, residue_numbers, protein_ca, R, stride):
    # Initialize
    ref, defm, selections = initialize(reference, deformed, traj_ref, traj_defm, residue_numbers, protein_ca, R, stride)

    start_time = time.time()

    # Initialize lists to store results
    shear_strains = []
    principal_strains = []

    # Iterate over the trajectory in batches
    num_frames = len(ref.trajectory[::stride])
    batch_size = 1  # Set your batch size based on GPU memory capacity

    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        for frame_idx in range(batch_start, batch_end):
            actual_frame_idx = frame_idx * stride
            frame_shear, frame_principal = process_frame(ref, defm, selections, actual_frame_idx)
            shear_strains.extend(frame_shear)
            principal_strains.extend(frame_principal)
    
    # Print progress
    progress_percent = (batch_end / num_frames) * 100
    print(f"Processed {batch_end} out of {num_frames} frames. Progress: {progress_percent:.2f}%")
    
    end = time.time()
    print(f'Time elapsed: {end - start_time} s')

    avg_shear_strains = np.mean(shear_strains, axis=0)
    avg_principal_strains = np.mean(principal_strains, axis=0)
    write_files(shear_strains, principal_strains, avg_shear_strains, avg_principal_strains)
    write_pdb_with_strains(deformed, avg_shear_strains, avg_principal_strains, out + '.pdb')

    return shear_strains, principal_strains

# %%
shear_strains, principal_strains = main(reference, deformed, traj_ref, traj_deformed, residue_numbers, protein_ca, R, stride)


# %%
