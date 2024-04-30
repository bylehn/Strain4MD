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
reference = 'cript_wt_b1us.pdb'
deformed = 'cript_g330t_b1us.pdb'
traj_ref = 'cript_wt_b1us.xtc'
traj_deformed = 'cript_g330t_b1us.xtc'
out = 'cript_wt_g330t'
protein_ca = '(name CA and resid 7-98)'


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
    ref_selections = []
    defm_selections = []

    # Iterate over residue numbers to create selection strings
    for resid in residue_numbers:
        selection_str = f"({protein_ca} and around {R} (resid {resid} and name CA))"
        center_str = f"resid {resid} and name CA"

        ref_sel = ref.select_atoms(selection_str)
        defm_sel = defm.select_atoms(selection_str)

        ref_center = ref.select_atoms(center_str)
        defm_center = defm.select_atoms(center_str)

        if len(ref_sel) != len(defm_sel):
            raise ValueError("The number of atoms in reference and deformed selections do not match.")
        
        ref_selections.append((ref_sel, ref_center))
        defm_selections.append((defm_sel, defm_center))

    return ref, ref_selections, defm_selections

def process_frame(ref, ref_selections, defm_selections, stride):
    """Process frames and calculate strains."""
    shear_strains = []
    principal_strains = []

    for ts in ref.trajectory[::stride]:  # Process every 'stride' frames
        frame_shear = []
        frame_principal = []

        for (ref_sel, ref_center), (defm_sel, defm_center) in zip(ref_selections, defm_selections):
            A = ref_sel.positions - ref_center.positions[0]
            B = defm_sel.positions - defm_center.positions[0]
            Q = compute_strain_tensor(device_put(A), device_put(B))
            shear, principal = compute_principal_strains_and_shear(Q)
            frame_shear.append(shear.tolist())
            frame_principal.append(principal.tolist())
        shear_strains.append(frame_shear)
        principal_strains.append(frame_principal)

    return np.array(shear_strains), np.array(principal_strains)

def write_files(shear_strains, principal_strains):
    # Write shear strains to file
    with open('shear_strains.txt', 'w') as f_shear:
        for i, shear in enumerate(shear_strains):
            f_shear.write(f'{i+1} ')
            f_shear.write(' '.join([str(s) for s in shear]))
            f_shear.write('\n')

    # Write principal strains to file
    with open('principal_1.txt', 'w') as f_principal:
        for i, principal in enumerate(principal_strains):
            f_principal.write(f'{i+1} ')
            f_principal.write(' '.join([str(p[0]) for p in principal]))
            f_principal.write('\n')

    # Write principal strains to file
    with open('principal_2.txt', 'w') as f_principal:
        for i, principal in enumerate(principal_strains):
            f_principal.write(f'{i+1} ')
            f_principal.write(' '.join([str(p[1]) for p in principal]))
            f_principal.write('\n')

    # Write principal strains to file
    with open('principal_3.txt', 'w') as f_principal:
        for i, principal in enumerate(principal_strains):
            f_principal.write(f'{i+1} ')
            f_principal.write(' '.join([str(p[2]) for p in principal]))
            f_principal.write('\n')

# %%
# Initialize the reference and deformed selections
ref_positions, ref_positions_center, deformed_selection, deformed_centers, deformed = initialize(reference, deformed, residue_numbers, protein_ca, traj)

# %%
def main(reference, deformed, traj_ref, traj_deformed, residue_numbers, protein_ca, R, stride):
    # Initialize
    ref, ref_selections, defm_selections = initialize(reference, deformed, traj_ref, traj_deformed, residue_numbers, protein_ca, R, stride)

    start_time = time.time()

    # Initialize lists to store results
    shear_strains = []
    principal_strains = []

    # Iterate over the trajectory in batches
    num_frames = len(ref.trajectory[::stride])
    batch_size = 1000  # Set your batch size based on GPU memory capacity

    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        for frame_idx in range(batch_start, batch_end):
            actual_frame_idx = frame_idx * stride
            ref.trajectory[actual_frame_idx]  # Load the frame
            frame_shear, frame_principal = process_frame(ref, ref_selections, defm_selections, stride)
            shear_strains.extend(frame_shear)
            principal_strains.extend(frame_principal)
    
    # Print progress
    progress_percent = (batch_end / num_frames) * 100
    print(f"Processed {batch_end} out of {num_frames} frames. Progress: {progress_percent:.2f}%")
    
    end = time.time()
    print(f'Time elapsed: {end - start_time} s')

    write_files(shear_strains, principal_strains)

# %%
main(reference, deformed, traj_ref, traj_deformed, residue_numbers, protein_ca, R, stride)
