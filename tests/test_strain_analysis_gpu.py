import pytest
from unittest.mock import MagicMock
from src.strain_analysis_gpu import initialize

@pytest.fixture
def mock_universe(monkeypatch):
    # Mock Universe and Selection objects from MDAnalysis
    mock_selection = MagicMock()
    mock_selection.positions = [[1, 2, 3], [4, 5, 6]]
    mock_selection.resids = [1, 2]
    
    mock_universe = MagicMock()
    mock_universe.select_atoms.return_value = mock_selection
    
    # Patch the Universe class in your module
    monkeypatch.setattr('src.your_script.mda.Universe', MagicMock(return_value=mock_universe))
    return mock_universe

def test_initialize(mock_universe):
    reference = 'dummy_reference.pdb'
    deformed = 'dummy_deformed.pdb'
    traj_ref = 'dummy_traj_ref.xtc'
    traj_deformed = 'dummy_traj_deformed.xtc'
    residue_numbers = [1, 2]
    protein_ca = 'some_selection_string'
    R = 10
    stride = 1
    
    # Call the function
    ref_selections, defm_selections = initialize(reference, deformed, traj_ref, traj_deformed, residue_numbers, protein_ca, R, stride)
    
    # Assertions to check if selections are as expected
    assert len(ref_selections) == len(defm_selections) == 2
    mock_universe.select_atoms.assert_called()  # Check if select_atoms was called

### 2. Testing Strain Tensor Computation
#### Test Case Example:

from src.strain_analysis_gpu import compute_strain_tensor
import numpy as np

def test_compute_strain_tensor():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[2, 0], [0, 2]])
    expected_output = np.array([[1.5, 0], [0, 1.5]])  # Expected output for a simple case
    
    # Execute
    result = compute_strain_tensor(A, B)
    
    # Assert
    np.testing.assert_array_almost_equal(result, expected_output)


from src.strain_analysis_gpu import write_files
import numpy as np

def test_write_files(tmpdir):
    shear_strains = np.array([[1, 2], [3, 4]])
    principal_strains = np.array([[[5, 6, 7], [8, 9, 10]], [[11, 12, 13], [14, 15, 16]]])
    
    # Create a temporary directory and run the function
    output_dir = tmpdir.mkdir("output")
    write_files(shear_strains, principal_strains)
    
    # Check contents of one of the output files
    shear_file = output_dir.join("shear_strains.txt")
    with open(shear_file) as f:
        lines = f.readlines()
    
    # Check if the contents are as expected
    assert lines == ['1 1 2\n', '2 3 4\n']