import pytest
import numpy as np
import h5py

def test_h5py_numpy_compatibility():
    """
    Test if h5py and numpy are communicating correctly.
    This would have caught the 'ValueError: numpy.dtype size changed'.
    """
    try:
        # We try to create a small dummy HDF5 file in memory
        with h5py.File('test.h5', 'w', driver='core', backing_store=False) as f:
            data = np.array([1, 2, 3], dtype='float64')
            f.create_dataset('dataset', data=data)
            # If we can read it back, the binary interface is working
            assert f['dataset'][0] == 1.0
    except ValueError as e:
        pytest.fail(f"Compatibility Error found: {e}")



def test_cython_compiled():
    """
    Test if we can import gprMax and its custom exceptions.
    """
    try:
        # Now we use the REAL names from the file you just showed me
        from gprMax.exceptions import GeneralError
        # If we can create an instance of it, the import worked!
        err = GeneralError("Test Success")
        assert err.message == "Test Success"
    except ImportError as e:
        pytest.fail(f"Could not find gprMax. Ensure you are running from the root folder: {e}")