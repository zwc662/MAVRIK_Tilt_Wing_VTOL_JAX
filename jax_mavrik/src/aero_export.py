import scipy.io 
import numpy as np
 

class AeroExport:
    """
    JAX-compatible class to hold .mat file data with perturbation support.
    """
    def __init__(self, data):
        # Core data loaded from .mat file, flattened for efficient access
        self.data = data
        # Dictionary to hold temporary perturbations
   
    @classmethod
    def load_mat(cls, file_path: str):
        """
        Loads a .mat file and stores it in a JAX-compatible AeroExport instance.
        """
        def mat_struct_to_dict(matobj):
            """
            Recursively converts mat_struct to dictionary.
            """
            data = {}
            for field in matobj._fieldnames:
                elem = getattr(matobj, field)
                if isinstance(elem, scipy.io.matlab._mio5_params.mat_struct):
                    data[field] = mat_struct_to_dict(elem)
                else:
                    data[field] = elem
            return data

        mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        def flatten_data(data, parent_key='', sep='.'):
            """
            Recursively flattens nested dictionaries.
            """
            items = {}
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_data(v, new_key, sep=sep))
                elif isinstance(v, list) and v and isinstance(v[0], dict):  # Handle list of dicts (MATLAB struct arrays)
                    for i, sub_v in enumerate(v):
                        items.update(flatten_data(sub_v, f"{new_key}_{i}", sep=sep))
                else:
                    items[new_key] = np.array(v) if isinstance(v, (int, float, list)) else v
            return items

        # Process each variable in the .mat file
        data_dict = {}
        for key, value in mat_data.items():
            if isinstance(value, scipy.io.matlab._mio5_params.mat_struct):
                data_dict[key] = flatten_data(mat_struct_to_dict(value))
            elif not key.startswith("__"):  # Skip meta-keys
                data_dict[key] = value

        # Flatten any nested dictionaries and initialize AeroExport
        return cls(data_dict)

if __name__ == '__main__':
    # Usage Example with Perturbation and JIT Compatibility
    file_path = '/Users/weichaozhou/Workspace/Mavrik_JAX/Mavrik/aero_export.mat'  # Replace with your .mat file path
    mat_data = AeroExport.load_mat(file_path)
