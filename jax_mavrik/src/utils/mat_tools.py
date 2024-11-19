import scipy.io
import pickle 
import jax.numpy as jnp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq



# Step 1: Define a Python class to hold the data
class MatStruct:
    def __init__(self, **entries):
        # Update the instance dictionary with entries from the MATLAB struct
        self.__dict__.update(entries)

# Helper function to convert mat_struct to a dictionary
def mat_struct_to_dict(matobj):
    """
    Recursively converts mat_struct objects to nested dictionaries.
    """
    data_dict = {}
    for field in matobj._fieldnames:
        elem = getattr(matobj, field)
        if isinstance(elem, scipy.io.matlab._mio5_params.mat_struct):
            data_dict[field] = mat_struct_to_dict(elem)
        elif isinstance(elem, list):
            # If the element is a list, convert each item if it is a mat_struct
            data_dict[field] = [mat_struct_to_dict(sub_elem) if isinstance(sub_elem, scipy.io.matlab._mio5_params.mat_struct) else sub_elem for sub_elem in elem]
        else:
            data_dict[field] = elem
    return data_dict

# Function to convert all mat_structs in the data
def convert_mat_data(mat_data):
    for key, value in mat_data.items():
        if isinstance(value, scipy.io.matlab._mio5_params.mat_struct):
            mat_data[key] = mat_struct_to_dict(value)
        elif isinstance(value, list):
            mat_data[key] = [mat_struct_to_dict(item) if isinstance(item, scipy.io.matlab._mio5_params.mat_struct) else item for item in value]
    return mat_data

def mat2pickle(file_path):
    # Step 1: Load the .mat file
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    # Step 2: Convert the MATLAB structure to Python class instances
    data_class = convert_mat_data(mat_data)
    # Step 3: Save the converted data to a pickle file
    pickle_file_path = 'data.pkl'  # Define the output pickle file path
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(data_class, f)

    print(f"Data saved to {pickle_file_path} as a pickle file.")


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary by concatenating keys with `sep`.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) and v and isinstance(v[0], dict):  # List of dicts (e.g., struct arrays in MATLAB)
            for i, sub_v in enumerate(v):
                items.extend(flatten_dict(sub_v, f"{new_key}_{i}", sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def pickle2pd(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        # Step 2: Flatten the dictionary to ensure no nested structures
        data_flat = flatten_dict(data)
        # Step 3: Convert the flattened dictionary to a Pandas DataFrame
        # Ensure that all values are lists or arrays of the same length for DataFrame compatibility
        data_df = pd.DataFrame({k: pd.Series(v) for k, v in data_flat.items()})
        # Step 4: (Optional) Write the DataFrame to a Parquet file for efficient storage
        parquet_file_path = 'data.parquet'
        table = pa.Table.from_pandas(data_df)
        pq.write_table(table, parquet_file_path)

        return data_df


def euler_to_dcm(roll, pitch, yaw):
    """
    Calculates the Direction Cosine Matrix (DCM) from Euler angles.
    
    Args:
        phi (float): Roll angle.
        theta (float): Pitch angle.
        psi (float): Yaw angle.
    
    Returns:
        jax.numpy.ndarray: The 3x3 Direction Cosine Matrix (DCM).
    """
    
    # phi, theta, psi = euler_angles
    cos, sin =jnp.cos, jnp.sin
    array = jnp.asarray
    
        
    s_roll = sin(roll) #1
    s_pitch = sin(pitch) #2
    s_yaw = sin(yaw) #3

    c_roll = cos(roll) #4
    c_pitch = cos(pitch) #5
    c_yaw = cos(yaw) #6

    # Rotation matrix from body frame to inertial frame
    return jnp.array([
        [c_roll * c_pitch, s_roll * c_pitch, -s_pitch],
        [c_roll * s_pitch * s_yaw - s_roll * c_yaw, s_roll * s_pitch * s_yaw + c_roll * c_yaw, c_pitch * s_yaw],
        [c_roll * s_pitch * c_yaw + s_roll * s_yaw, s_roll * s_pitch * c_yaw - c_roll * s_yaw, c_pitch * c_yaw]
    ])
    '''
    return jnp.array([
        [c_pitch * c_yaw, c_pitch * s_yaw, -s_pitch],
        [s_roll * s_pitch * c_yaw - c_roll * s_yaw, s_roll * s_pitch * s_yaw + c_roll * c_yaw, s_roll * c_pitch],
        [c_roll * s_pitch * c_yaw + s_roll * s_yaw, c_roll * s_pitch * s_yaw - s_roll * c_yaw, c_roll * c_pitch]
    ])
    '''