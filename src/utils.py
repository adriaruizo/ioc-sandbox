import gcsfs
import pandas as pd

class GCSFileLoadError(Exception):
    """
    Custom exception for errors encountered while loading files from Google Cloud Storage.
    
    This exception is raised when there's an issue with loading a parquet file from GCS,
    such as when the file is not found or an unexpected error occurs during the file loading process.
    """
    pass

def load_parquet_file(gcs_parquet_path: str) -> pd.DataFrame:
    """
    Loads a parquet file from Google Cloud Storage (GCS) into a pandas DataFrame.
    
    This function establishes a connection to GCS using gcsfs, attempts to open and read
    the specified parquet file, and then loads its contents into a pandas DataFrame. If the
    specified file does not exist or an error occurs during the loading process, a GCSFileLoadError
    is raised with an appropriate error message.
    
    Parameters:
    - gcs_parquet_path (str): The GCS path to the parquet file, including the bucket name
      and file path. For example, 'gs://my-bucket/path/to/file.parquet'.
    
    Returns:
    - pd.DataFrame: The loaded pandas DataFrame containing the data from the parquet file.
    
    Raises:
    - GCSFileLoadError: If the parquet file is not found or if an unexpected error occurs
      during the file loading process.
    """
    try:
        gcs = gcsfs.GCSFileSystem()
        with gcs.open(gcs_parquet_path, 'rb') as f:
            df = pd.read_parquet(f)
        return df
    except FileNotFoundError:
        raise GCSFileLoadError(f"Parquet file not found: {gcs_parquet_path}")
    except Exception as e:
        raise GCSFileLoadError(f"An error occurred: {e}")
