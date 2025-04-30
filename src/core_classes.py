import pandas as pd # mandatory
import os
from pathlib import Path

# Optional duckdb import
try:
    import duckdb as db
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Note: duckdb is not installed. Some SQL-like operations will not be available.")

class DataLoader:
    """
    Load data from various sources including Kaggle datasets.
    
    Args:
        filepath (str, optional): Path to local CSV file. If None, loads Titanic dataset.
        cache_locally (bool, optional): Whether to cache the dataset locally. Defaults to False.
        use_mock (bool, optional): Whether to use mock data instead of real data. Defaults to False.
    """
    def __init__(self, filepath: str = None, cache_locally: bool = False, use_mock: bool = False):
        self.filepath = filepath
        self.cache_locally = cache_locally
        self.use_mock = use_mock
        self.df = None
        
        if filepath is None:
            self._load_titanic()
        else:
            self.df = pd.read_csv(filepath)
            
    def _create_mock_data(self):
        """Create a mock Titanic dataset for testing."""
        return pd.DataFrame({
            'PassengerId': range(1, 11),
            'Survived': [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
            'Pclass': [3, 1, 3, 1, 2, 3, 1, 2, 3, 1],
            'Name': ['John Doe'] * 10,
            'Sex': ['male', 'female'] * 5,
            'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'SibSp': [1, 0, 0, 1, 1, 0, 2, 1, 0, 0],
            'Parch': [0, 0, 1, 1, 0, 2, 1, 1, 0, 0],
            'Ticket': ['ABC123'] * 10,
            'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.45, 51.86, 21.07, 11.13, 30.07],
            'Cabin': [''] * 10,
            'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'C', 'C']
        })
            
    def _load_titanic(self):
        """Load Titanic dataset from Kaggle and optionally cache it locally."""
        # Use mock data if explicitly requested
        if self.use_mock:
            self.df = self._create_mock_data()
            return
            
        try:
            import kaggle
        except ImportError:
            print("Warning: kaggle package not found. Using mock data instead.")
            self.df = self._create_mock_data()
            return
            
        # Ensure data/raw directory exists
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Path for cached file
        cache_path = raw_dir / "titanic.csv"
        
        # Load from cache if exists and caching is enabled
        if cache_path.exists() and self.cache_locally:
            print("Loading from cache...")
            self.df = pd.read_csv(cache_path)
            return
            
        try:
            print("Downloading Titanic dataset from Kaggle...")
            # Download from Kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_file(
                'heptapod/titanic',
                'train_and_test2.csv',
                path=str(raw_dir),
                quiet=True
            )
            
            # Extract and load
            self.df = pd.read_csv(raw_dir / 'train_and_test2.csv.zip')
            
            # Cache if requested
            if self.cache_locally:
                print("Caching dataset...")
                self.df.to_csv(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to download Titanic dataset: {e}")
            print("Using mock data instead.")
            self.df = self._create_mock_data()


class DataCleaner:
    """
    Main class for data cleaning operations.
    Each cleaning step can be used as a pipeline component.

    Todo:
    - print the columns of the dataframe
    - print the unique values of the dataframe
    - print the number of unique values of the dataframe
    - print the number of missing values of the dataframe
    - print the number of duplicate rows of the dataframe
    - print the number of rows and columns of the dataframe
    - print the number of rows and columns of the dataframe
    - Find date and convert to datetime
    - Find and convert to int
    - Find and convert to float
    - Find and convert to bool
    - Find and convert to category
    - Find and convert to object
    - Add more cleaning steps
    - Add more tests
    - Add more documentation
    - Add more error handling
    - Add more logging
    - Add more metrics
    """

    def __init__(self, df=None):
        self.df = df

    def load_data(self, filepath: str):
        """Load a DataFrame from a CSV file."""
        self.df = pd.read_csv(filepath)
        return self

    def save_data(self, filepath: str):
        """Save the current DataFrame to a CSV file."""
        if self.df is not None:
            self.df.to_csv(filepath, index=False)

    def remove_duplicates(self):
        """Remove duplicate rows from the DataFrame."""
        self.df = self.df.drop_duplicates()
        return self

    def handle_missing_values(self, strategy='mean', columns=None):
        """
        Handle missing values in the DataFrame.
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to apply the strategy
        """
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if strategy == 'mean':
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median':
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'mode':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
        return self

    def encode_categorical(self, columns):
        """Encode categorical columns using one-hot encoding."""
        # Filter columns to only those that exist in the DataFrame
        existing_columns = [col for col in columns if col in self.df.columns]
        if not existing_columns:
            print(f"Warning: None of the specified columns {columns} exist in the DataFrame.")
            return self
        self.df = pd.get_dummies(self.df, columns=existing_columns)
        return self

    def scale_features(self, columns):
        """Scale numerical features to zero mean and unit variance."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self

    def get_clean_data(self):
        """Return the cleaned DataFrame."""
        return self.df

    def execute_sql(self, query: str):
        """
        Execute SQL-like operations on the DataFrame if duckdb is available.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            pd.DataFrame: Result of the query
            
        Raises:
            ImportError: If duckdb is not installed
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb is not installed. Please install it to use SQL operations.")
        
        # Create a temporary duckdb connection
        conn = db.connect(':memory:')
        # Register the DataFrame as a table
        conn.register('df', self.df)
        # Execute the query
        result = conn.execute(query).fetchdf()
        return result