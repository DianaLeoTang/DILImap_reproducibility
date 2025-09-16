#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
DILIPRedictor Installation Instructions:

1. Clone the repository:
    git clone https://github.com/Manas02/dili-pip.git

2. Navigate to the directory:
    cd git-repos/dili-pip/

3. Activate conda environment:
    conda activate dilipr

4. Install poetry2conda:
    pip install poetry2conda

5. Add the following to the end of pyproject.toml:
    [tool.poetry2conda]
    name = "dilipr-conda"

    # !pip install poetry2conda
# !conda env create -f ./benchmark_models/dilipr_environment.yaml
# !conda activate dilipr-conda

6. Convert poetry to conda environment:
    poetry2conda pyproject.toml environment.yaml

7. Deactivate current environment and create new one:
    conda deactivate
    conda env create -f /home/jovyan/git-repos/DILImap_reproducibility/reproducibility/benchmark_models/dilipr_environment.yaml

8. Activate the new environment:
    conda activate dilipr-conda

9. Install compatible scikit-learn version:
    pip install scikit-learn==1.2.0
    # Note: This matches the sklearn version used to train the pretrained model
"""


def setup_dilipredictor_environment():
    """
    Set up the DILIPRedictor conda environment if it doesn't exist.
    """
    env_file_path = './insilico_benchmarks/dilipr_environment.yaml'

    # Check if environment already exists
    try:
        result = subprocess.run(
            ['conda', 'env', 'list'], capture_output=True, text=True, check=True
        )
        if 'dilipr-conda' in result.stdout:
            logger.info('DILIPRedictor conda environment already exists.')

            # Check if required packages are installed
            logger.info('Checking if required packages are installed...')
            try:
                subprocess.run(
                    [
                        'conda',
                        'run',
                        '-n',
                        'dilipr-conda',
                        'python',
                        '-c',
                        'import joblib, pandas, sklearn; print("All packages available")',
                    ],
                    check=True,
                    capture_output=True,
                )
                logger.info('All required packages are available.')
                return True
            except subprocess.CalledProcessError:
                logger.info('Some packages are missing. Installing them...')
                try:
                    subprocess.run(
                        [
                            'conda',
                            'run',
                            '-n',
                            'dilipr-conda',
                            'pip',
                            'install',
                            'scikit-learn==1.2.0',
                            'joblib',
                            'pandas',
                        ],
                        check=True,
                    )
                    logger.info('Required packages installed successfully.')
                    return True
                except subprocess.CalledProcessError as e:
                    logger.error(f'Error installing packages: {e}')
                    return False

    except subprocess.CalledProcessError:
        logger.warning('Could not check existing conda environments.')

    # Create environment if it doesn't exist
    if os.path.exists(env_file_path):
        try:
            logger.info('Creating DILIPRedictor conda environment...')
            subprocess.run(['conda', 'env', 'create', '-f', env_file_path], check=True)
            logger.info('Environment created successfully.')

            # Install scikit-learn, joblib, and pandas in the new environment
            logger.info(
                'Installing compatible scikit-learn, joblib, and pandas versions...'
            )
            subprocess.run(
                [
                    'conda',
                    'run',
                    '-n',
                    'dilipr-conda',
                    'pip',
                    'install',
                    'scikit-learn==1.2.0',
                    'joblib',
                ],
                check=True,
            )
            logger.info('Required packages installed successfully.')

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f'Error setting up environment: {e}')
            return False
    else:
        logger.error(f'Environment file not found at {env_file_path}')
        return False


def run_dilipredictor_predictions(input_df, dili_pip_repo_path, n_jobs=None):
    """
    Run DILIPRedictor predictions on a list of SMILES.

    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    dili_pip_repo_path : str
        Path to the DILIPRedictor repository
    n_jobs : int, optional
        Number of parallel jobs. If None, uses half of available CPU cores

    Returns:
    --------
    pd.DataFrame
        DataFrame with SMILES and predictions
    """

    # Import these modules only when function is called (after environment setup)
    try:
        from joblib import Parallel, delayed
        import multiprocessing
        import pandas as pd
    except ImportError as e:
        logger.error(f'Failed to import required modules: {e}')
        logger.error(
            "Please ensure you're running in the 'dilipr-conda' environment with all dependencies installed"
        )
        raise ImportError(f'Missing required dependencies: {e}') from e

    # Load data and extract SMILES
    logger.info(f'Loading data from {input_df}')
    try:
        dmap_db = pd.read_csv(input_df)
        smiles_list = dmap_db['smiles'].tolist()
        logger.info(f'Loaded {len(smiles_list)} SMILES strings')
    except Exception as e:
        logger.error(f'Error loading input file: {e}')
        sys.exit(1)

    # Check if DILIPRedictor repository exists
    if not os.path.exists(dili_pip_repo_path):
        raise FileNotFoundError(
            f'DILIPRedictor repository not found at {dili_pip_repo_path}. '
            'Please clone the repository using: git clone https://github.com/Manas02/dili-pip.git'
        )

    sys.path.append(dili_pip_repo_path)

    try:
        from dilipred import DILIPRedictor
    except ImportError as e:
        raise ImportError(
            f'Failed to import DILIPRedictor: {e}. '
            'Please ensure the dili-pip repository is properly cloned and the environment is set up correctly.'
        ) from e

    def process_smiles(smiles, predictor):
        if pd.notna(smiles) and smiles:
            try:
                result, dilipr_pred, dilipr_proba = predictor.predict(smiles)
                pred_label = 'DILI' if dilipr_pred[0] == 1 else 'No DILI'
                # pred_prob =  if hasattr(dilipr_prob, '__getitem__') else dilipr_prob
                return pred_label, dilipr_proba[0]
            except Exception as e:
                return 'Error', f'Error: {str(e)}'
        else:
            return 'Invalid SMILES', None

    # Initialize predictor
    dp = DILIPRedictor()

    # Set number of cores
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() // 2

    # Run predictions in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_smiles)(smiles, dp) for smiles in smiles_list
    )

    # Unpack results into two lists
    predictions, probabilities = zip(*results)

    # Create output dataframe
    result_df = pd.DataFrame(
        {
            'smiles': smiles_list,
            'DILIPRedictor_prediction': predictions,
            'DILIPRedictor_probability': probabilities,
        }
    )

    return result_df


def main():
    """
    Main function to handle command line arguments and run predictions.
    python run_dilipredictor.py -i smiles_data/unseen_smiles_benchmark.csv  -o predictions/dilipredictor_benchmark_predictions.csv
    """

    # Default path configuration for DILIPRedictor
    DEFAULT_DILI_PIP_REPO_PATH = '/home/jovyan/git-repos/dili-pip'

    # Parse command line arguments first
    parser = argparse.ArgumentParser(
        description='Run DILIPRedictor predictions on SMILES data'
    )
    HERE = Path(__file__).resolve().parent
    file = (
        HERE.parent / 'benchmark_models' / 'smiles_data' / 'dilipr_unseen_dilirank.csv'
    )

    parser.add_argument(
        '--input_path',
        '-i',
        default=file,
        help='Path to input CSV file containing SMILES data',
    )
    parser.add_argument(
        '--output_path',
        '-o',
        default=None,
        help='Path to save output CSV file (if not specified, returns dataframe)',
    )
    parser.add_argument(
        '--dili_pip_repo_path',
        '-g',
        default=DEFAULT_DILI_PIP_REPO_PATH,
        help='Path to the DILIPRedictor repository',
    )
    parser.add_argument(
        '--n_jobs',
        '-j',
        type=int,
        default=None,
        help='Number of parallel jobs (default: half of available CPU cores)',
    )

    args = parser.parse_args()

    # Expand user paths
    input_path = os.path.expanduser(args.input_path)
    dili_pip_repo_path = os.path.expanduser(args.dili_pip_repo_path)
    if args.output_path:
        output_path = os.path.expanduser(args.output_path)

    # Check if we're in the correct environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != 'dilipr-conda':
        logger.info('Not in dilipr-conda environment. Setting up environment...')

        # Set up environment
        if not setup_dilipredictor_environment():
            logger.error('Failed to set up environment. Exiting.')
            sys.exit(1)

        logger.error(
            'Environment setup complete, but you need to activate it manually.'
        )
        logger.error('Please run: conda activate dilipr-conda')
        logger.error('Then re-run this script.')
        sys.exit(1)

    # If we're here, we're in the correct environment
    logger.info('Running in dilipr-conda environment')

    # Run predictions
    logger.info('Running DILIPRedictor predictions...')
    try:
        result_df = run_dilipredictor_predictions(
            input_path, dili_pip_repo_path, args.n_jobs
        )
    except Exception as e:
        logger.error(f'Error running predictions: {e}')
        sys.exit(1)

    # Save results or return dataframe
    logger.info('Predictions completed.')
    if args.output_path:
        logger.info(f'Saving results to {output_path}')
        result_df.to_csv(output_path, index=False)
        logger.info(f'Results saved to {output_path}')
    else:
        return result_df


if __name__ == '__main__':
    main()
