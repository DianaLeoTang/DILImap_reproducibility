#!/usr/bin/env python3
"""
TxGemma DILI Benchmark Script

This script provides a modular interface for running DILI (Drug-Induced Liver Injury)
predictions using Google's TxGemma model deployed on Vertex AI. It includes functions
for loading data, making predictions, and saving results.

Key components:
- Vertex AI endpoint management for TxGemma model
- DILI prediction pipeline for SMILES strings
- Data processing and result storage utilities
- Batch prediction capabilities
"""

import json
import os
from typing import Optional

import pandas as pd
from google.cloud import aiplatform
from google.cloud import storage


class TxGemmaDILIPredictor:
    """Class for managing TxGemma DILI predictions via Vertex AI."""

    def __init__(self, endpoint_id: str, endpoint_region: str, project_id: str):
        """Initialize the DILI predictor with Vertex AI endpoint details."""
        self.endpoint_id = endpoint_id
        self.endpoint_region = endpoint_region
        self.project_id = project_id
        self.endpoint = None
        self.tdc_prompts = None

    def setup_vertex_ai(self):
        """Initialize Vertex AI and load the endpoint."""
        aiplatform.init(project=self.project_id, location=self.endpoint_region)
        self.endpoint = aiplatform.Endpoint(
            endpoint_name=self.endpoint_id,
            project=self.project_id,
            location=self.endpoint_region,
        )

    def load_prompt_templates(
        self, gcs_path: str = 'gs://healthai-us/txgemma/templates/tdc_prompts.json'
    ):
        """Load TDC prompt templates from GCS."""
        # Download the prompt templates
        os.system(f'gcloud storage cp {gcs_path} tdc_prompts.json')

        with open('tdc_prompts.json', 'r') as f:
            self.tdc_prompts = json.load(f)

    def extract_dili_label(output_str):
        import re

        match = re.search(
            r'Output:\s*[\(]?\s*([AB])\s*[\)]?', str(output_str), re.IGNORECASE
        )
        if match:
            if match.group(1) == 'B':
                return 'DILI'
            elif match.group(1) == 'A':
                return 'No DILI'
        return 'Unknown'

    def predict_single_smiles(
        self, drug_smiles: str, max_tokens: int = 8, temperature: float = 0
    ) -> str:
        """Predict DILI for a single SMILES string."""
        if not self.tdc_prompts:
            raise ValueError(
                'Prompt templates not loaded. Call load_prompt_templates() first.'
            )

        task_name = 'DILI'
        input_type = '{Drug SMILES}'

        # Format the prompt
        tdc_prompt = self.tdc_prompts[task_name].replace(input_type, drug_smiles)

        # Prepare request
        instances = [
            {'prompt': tdc_prompt, 'max_tokens': max_tokens, 'temperature': temperature}
        ]

        # Get prediction
        response = self.endpoint.predict(instances=instances)
        predictions = response.predictions
        dili_prediction = self.extract_dili_label(predictions)

        return dili_prediction

    def predict_dili_label(self, smiles: str) -> str:
        """Predict DILI label for a given SMILES string with error handling."""
        try:
            dili_prediction = self.predict_single_smiles(smiles)
            return dili_prediction
        except Exception as e:
            print(f'Error predicting DILI for {smiles}: {e}')
            return 'Error'


class DataManager:
    """Class for managing data loading and saving operations."""

    @staticmethod
    def load_from_gcs(bucket_name: str, file_name: str) -> pd.DataFrame:
        """Load a CSV file from Google Cloud Storage."""
        gcs_path = f'gs://{bucket_name}/{file_name}'

        try:
            df = pd.read_csv(gcs_path)
            print(f'File loaded successfully from {gcs_path}')
            return df
        except Exception as e:
            print(f'Error loading file from {gcs_path}: {e}')
            raise

    @staticmethod
    def upload_to_gcs(df: pd.DataFrame, bucket_name: str, file_name: str):
        """Upload a Pandas DataFrame to GCS as a CSV file."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        csv_string = df.to_csv(index=False)
        blob.upload_from_string(csv_string, content_type='text/csv')

        print(f'DataFrame uploaded to gs://{bucket_name}/{file_name}')


class ModelUploader:
    """Class for uploading TxGemma models to Vertex AI Model Registry."""

    def __init__(self, model_variant: str = '2b-predict'):
        """Initialize model uploader with variant specification."""
        self.model_variant = model_variant
        self.model_id = f'txgemma-{model_variant}'
        self.serve_docker_uri = 'us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250114_0916_RC00_maas'
        self._set_compute_config()

    def _set_compute_config(self):
        """Set compute configuration based on model variant."""
        if '2b' in self.model_id:
            self.machine_type = 'g2-standard-12'
            self.accelerator_type = 'NVIDIA_L4'
            self.accelerator_count = 1
        elif '9b' in self.model_id:
            self.machine_type = 'g2-standard-24'
            self.accelerator_type = 'NVIDIA_L4'
            self.accelerator_count = 2
        else:
            self.machine_type = 'g2-standard-48'
            self.accelerator_type = 'NVIDIA_L4'
            self.accelerator_count = 4

    def upload_model(
        self,
        model_name: str,
        hf_token: str,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.95,
    ) -> aiplatform.Model:
        """Upload model to Vertex AI Model Registry."""
        if tensor_parallel_size is None:
            tensor_parallel_size = self.accelerator_count

        vllm_args = [
            'python',
            '-m',
            'vllm.entrypoints.api_server',
            '--host=0.0.0.0',
            '--port=8080',
            f'--model={self.model_id}',
            f'--tensor-parallel-size={tensor_parallel_size}',
            '--swap-space=16',
            f'--gpu-memory-utilization={gpu_memory_utilization}',
            '--enable-chunked-prefill',
            '--disable-log-stats',
        ]

        env_vars = {
            'MODEL_ID': self.model_id,
            'DEPLOY_SOURCE': 'notebook',
            'HF_TOKEN': hf_token,
        }

        model = aiplatform.Model.upload(
            display_name=model_name,
            serving_container_image_uri=self.serve_docker_uri,
            serving_container_args=vllm_args,
            serving_container_ports=[8080],
            serving_container_predict_route='/generate',
            serving_container_health_route='/ping',
            serving_container_environment_variables=env_vars,
        )

        return model


def setup_gcp_environment(project_id: str, region: str):
    """Set up Google Cloud environment and enable necessary APIs."""
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
    os.environ['GOOGLE_CLOUD_REGION'] = region


def run_dili_benchmark(
    endpoint_id: str,
    endpoint_region: str,
    project_id: str,
    input_bucket: str,
    input_file: str,
    output_bucket: str,
    output_file: str,
) -> pd.DataFrame:
    """Run the complete DILI benchmark pipeline."""

    # Initialize predictor
    predictor = TxGemmaDILIPredictor(endpoint_id, endpoint_region, project_id)
    predictor.setup_vertex_ai()
    predictor.load_prompt_templates()

    # Load data
    data_manager = DataManager()
    df = data_manager.load_from_gcs(input_bucket, input_file)
    print(f'Loaded {len(df)} compounds for prediction')

    # Run predictions
    print('Running DILI predictions...')
    df['TxGemma2B_Predicted_DILI_Label'] = df['smiles'].apply(
        predictor.predict_dili_label
    )

    # Save results
    data_manager.upload_to_gcs(df, output_bucket, output_file)

    return df


def main():
    """Main function to run DILI benchmark with default parameters."""

    # Configuration
    ENDPOINT_ID = os.getenv('VERTEX_AI_ENDPOINT_ID', 'your-endpoint-id')
    ENDPOINT_REGION = os.getenv('VERTEX_AI_REGION', 'us-central1')
    PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id')

    INPUT_BUCKET = os.getenv('INPUT_BUCKET', 'your-input-bucket')
    INPUT_FILE = 'unseen_smiles_txgemma_714.csv'
    OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET', 'your-output-bucket')
    OUTPUT_FILE = 'benchmark_txgemma_all_variants_715_predictions.csv'

    # Setup environment
    setup_gcp_environment(PROJECT_ID, ENDPOINT_REGION)

    # Run benchmark
    results_df = run_dili_benchmark(
        ENDPOINT_ID,
        ENDPOINT_REGION,
        PROJECT_ID,
        INPUT_BUCKET,
        INPUT_FILE,
        OUTPUT_BUCKET,
        OUTPUT_FILE,
    )

    print(f'Benchmark completed. Results saved with {len(results_df)} predictions.')
    return results_df


if __name__ == '__main__':
    main()
