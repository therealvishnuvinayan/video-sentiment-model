from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()


def start_training():
    role = os.getenv("EXECUTION_ROLE")
    bucket = os.getenv("S3_BUCKET")

    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{bucket}/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25
        },
        tensorboard_config=tensorboard_config
    )

    # Start training
    estimator.fit({
        "training": f"s3://{bucket}/dataset/train",
        "validation": f"s3://{bucket}/dataset/dev",
        "test": f"s3://{bucket}/dataset/test"
    })


if __name__ == "__main__":
    start_training()