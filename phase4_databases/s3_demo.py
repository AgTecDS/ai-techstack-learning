"""
s3_demo.py
----------
boto3 demo for S3-compatible object storage (works with AWS S3 or local MinIO).

Object storage stores arbitrary blobs (files) by key, not in a directory tree.
It is the standard place to keep:
  - Model checkpoints (.pth, .onnx)
  - Training datasets (large ZIP files)
  - Inference images uploaded by users
  - Experiment artifacts (logs, configs)

Requires:
    pip install boto3
    docker run -p 9000:9000 -p 9001:9001 \\
        -e MINIO_ROOT_USER=minio -e MINIO_ROOT_PASSWORD=minio123 \\
        minio/minio server /data --console-address ":9001"

MinIO console: http://localhost:9001 (login: minio / minio123)
"""

import io
import os
import tempfile

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# ── S3 client configuration ────────────────────────────────────────────────────

# For AWS S3, omit endpoint_url and set real AWS credentials via env vars or ~/.aws/credentials
# For MinIO (local), point to localhost and use path-style addressing.

S3_ENDPOINT  = os.getenv("S3_ENDPOINT",  "http://localhost:9000")
S3_KEY       = os.getenv("S3_KEY",       "minio")
S3_SECRET    = os.getenv("S3_SECRET",    "minio123")
BUCKET_NAME  = "ai-models"


def get_s3_client():
    """
    signature_version='s3v4' is required for MinIO and most non-AWS providers.
    addressing_style='path' puts the bucket name in the URL path:
      http://localhost:9000/ai-models/file.onnx
    rather than subdomain style:
      http://ai-models.localhost:9000/file.onnx  (doesn't work locally)
    """
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


# ── Operations ─────────────────────────────────────────────────────────────────

def ensure_bucket(s3, bucket: str):
    """Create the bucket if it doesn't exist. Idempotent."""
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"Bucket '{bucket}' already exists.")
    except ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket '{bucket}'.")
        else:
            raise


def upload_file(s3, local_path: str, s3_key: str, bucket: str = BUCKET_NAME):
    """
    Upload a local file to S3.

    ExtraArgs allows setting:
      - ContentType: helps browsers/clients handle downloads correctly
      - Metadata: key-value pairs stored alongside the object
      - StorageClass: STANDARD, INTELLIGENT_TIERING, GLACIER, etc.
    """
    file_size = os.path.getsize(local_path)
    s3.upload_file(
        Filename=local_path,
        Bucket=bucket,
        Key=s3_key,
        ExtraArgs={
            "ContentType": "application/octet-stream",
            "Metadata": {
                "model-framework": "pytorch",
                "phase": "phase3-edge-ai",
            },
        },
    )
    print(f"Uploaded '{local_path}' → s3://{bucket}/{s3_key}  ({file_size} bytes)")


def upload_bytes(s3, data: bytes, s3_key: str, bucket: str = BUCKET_NAME):
    """Upload in-memory bytes without writing to disk first."""
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=data,
        ContentType="text/plain",
    )
    print(f"Uploaded {len(data)} bytes → s3://{bucket}/{s3_key}")


def list_objects(s3, prefix: str = "", bucket: str = BUCKET_NAME):
    """
    List objects with an optional prefix filter.
    S3 doesn't have real directories — prefixes simulate folder structure.
    list_objects_v2 is the current API (v1 is deprecated).
    """
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects  = response.get("Contents", [])
    print(f"\nObjects in s3://{bucket}/{prefix}* — {len(objects)} items:")
    for obj in objects:
        size_kb = obj["Size"] / 1024
        print(f"  {obj['Key']:50s}  {size_kb:8.2f} KB  {obj['LastModified']}")
    return objects


def download_file(s3, s3_key: str, local_path: str, bucket: str = BUCKET_NAME):
    """Download an object to a local file."""
    s3.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)
    print(f"Downloaded s3://{bucket}/{s3_key} → {local_path}")


def generate_presigned_url(
    s3, s3_key: str, expiry_seconds: int = 3600, bucket: str = BUCKET_NAME
) -> str:
    """
    Generate a time-limited URL that grants temporary read access to a private object.
    Share this URL with a client so they can download directly from S3
    without needing AWS credentials. The URL expires after `expiry_seconds`.

    Common use case: give a user a download link for their model checkpoint.
    """
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=expiry_seconds,
    )
    print(f"\nPresigned URL (expires in {expiry_seconds}s):")
    print(f"  {url[:80]}...")
    return url


def get_object_metadata(s3, s3_key: str, bucket: str = BUCKET_NAME):
    """head_object returns metadata without downloading the object body."""
    response = s3.head_object(Bucket=bucket, Key=s3_key)
    print(f"\nMetadata for s3://{bucket}/{s3_key}:")
    print(f"  ContentLength : {response['ContentLength']} bytes")
    print(f"  ContentType   : {response.get('ContentType', 'n/a')}")
    print(f"  LastModified  : {response['LastModified']}")
    print(f"  Metadata      : {response.get('Metadata', {})}")


def delete_object(s3, s3_key: str, bucket: str = BUCKET_NAME):
    s3.delete_object(Bucket=bucket, Key=s3_key)
    print(f"Deleted s3://{bucket}/{s3_key}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    try:
        s3 = get_s3_client()

        # Verify connection
        s3.list_buckets()
        print("Connected to S3-compatible storage.")
    except Exception as e:
        print(f"\nCould not connect to S3/MinIO: {e}")
        print("Start MinIO with:")
        print("  docker run -p 9000:9000 -p 9001:9001 \\")
        print("      -e MINIO_ROOT_USER=minio -e MINIO_ROOT_PASSWORD=minio123 \\")
        print("      minio/minio server /data --console-address ':9001'")
        return

    # Bucket setup
    ensure_bucket(s3, BUCKET_NAME)

    # Create a small dummy "model" file for the demo
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(b"dummy onnx model bytes " * 100)
        dummy_path = f.name

    try:
        # Upload a file
        upload_file(s3, dummy_path, "models/mnist_cnn_v1.onnx")
        upload_file(s3, dummy_path, "models/mnist_cnn_int8_v1.onnx")

        # Upload bytes directly
        config_bytes = b'{"model": "mnist_cnn", "version": "1.0", "classes": 10}'
        upload_bytes(s3, config_bytes, "configs/mnist_cnn_v1.json")

        # List
        list_objects(s3, prefix="models/")
        list_objects(s3, prefix="configs/")

        # Metadata
        get_object_metadata(s3, "models/mnist_cnn_v1.onnx")

        # Presigned URL
        generate_presigned_url(s3, "models/mnist_cnn_v1.onnx", expiry_seconds=900)

        # Download
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            download_path = f.name
        download_file(s3, "models/mnist_cnn_v1.onnx", download_path)
        print(f"Downloaded file size: {os.path.getsize(download_path)} bytes")
        os.unlink(download_path)

        # Cleanup
        delete_object(s3, "configs/mnist_cnn_v1.json")

    finally:
        os.unlink(dummy_path)

    print("\nS3 demo complete.")
    print("Key takeaway: object storage is cheap, durable, and decouples your")
    print("application from the filesystem. Always use it for large binary assets.")


if __name__ == "__main__":
    main()
