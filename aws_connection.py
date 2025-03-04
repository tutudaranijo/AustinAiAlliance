# aws_connection.py Amazon ECS (Fargate)
import boto3
import os

def list_s3_buckets():
    # Retrieve AWS credentials and region from environment variables.
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    # Create an S3 client using boto3
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    
    # List all S3 buckets
    response = s3_client.list_buckets()
    buckets = [bucket["Name"] for bucket in response.get("Buckets", [])]
    return buckets

if __name__ == '__main__':
    buckets = list_s3_buckets()
    print("S3 Buckets:")
    for bucket in buckets:
        print(" -", bucket)