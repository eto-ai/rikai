#  Copyright (c) 2021 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Various s3 utilities for Rikai functionality. For example, libraries like
OpenCV cannot open s3 url's directly. Instead we make a presigned s3 url.
"""
import logging

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as e:
    raise ImportError("Please `pip install rikai[aws]` to use s3 utils") from e


def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    Parameters
    ----------
    bucket_name: str
        S3 bucket name
    object_name: str
        S3 key path (everything after the bucket name)
    expiration: int
        Expiration in seconds
    """
    s3 = boto3.client("s3")
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        logging.error(e)
        return None
