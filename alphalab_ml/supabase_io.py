"""Upload / download artifacts to Supabase Storage."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

log = logging.getLogger(__name__)


def _client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def _bucket() -> str:
    return os.environ.get("SUPABASE_STORAGE_BUCKET", "alphalab-ml-artifacts")


def upload_file(local_path: str | Path, remote_name: str | None = None) -> str:
    """Upload *local_path* to the configured Supabase Storage bucket.

    Parameters
    ----------
    local_path:
        Path to the local file to upload.
    remote_name:
        Object key inside the bucket.  Defaults to the file's base name.

    Returns
    -------
    str
        The public (or signed) path of the uploaded object.
    """
    local_path = Path(local_path)
    remote_name = remote_name or local_path.name
    bucket = _bucket()

    with local_path.open("rb") as fh:
        data = fh.read()

    client = _client()
    # upsert so re-runs overwrite the previous artifact
    client.storage.from_(bucket).upload(
        remote_name,
        data,
        file_options={"upsert": "true"},
    )
    log.info("Uploaded %s → %s/%s", local_path, bucket, remote_name)
    return f"{bucket}/{remote_name}"


def download_file(remote_name: str, local_path: str | Path) -> Path:
    """Download *remote_name* from the bucket to *local_path*.

    Parameters
    ----------
    remote_name:
        Object key inside the bucket.
    local_path:
        Destination on disk.

    Returns
    -------
    Path
        Resolved path of the downloaded file.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket = _bucket()

    client = _client()
    raw: bytes = client.storage.from_(bucket).download(remote_name)
    local_path.write_bytes(raw)
    log.info("Downloaded %s/%s → %s", bucket, remote_name, local_path)
    return local_path
