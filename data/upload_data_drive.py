import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from constants import *


def load_env_file(env_path: Path) -> None:
    """Minimal .env loader without extra dependencies."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_drive_folder_ref(env_path: Path) -> str:
    load_env_file(env_path)
    folder_ref = os.getenv(GDRIVE_FOLDER_URL_ENV_KEY, "").strip()
    if folder_ref:
        return folder_ref
    raise RuntimeError(
        f"Missing {GDRIVE_FOLDER_URL_ENV_KEY}. Add it to {env_path}."
    )


def extract_drive_folder_id(folder_url_or_id: str) -> str:
    token = folder_url_or_id.strip()
    if re.fullmatch(GDRIVE_FOLDER_ID_RAW_REGEX, token):
        return token

    parsed = urlparse(token)
    match = re.search(GDRIVE_FOLDER_ID_PATH_REGEX, parsed.path)
    if match:
        return match.group(1)

    query = parse_qs(parsed.query)
    for key in GDRIVE_FOLDER_QUERY_KEYS:
        if key in query and query[key]:
            return query[key][0]

    raise ValueError(f"Could not extract Drive folder ID from: {folder_url_or_id}")


def build_backup_base_name() -> str:
    date_stamp = datetime.now().strftime(BACKUP_DATE_FORMAT)
    return f"{BACKUP_PREFIX}_{date_stamp}"


def make_zip_for_directory(source_dir: Path, output_stem: Path) -> Path:
    if not source_dir.exists() or not source_dir.is_dir():
        raise RuntimeError(f"Missing source directory: {source_dir}")

    archive_path = shutil.make_archive(
        base_name=str(output_stem),
        format=BACKUP_ARCHIVE_FORMAT,
        root_dir=str(source_dir.parent),
        base_dir=str(source_dir.name),
    )
    return Path(archive_path)


def authenticate_drive_service(credentials_path: Path, token_path: Path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), GDRIVE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise RuntimeError(
                    f"Credentials file not found: {credentials_path}\n"
                    "Create OAuth Desktop credentials in Google Cloud Console "
                    "and save the JSON at that path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), GDRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def upload_file_to_drive(service, folder_id: str, local_file: Path) -> str:
    from googleapiclient.http import MediaFileUpload

    body = {"name": local_file.name, "parents": [folder_id]}
    media = MediaFileUpload(str(local_file), mimetype=GDRIVE_UPLOAD_MIME_TYPE, resumable=True)
    created = (
        service.files()
        .create(
            body=body,
            media_body=media,
            fields=GDRIVE_UPLOAD_RESPONSE_FIELDS,
            supportsAllDrives=GDRIVE_SUPPORTS_ALL_DRIVES,
        )
        .execute()
    )
    return created["id"]


def main():
    raw_dir = Path(RAW_DATA_ROOT)
    labels_dir = Path(OUT_DIR)
    env_path = Path(ENV_FILE_PATH)
    credentials_path = Path(GDRIVE_CREDENTIALS_PATH)
    token_path = Path(GDRIVE_TOKEN_PATH)

    folder_ref = get_drive_folder_ref(env_path)
    folder_id = extract_drive_folder_id(folder_ref)
    backup_base = build_backup_base_name()

    raw_zip_name = f"{backup_base}_{BACKUP_RAW_SUFFIX}"
    labels_zip_name = f"{backup_base}_{BACKUP_LABELS_SUFFIX}"

    print(f"Drive folder ID: {folder_id}")
    print(f"Raw dir: {raw_dir}")
    print(f"Labels dir: {labels_dir}")

    with tempfile.TemporaryDirectory(prefix=BACKUP_TEMP_DIR_PREFIX) as tmp:
        tmp_dir = Path(tmp)
        raw_zip = make_zip_for_directory(raw_dir, tmp_dir / raw_zip_name)
        labels_zip = make_zip_for_directory(labels_dir, tmp_dir / labels_zip_name)

        service = authenticate_drive_service(credentials_path, token_path)
        raw_file_id = upload_file_to_drive(service, folder_id, raw_zip)
        labels_file_id = upload_file_to_drive(service, folder_id, labels_zip)

    print("Upload complete.")
    print(f"- {raw_zip.name} -> file id {raw_file_id}")
    print(f"- {labels_zip.name} -> file id {labels_file_id}")


if __name__ == "__main__":
    main()
