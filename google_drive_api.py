# Reference: https://developers.google.com/drive/api/quickstart/python

import argparse
import io
import json
import os
import mimetypes
from pathlib import Path
# `pip install google-auth-oauthlib`
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
# `pip install google-api-python-client`
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from getfilelistpy import getfilelist


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir_id")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--is_download", action="store_true", default=False)
    parser.add_argument("--is_upload", action="store_true", default=False)
    parser.add_argument("--upload_dir")
    return parser.parse_args()


def load_credentials():
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    creds = None
    
    with open("config.json", mode="r") as f:
        config = json.load(f)
    credential_json_path = config["credential_json_path"]

    # The file `token.json`` stores the user"s access and refresh tokens, and is created automatically when the authorization flow completes for the first time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credential_json_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run.
        with open("token.json", mode="w") as f:
            f.write(creds.to_json())
    return creds


def create_folder(drive_service, name, parent_fol_id=None):
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder"
    }
    if parent_fol_id:
        file_metadata["parents"] = [parent_fol_id]
    file = drive_service.files().create(
        body=file_metadata,
        fields="id"
    ).execute()
    return file.get("id")


def download_files(drive_service, save_dir, res):
    all_subdirs = res["folderTree"]
    dirid2dirname = {id:name for id, name in zip(all_subdirs["folders"], all_subdirs["names"])}

    file_list = res["fileList"]
    n_files = sum([len(i["files"]) for i in file_list])

    i = 1
    for files_dirtree in file_list:
        dirtree = files_dirtree["folderTree"]

        for files in files_dirtree["files"]:
            fileid = files["id"]
            filename = files["name"]

            save_dir_single_file = save_dir / "/".join(map(dirid2dirname.get, dirtree))
            os.makedirs(save_dir_single_file, exist_ok=True)

            request = drive_service.files().get_media(fileId=fileid)
            fh = io.FileIO(save_dir_single_file / filename, mode="wb")
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"[{str(i):>4s}/{str(n_files):>4s}] {filename}")
            i += 1


def upload_file(drive_service, tar_file, name, save_fol=None):
    file_metadata = {
        "name": name
    }
    if save_fol:
        file_metadata["parents"] = [save_fol]
    media = MediaFileUpload(
        tar_file,
        mimetype=mimetypes.guess_type(tar_file)[0]
    )
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()
    print(f"{name}")


def upload_directory(drive_service, upload_dir):
    upload_dir = Path(upload_dir)

    fol_id = create_folder(
        drive_service, name=upload_dir.name
    )

    dic = {str(upload_dir): fol_id}
    for file_or_dir in upload_dir.glob("**/*"):
        if file_or_dir.is_dir():
            fol_id = create_folder(
                drive_service,
                name=file_or_dir.name,
                parent_fol_id=dic[str(file_or_dir.parent)]
            )
            dic[str(file_or_dir)] = fol_id
        if file_or_dir.is_file():
            upload_file(
                drive_service,
                tar_file=file_or_dir,
                name = file_or_dir.name,
                save_fol=dic[str(file_or_dir.parent)]
            )


def main():
    args = get_args()
    download_dir_id = args.download_dir_id
    save_dir = Path(args.save_dir) if args.save_dir else args.save_dir
    is_download = args.is_download
    is_upload = args.is_upload
    upload_dir = args.upload_dir
    
    creds = load_credentials()

    resource = {
        "oauth2": creds,
        "id": download_dir_id,
        "fields": "files(name, id)",
    }
    res = getfilelist.GetFileList(resource)
    drive_service = build("drive", "v3", credentials=creds)

    if is_download:
        download_files(drive_service, save_dir, res)

    if is_upload:
        upload_directory(
            drive_service, upload_dir
        )


if __name__ == "__main__":
    main()
