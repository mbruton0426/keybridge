"""Helpers for the Colab notebooks (uploads, downloads, Google Drive). They also work
in a local Jupyter notebook, where uploads are replaced by file paths."""
import os
import shutil
import zipfile

from .common import stop


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def find_or_upload(value, what, upload_dir, multiple=False, required=True):
    """Returns path(s) given in `value` (comma-separated paths or folders), or asks for an upload."""
    os.makedirs(upload_dir, exist_ok=True)
    paths = [p.strip() for p in (value or "").split(",") if p.strip()]
    found = []
    for p in paths:
        if os.path.isdir(p):
            found += [os.path.join(p, f) for f in sorted(os.listdir(p)) if not f.startswith(".")]
        elif os.path.exists(p):
            found.append(p)
        elif os.path.exists(os.path.join(upload_dir, p)):
            found.append(os.path.join(upload_dir, p))
        else:
            stop(f"The {what} could not be found at: {p}\nLeave the box empty to upload it instead, or check the path.")
    if found:
        return found if multiple else found[0]
    if not in_colab():
        if required:
            stop(f"Please give the path to your {what} in step ①.")
        return [] if multiple else None
    from google.colab import files
    print(f"Please upload your {what}" + (" (you may select several files)" if multiple else "") + ":")
    uploaded = files.upload()
    if not uploaded:
        if required:
            stop(f"No {what} was uploaded. Run this step again to try once more.")
        return [] if multiple else None
    saved = []
    for name, data in uploaded.items():
        dest = os.path.join(upload_dir, name)
        with open(dest, "wb") as f:
            f.write(data)
        saved.append(dest)
    return saved if multiple else saved[0]


def zip_files(paths, zip_path, folder_name):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, arcname=f"{folder_name}/{os.path.basename(p)}")
    return zip_path


def offer_download(path):
    if in_colab():
        from google.colab import files
        print(f"Downloading {os.path.basename(path)} ... (if nothing happens: open the folder icon on the left, "
              "find the file, right-click → Download)")
        files.download(path)
    else:
        print("Saved:", path)


def copy_to_drive(folder, subfolder):
    if not in_colab():
        return None
    from google.colab import drive
    drive.mount("/content/drive")
    dest = f"/content/drive/MyDrive/{subfolder}"
    shutil.copytree(folder, dest, dirs_exist_ok=True)
    print(f"Copied to your Google Drive: My Drive/{subfolder}")
    return dest
