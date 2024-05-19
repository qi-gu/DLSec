# !/usr/bin/env python
# coding=UTF-8

import os
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import requests
from tqdm.auto import tqdm

cache_dir = Path(__file__).resolve().parents[1] / "data"  # git ignored
cache_dir.mkdir(parents=True, exist_ok=True)


AITESTING_DOMAIN = "http://218.245.5.12/audio"


def download_if_needed(url: str, dst_dir: str = cache_dir, extract: bool = False) -> str:
    """ """
    dst_dir = str(Path(dst_dir).expanduser().resolve())
    dst, need_download = _format_dst(url, dst_dir)
    dst = str(Path(dst).resolve())
    if not need_download:
        if not is_compressed_file(dst) or not extract:
            return dst
        stem = _stem(dst)
        dst_dir = str(Path(dst_dir) / stem)
        if os.path.exists(dst_dir):
            return dst_dir
        suffixes = Path(dst).suffixes
        if ".zip" in suffixes:
            _unzip_file(str(dst), dst_dir)
        elif ".tar" in suffixes:  # tar files
            _untar_file(str(dst), dst_dir)
        return dst_dir
    dst = download(url, dst)
    if extract and is_compressed_file(dst):
        suffixes = Path(dst).suffixes
        stem = _stem(dst)
        dst_dir = str(Path(dst_dir) / stem)
        if ".zip" in suffixes:
            _unzip_file(str(dst), dst_dir)
        elif ".tar" in suffixes:  # tar files
            _untar_file(str(dst), dst_dir)
        return dst_dir
    else:
        return dst


def _format_dst(url: str, dst_dir: str = cache_dir) -> Tuple[str, bool]:
    """ """
    dst = os.path.join(dst_dir, (url.strip("/").split("/"))[-1])
    return dst, not os.path.exists(dst)


def download(url: str, dst: str) -> str:
    """ """
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    print(f"downloading from {url} into {dst_dir}")
    http_get(url, dst)
    return dst


def http_get(url: str, fname: str):
    # https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    print(fname)
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=3.0,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def is_compressed_file(path: Union[str, Path]) -> bool:
    """
    check if the file is a valid compressed file

    Parameters
    ----------
    path: str or Path,
        path to the file

    Returns
    -------
    bool,
        True if the file is a valid compressed file, False otherwise.

    """
    compressed_file_pattern = "(\\.zip)|(\\.tar)"
    return re.search(compressed_file_pattern, _suffix(path)) is not None


def _unzip_file(path_to_zip_file: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    """
    Unzips a .zip file to folder path.

    Parameters
    ----------
    path_to_zip_file: str or Path,
        path to the .zip file
    dst_dir: str or Path,
        path to the destination folder

    """
    print(f"Extracting file {path_to_zip_file} to {dst_dir}.")
    with zipfile.ZipFile(str(path_to_zip_file)) as zip_ref:
        zip_ref.extractall(str(dst_dir))


def _untar_file(path_to_tar_file: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    """
    Decompress a .tar.xx file to folder path.

    Parameters
    ----------
    path_to_tar_file: str or Path,
        path to the .tar.xx file
    dst_dir: str or Path,
        path to the destination folder

    """
    print(f"Extracting file {path_to_tar_file} to {dst_dir}.")
    mode = Path(path_to_tar_file).suffix.replace(".", "r:").replace("tar", "").strip(":")
    with tarfile.open(str(path_to_tar_file), mode) as tar_ref:
        # tar_ref.extractall(str(dst_dir))
        # CVE-2007-4559 (related to  CVE-2001-1267):
        # directory traversal vulnerability in `extract` and `extractall` in `tarfile` module
        _safe_tar_extract(tar_ref, str(dst_dir))


def _stem(path: Union[str, Path]) -> str:
    """
    get filename without extension, especially for .tar.xx files

    Parameters
    ----------
    path: str or Path,
        path to the file

    Returns
    -------
    str,
        filename without extension

    """
    ret = Path(path).stem
    for _ in range(3):
        ret = Path(ret).stem
    return ret


def _suffix(path: Union[str, Path]) -> str:
    """
    get file extension, including all suffixes

    Parameters
    ----------
    path: str or Path,
        path to the file
    ignore_pattern: str, default PHYSIONET_DB_VERSION_PATTERN,
        pattern to ignore in the filename

    Returns
    -------
    str,
        full file extension

    """
    return "".join(Path(path).suffixes)


def _is_within_directory(directory: Union[str, Path], target: Union[str, Path]) -> bool:
    """
    check if the target is within the directory

    Parameters
    ----------
    directory: str or Path,
        path to the directory
    target: str or Path,
        path to the target

    Returns
    -------
    bool,
        True if the target is within the directory, False otherwise.

    """
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


def _safe_tar_extract(
    tar: tarfile.TarFile,
    dst_dir: Union[str, Path],
    members: Optional[Iterable[tarfile.TarInfo]] = None,
    *,
    numeric_owner: bool = False,
) -> None:
    """
    Extract members from a tarfile **safely** to a destination directory.

    Parameters
    ----------
    tar: tarfile.TarFile,
        the tarfile to extract from
    dst_dir: str or Path,
        the destination directory
    members: Iterable[tarfile.TarInfo], optional,
        the members to extract,
        if None, extract all members,
        if not None, must be a subset of the list returned by `tar.getmembers()`
    numeric_owner: bool, default False,
        if True, only the numbers for user/group names are used and not the names.

    """
    for member in members or tar.getmembers():
        member_path = os.path.join(dst_dir, member.name)
        if not _is_within_directory(dst_dir, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(dst_dir, members, numeric_owner=numeric_owner)
