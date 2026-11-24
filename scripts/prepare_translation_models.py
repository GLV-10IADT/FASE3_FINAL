from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


_PACKAGES = (
    {
        "direction": "pt_en",
        "package_name": "translate-pt_en-1_9",
        "from_code": "pt",
        "to_code": "en",
        "url": "https://argos-net.com/v1/translate-pt_en-1_9.argosmodel",
        "sha256": "AE76DF6F650895C16F2B582065014FAB496755CA846ECB19FAE81D51F332A38E",
    },
    {
        "direction": "en_pt",
        "package_name": "translate-en_pt-1_9",
        "from_code": "en",
        "to_code": "pt",
        "url": "https://argos-net.com/v1/translate-en_pt-1_9.argosmodel",
        "sha256": "0C5350A2FA5B923DE1346EDC0D42E08E38BAB2E33CEDE3A0B9A48EB4281AD8A9",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as package_file:
        for block in iter(lambda: package_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _download_package(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "academic-medical-assistant/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        with destination.open("wb") as package_file:
            shutil.copyfileobj(response, package_file)


def _write_member(
    archive: zipfile.ZipFile,
    member: str,
    *,
    package_root: str,
    output_directory: Path,
) -> None:
    member_path = PurePosixPath(member)
    root_path = PurePosixPath(package_root)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe path in translation package: {member!r}")
    relative_path = member_path.relative_to(root_path)
    destination = output_directory.joinpath(*relative_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(archive.read(member))


def _extract_package(package_path: Path, spec: dict[str, Any], staging: Path) -> None:
    package_root = spec["package_name"]
    output_directory = staging / spec["direction"]
    with zipfile.ZipFile(package_path) as archive:
        metadata_path = f"{package_root}/metadata.json"
        metadata = json.loads(archive.read(metadata_path))
        if (
            metadata.get("from_code") != spec["from_code"]
            or metadata.get("to_code") != spec["to_code"]
            or metadata.get("package_version") != "1.9"
        ):
            raise ValueError(f"Unexpected metadata in translation package {package_path}")

        required_members = {
            f"{package_root}/sentencepiece.model",
            f"{package_root}/README.md",
            metadata_path,
        }
        members = archive.namelist()
        if not required_members.issubset(members):
            raise ValueError(f"Translation package is missing required files: {package_path}")

        for member in members:
            if member.startswith(f"{package_root}/model/") and not member.endswith("/"):
                _write_member(
                    archive,
                    member,
                    package_root=package_root,
                    output_directory=output_directory,
                )
        _write_member(
            archive,
            f"{package_root}/sentencepiece.model",
            package_root=package_root,
            output_directory=output_directory,
        )
        _write_member(
            archive,
            metadata_path,
            package_root=package_root,
            output_directory=output_directory,
        )
        _write_member(
            archive,
            f"{package_root}/README.md",
            package_root=package_root,
            output_directory=output_directory / "OPUS-MT-ATTRIBUTION",
        )


def prepare_translation_models(
    output_directory: Path,
    *,
    package_directory: Path | None = None,
) -> tuple[Path, Path]:
    if output_directory.exists():
        raise FileExistsError(
            f"Translation model output already exists: {output_directory}"
        )
    if package_directory is not None and not package_directory.is_dir():
        raise FileNotFoundError(f"Translation package directory does not exist: {package_directory}")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_directory.parent) as temp_directory:
        temporary_root = Path(temp_directory)
        staging = temporary_root / "translation"
        staging.mkdir()

        for spec in _PACKAGES:
            package_name = f"{spec['package_name']}.argosmodel"
            package_path = (
                package_directory / package_name
                if package_directory is not None
                else temporary_root / package_name
            )
            if package_directory is None:
                _download_package(spec["url"], package_path)
            if _sha256(package_path) != spec["sha256"]:
                raise ValueError(f"SHA-256 mismatch for translation package: {package_name}")
            _extract_package(package_path, spec, staging)

        staging.replace(output_directory)

    return output_directory / "pt_en", output_directory / "en_pt"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download and prepare local OPUS-MT Portuguese/English models"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/translation"),
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        help="Use already downloaded .argosmodel files from this directory",
    )
    args = parser.parse_args()
    output_directory = args.output_dir
    if not output_directory.is_absolute():
        output_directory = project_root / output_directory
    package_directory = args.package_dir
    if package_directory is not None and not package_directory.is_absolute():
        package_directory = project_root / package_directory

    pt_en, en_pt = prepare_translation_models(
        output_directory,
        package_directory=package_directory,
    )
    print("Modelos locais preparados:")
    print("  português -> inglês:", pt_en)
    print("  inglês -> português:", en_pt)


if __name__ == "__main__":
    main()
