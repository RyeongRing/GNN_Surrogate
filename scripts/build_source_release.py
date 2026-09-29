"""Export an explicit source-only snapshot without Git history or study data."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitignore", "LICENSE", "README.md", "PUBLIC_FILES.txt",
    "requirements.txt", "requirements-swmm.txt",
}
SOURCE_DIRS = {"src", "scripts", "bellinge", "tests"}
SECRET_PATTERNS = (
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"AKIA[0-9A-Z]{16}",
)


def validate_public_path(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or path.as_posix() != name or any(p in {".", "..", ".git"} for p in path.parts)):
        raise ValueError(f"Unsafe public path: {name!r}")
    allowed = name in ROOT_FILES or name == "environment/original_freeze.txt"
    if len(path.parts) > 1:
        allowed |= path.parts[0] in SOURCE_DIRS and path.suffix == ".py"
        allowed |= path.parts[0] == "configs" and path.suffix == ".yaml"
        allowed |= path.parts[0] == "docs" and path.suffix == ".md"
    if not allowed:
        raise ValueError(f"Not a permitted source/documentation path: {name}")
    return path


def read_regular_file(root, relative):
    current = root
    for part in relative.parts:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Linked/reparse paths are excluded: {relative}")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 256 * 1024:
        raise ValueError(f"Not a small regular source file: {relative}")
    content = current.read_bytes()
    text = content.decode("utf-8-sig")
    if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
        raise ValueError(f"Binary/control content is excluded: {relative}")
    if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
        raise ValueError(f"Possible credential detected; review locally: {relative}")
    return content


def collect_public_files(root=ROOT):
    root = Path(root).resolve(strict=True)
    manifest = read_regular_file(root, PurePosixPath("PUBLIC_FILES.txt")).decode("utf-8-sig")
    names = [line.strip() for line in manifest.splitlines() if line.strip() and not line.startswith("#")]
    if not names or len(names) != len(set(names)):
        raise ValueError("The public allowlist must be nonempty and contain no duplicates.")
    return {name: read_regular_file(root, validate_public_path(name)) for name in sorted(names)}


def export_snapshot(root, output):
    root = Path(root).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Export outside the working repository, into a new directory.")
    if output.exists():
        raise FileExistsError("Refusing to merge with or overwrite an existing snapshot.")
    files = collect_public_files(root)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    checksums = "".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in files.items())
    with (output / "SHA256SUMS.txt").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(checksums)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Validate and list files without copying")
    action.add_argument("--output", type=Path, help="New source-only snapshot directory")
    args = parser.parse_args()
    if args.check:
        files = collect_public_files()
        for name, content in files.items():
            print(f"{len(content):8d}  {name}")
        print(f"Checked {len(files)} allowlisted files; no upload performed.")
    else:
        count = export_snapshot(ROOT, args.output)
        print(f"Exported {count} files plus SHA256SUMS.txt; no upload performed.")


if __name__ == "__main__":
    main()
