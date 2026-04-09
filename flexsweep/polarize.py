import os
import shutil
import subprocess
import warnings
from pathlib import Path


_BIN_NAMES = ("flexsweep-polarize", "polarize")


def _is_executable(path):
    return path.is_file() and os.access(path, os.X_OK)


def _find_binary(binary=None):
    if binary:
        path = Path(binary)
        if _is_executable(path):
            return path
        resolved = shutil.which(str(binary))
        return Path(resolved) if resolved else None

    env_bin = os.environ.get("FLEXSWEEP_RUST_BIN")
    if env_bin:
        path = Path(env_bin)
        if _is_executable(path):
            return path

    data_dir = Path(__file__).resolve().parent / "data"
    for name in _BIN_NAMES:
        for suffix in ("", ".exe"):
            candidate = data_dir / f"{name}{suffix}"
            if _is_executable(candidate):
                return candidate

    src_root = Path(__file__).resolve().parent / "src"

    candidates: list[Path] = []
    target_base = src_root / "target"
    for profile in ("release", "debug"):
        for name in _BIN_NAMES:
            candidates.append(target_base / profile / name)
            candidates.append(target_base / profile / f"{name}.exe")

    for candidate in candidates:
        if _is_executable(candidate):
            return candidate

    for name in _BIN_NAMES:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)

    return None


def build_rust_polarization(
    *,
    release=True,
    target_dir=None,
    cargo: str = "cargo",
):
    if not shutil.which(cargo):
        raise FileNotFoundError("cargo not found; install the Rust toolchain.")

    manifest = Path(__file__).resolve().parent / "src" / "Cargo.toml"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Cargo manifest not found at {manifest}. Build the binary separately "
            "and set FLEXSWEEP_RUST_BIN."
        )

    cmd = [cargo, "build", "--manifest-path", str(manifest)]
    if release:
        cmd.append("--release")
    if target_dir:
        cmd += ["--target-dir", str(target_dir)]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, end="")
        raise

    target_base = Path(target_dir) if target_dir else manifest.parent / "target"
    profile = "release" if release else "debug"
    for name in _BIN_NAMES:
        candidate = target_base / profile / name
        if candidate.exists():
            return candidate
        candidate = target_base / profile / f"{name}.exe"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cargo build succeeded but the expected binary was not found."
    )


def _ensure_binary(binary=None):
    path = _find_binary(binary)
    if path:
        return path

    warnings.warn(
        "Rust CLI binary not found; attempting to build with cargo.",
        RuntimeWarning,
    )

    if not shutil.which("cargo"):
        warnings.warn(
            "cargo not found; cannot build Rust CLI.",
            RuntimeWarning,
        )
        raise FileNotFoundError("cargo not found; install the Rust toolchain.")

    built_path = build_rust_polarization()
    dest = Path(__file__).resolve().parent / "data" / built_path.name
    try:
        shutil.copy2(built_path, dest)
        dest.chmod(dest.stat().st_mode | 0o111)
        return dest
    except OSError:
        return built_path


def _default_output_vcf(vcf_path):
    vcf_path = Path(vcf_path)
    fname = vcf_path.name
    if fname.endswith(".vcf.gz"):
        base = fname[: -len(".vcf.gz")]
    elif fname.endswith(".vcf"):
        base = fname[: -len(".vcf")]
    else:
        base = fname
    return vcf_path.with_name(f"{base}.polarized.vcf.gz").as_posix()


def _join_outgroups(outgroup):
    if isinstance(outgroup, str):
        return outgroup
    return ",".join([s for s in outgroup if s])


def run_sort_maf(
    input_path,
    contig_order=None,
    chunk_bytes=268_435_456,
    tmp_dir=None,
    binary=None,
    verbosity=None,
    check=True,
):
    bin_path = _ensure_binary(binary)
    output_path = input_path.replace(".maf", ".sorted.maf")

    cmd = [
        str(bin_path),
        "sort-maf",
        "--input",
        str(input_path),
        "--output",
        output_path,
        "--chunk-bytes",
        str(chunk_bytes),
    ]

    if contig_order:
        cmd += ["--contig-order", str(contig_order)]
    if tmp_dir:
        cmd += ["--tmp-dir", str(tmp_dir)]
    if verbosity:
        cmd += ["--verbosity", verbosity]

    subprocess.run(cmd, check=check)
    return output_path


def run_polarize(
    maf,
    vcf,
    outgroup,
    method="kimura",
    nrandom=10,
    output=None,
    binary=None,
    verbosity=None,
    check=True,
):
    bin_path = _ensure_binary(binary)
    outgroup_str = _join_outgroups(outgroup)
    if not outgroup_str:
        raise ValueError("outgroup cannot be empty")

    cmd = [
        str(bin_path),
        "polarize",
        "--maf",
        str(maf),
        "--vcf",
        str(vcf),
        "--outgroup",
        outgroup_str,
        "--nrandom",
        str(nrandom),
    ]

    if method:
        cmd += ["--method", method]
    if output:
        cmd += ["--output", str(output)]
    if verbosity:
        cmd += ["--verbosity", verbosity]

    subprocess.run(cmd, check=check)
    
    return output if output else _default_output_vcf(vcf)
