"""Fills in the repository names everywhere they are needed, then rebuilds the notebooks.

    python tools/set_names.py --github-repo my-repo-name --hf-repo mbruton/my-model-collection
    python tools/set_names.py --contact-email name@example.org
    python tools/set_names.py --citation "Bruton, M., Beloucif, M. and Megyesi, B. (2027). ..."

Only the exact placeholder texts are replaced, so running it twice does no harm.
Afterwards, check the changes (git diff) and commit them.
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT_FILES = [p for p in ROOT.rglob("*")
              if p.is_file() and p.suffix in {".py", ".md", ".toml", ".ipynb", ".json", ".cff"}
              and not {".git", "build"} & set(p.parts) and p.name not in ("set_names.py", "MAINTAINER_SETUP.md")]


def replace_everywhere(old, new):
    changed = []
    for p in TEXT_FILES:
        text = p.read_text(encoding="utf-8")
        if old in text:
            p.write_text(text.replace(old, new), encoding="utf-8")
            changed.append(p.relative_to(ROOT))
    print(f"{old!r} -> {new!r} in {len(changed)} file(s)" + (": " + ", ".join(map(str, changed)) if changed else ""))


def tidy_config():
    """Drops the PLACEHOLDER marker from config lines that have been filled in."""
    cfg = ROOT / "src" / "keybridge" / "config.py"
    lines = cfg.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        filled = ("GITHUB_REPO_NAME" not in line and "HF_MODEL_REPO\"" not in line
                  and 'SHARE_CONTACT_EMAIL = ""' not in line and "citation to be added" not in line)
        out.append(line.replace("  # PLACEHOLDER", "") if filled else line)
    cfg.write_text("".join(out), encoding="utf-8")


def set_config(name, value):
    cfg = ROOT / "src" / "keybridge" / "config.py"
    text = cfg.read_text(encoding="utf-8")
    new = re.sub(rf'^{name} = .*$', f'{name} = {value!r}', text, flags=re.M)
    cfg.write_text(new, encoding="utf-8")
    print(f"config.py: {name} = {value!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--github-repo", help="name of the GitHub repository under mbruton0426 (without the owner)")
    ap.add_argument("--hf-repo", help="Hugging Face model repository, as owner/name, e.g. mbruton/cipher-models")
    ap.add_argument("--contact-email", help="address for emailed model bundles")
    ap.add_argument("--citation", help="citation text for the paper")
    a = ap.parse_args()
    if not any(vars(a).values()):
        ap.error("give at least one option")
    if a.github_repo:
        replace_everywhere("GITHUB_REPO_NAME", a.github_repo.split("/")[-1])
    if a.hf_repo:
        if "/" not in a.hf_repo:
            ap.error("--hf-repo must be owner/name, e.g. mbruton/cipher-models")
        replace_everywhere("mbruton/HF_MODEL_REPO", a.hf_repo)
    if a.contact_email:
        set_config("SHARE_CONTACT_EMAIL", a.contact_email)
    if a.citation:
        set_config("CITATION", a.citation)
        replace_everywhere("[citation to be added]", a.citation)
    tidy_config()
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_notebooks.py")], check=True)


if __name__ == "__main__":
    main()
