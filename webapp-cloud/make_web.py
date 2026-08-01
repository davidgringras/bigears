"""Assemble the soloscribe-web deploy repo from this repository.

  .venv/bin/python webapp-cloud/make_web.py [target_dir]

Copies the soloscribe/ package (minus webapp/, which needs fastapi) plus the
Streamlit app, its requirements and packages.txt into a flat directory whose
ROOT is what Streamlit Community Cloud deploys — packages.txt is only honored
at the repo root, which is the reason a dedicated repo exists at all.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(target: str) -> None:
    os.makedirs(target, exist_ok=True)
    pkg_dst = os.path.join(target, "soloscribe")
    if os.path.exists(pkg_dst):
        shutil.rmtree(pkg_dst)
    shutil.copytree(
        os.path.join(ROOT, "soloscribe"), pkg_dst,
        ignore=shutil.ignore_patterns("webapp", "__pycache__", "*.pyc"),
    )
    for name in ("streamlit_app.py", "requirements.txt", "packages.txt"):
        shutil.copy2(os.path.join(HERE, name), os.path.join(target, name))
    readme = os.path.join(target, "README.md")
    with open(readme, "w") as f:
        f.write(
            "# SoloScribe (web)\n\nDeploy snapshot of "
            "[soloscribe](https://github.com/davidgringras/soloscribe) for "
            "Streamlit Community Cloud. Do not edit here — run "
            "`webapp-cloud/make_web.py` in the main repository and push.\n"
        )
    print(f"web deploy dir ready: {target}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(ROOT, "output", "soloscribe-web"))
