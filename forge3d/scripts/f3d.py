#!/usr/bin/env python3
"""forge3d launcher. Stdlib only, runs on any Python >= 3.8.

Creates (once) a private Python 3.13 environment holding headless Blender (the
`bpy` wheel) plus the imaging stack, then runs the requested forge3d command in
it. No Blender install is needed.

    python3 f3d.py doctor
    python3 f3d.py build model.py --ref ref.png:front
    python3 f3d.py python any_script.py      # raw access to the runtime
"""
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.environ.get("F3D_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "forge3d")
VENV = os.path.join(HOME, "venv")
REQ = os.path.join(HERE, "requirements.txt")
PY_VERSION = "3.13"
EXTRAS = {
    # background removal for photos with busy backgrounds (first use downloads a ~170 MB model)
    "rembg": ["rembg==2.0.87", "onnxruntime==1.31.1"],
    # hosted image-to-3D through Hugging Face Spaces
    "gen3d": ["gradio_client==2.7.2"],
}
# Supply-chain guard: uv resolves nothing published after this date, so unpinned transitive dependencies cannot
# pull in a release that is only days old. Move it forward together with the pins above and in requirements.txt.
EXCLUDE_NEWER = "2026-11-08"


def log(msg):
    print("[f3d] " + msg, file=sys.stderr, flush=True)


def venv_python():
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")


def find_uv():
    cands = [shutil.which("uv")]
    home = os.path.expanduser("~")
    cands += [os.path.join(home, ".local", "bin", "uv"), os.path.join(home, ".cargo", "bin", "uv"),
              "/opt/homebrew/bin/uv", "/usr/local/bin/uv"]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def req_hash():
    with open(REQ, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def marker(name="core"):
    tag = req_hash() if name == "core" else hashlib.sha1(" ".join(EXTRAS[name]).encode()).hexdigest()[:12]
    return os.path.join(VENV, ".f3d-%s-%s" % (name, tag))


def run(cmd):
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit("[f3d] command failed (exit %d): %s" % (r.returncode, " ".join(cmd)))


def pip_install(args):
    uv = find_uv()
    if uv:
        run([uv, "pip", "install", "--python", venv_python(), "--exclude-newer", EXCLUDE_NEWER] + args)
    else:
        run([venv_python(), "-m", "pip", "install"] + args)


def ensure_core():
    if os.path.exists(marker()) and os.path.exists(venv_python()):
        return
    os.makedirs(HOME, exist_ok=True)
    uv = find_uv()
    if not os.path.exists(venv_python()):
        log("first run: creating runtime in %s (downloads ~400 MB once, takes a few minutes)" % VENV)
        if uv:
            run([uv, "venv", "--python", PY_VERSION, VENV])
        else:
            py = shutil.which("python" + PY_VERSION)
            if not py and sys.version_info[:2] == (3, 13):
                py = sys.executable
            if not py:
                raise SystemExit(
                    "[f3d] need `uv` (recommended) or python%s to create the runtime.\n"
                    "      Install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh   (or: brew install uv / pip install uv)"
                    % PY_VERSION)
            run([py, "-m", "venv", VENV])
    pip_install(["-r", REQ])
    for f in os.listdir(VENV):
        if f.startswith(".f3d-core-"):
            os.remove(os.path.join(VENV, f))
    open(marker(), "w").close()
    log("runtime ready")


def ensure_extra(name):
    if name not in EXTRAS:
        raise SystemExit("[f3d] unknown extra %r (known: %s)" % (name, ", ".join(sorted(EXTRAS))))
    ensure_core()
    if os.path.exists(marker(name)):
        return
    log("installing optional component %r" % name)
    pip_install(EXTRAS[name])
    open(marker(name), "w").close()


def child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["F3D_HOME"] = HOME
    env["F3D_SKILL_DIR"] = os.path.dirname(HERE)
    env["F3D_LAUNCHER"] = os.path.abspath(__file__)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")      # keep the skill folder free of __pycache__
    return env


def exec_in_venv(args):
    cmd = [venv_python()] + args
    if os.name == "nt":
        raise SystemExit(subprocess.run(cmd, env=child_env()).returncode)
    os.execve(cmd[0], cmd, child_env())


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        ensure_core()
        exec_in_venv(["-m", "forge3d.cli", "--help"])
    cmd, rest = argv[0], argv[1:]
    if cmd == "setup":
        ensure_core()
        i = 0
        while i < len(rest):
            if rest[i] == "--extra" and i + 1 < len(rest):
                ensure_extra(rest[i + 1])
                i += 2
            else:
                raise SystemExit("[f3d] usage: f3d.py setup [--extra rembg|gen3d]")
        log("ok: " + VENV)
        return
    if cmd == "python":
        ensure_core()
        exec_in_venv(rest)
    ensure_core()
    # optional components install themselves the first time the command that needs them is used
    if cmd == "gen3d" and "--list" not in rest and "-h" not in rest and "--help" not in rest:
        ensure_extra("gen3d")
    if cmd == "ref" and any(rest[i] == "--bg" and rest[i + 1] == "rembg" for i in range(len(rest) - 1)):
        ensure_extra("rembg")
    exec_in_venv(["-m", "forge3d.cli", cmd] + rest)


if __name__ == "__main__":
    main(sys.argv[1:])
