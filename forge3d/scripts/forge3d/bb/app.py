"""Drive the real Blockbench desktop app (optional): prove a .bbmodel opens, look at it the way Blockbench draws
it, and use Blockbench's own exporters. Talks to a private, hidden instance over the Chrome DevTools protocol.
The instance has its own throw-away profile and no network access, so the user's Blockbench windows, settings,
recent files and installed version are left alone (in particular it cannot self-update the app)."""
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request


def find_blockbench():
    env = os.environ.get("BLOCKBENCH_PATH")
    cands = [env] if env else []
    if sys.platform == "darwin":
        cands += ["/Applications/Blockbench.app/Contents/MacOS/Blockbench",
                  os.path.expanduser("~/Applications/Blockbench.app/Contents/MacOS/Blockbench")]
    elif os.name == "nt":
        for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("PROGRAMFILES")):
            if base:
                cands += [os.path.join(base, "Programs", "Blockbench", "Blockbench.exe"),
                          os.path.join(base, "Blockbench", "Blockbench.exe")]
    else:
        cands += [shutil.which("blockbench"), "/usr/bin/blockbench", "/opt/Blockbench/blockbench"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


class _WS:
    """Minimal RFC 6455 client (text frames), enough for the DevTools protocol on localhost."""

    def __init__(self, url, timeout=60):
        assert url.startswith("ws://")
        host, _, path = url[5:].partition("/")
        h, _, p = host.partition(":")
        self.sock = socket.create_connection((h, int(p or 80)), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(("GET /%s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % (path, host, key)).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket handshake failed")
            buf += chunk
        if b" 101 " not in buf.split(b"\r\n", 1)[0]:
            raise ConnectionError("websocket handshake refused: %r" % buf[:80])
        self.rest = buf.split(b"\r\n\r\n", 1)[1]

    def send(self, text):
        data = text.encode()
        head = bytearray([0x81])
        n = len(data)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(data)) if n < 4096 else self._mask(data, mask)
        self.sock.sendall(bytes(head) + body)

    @staticmethod
    def _mask(data, mask):
        import numpy as np
        a = np.frombuffer(data, np.uint8)
        m = np.frombuffer((mask * (len(data) // 4 + 1))[:len(data)], np.uint8)
        return (a ^ m).tobytes()

    def _read(self, n):
        while len(self.rest) < n:
            chunk = self.sock.recv(max(65536, n - len(self.rest)))
            if not chunk:
                raise ConnectionError("websocket closed")
            self.rest += chunk
        out, self.rest = self.rest[:n], self.rest[n:]
        return out

    def recv(self):
        parts = []
        while True:
            b0, b1 = self._read(2)
            op, fin, n = b0 & 0x0F, b0 & 0x80, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            data = self._read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == 0x9:                                     # ping -> pong
                self.sock.sendall(bytes([0x8A, 0x80]) + os.urandom(4))
                continue
            if op == 0x8:
                raise ConnectionError("websocket closed by peer")
            if op in (0x0, 0x1, 0x2):
                parts.append(data)
                if fin:
                    return b"".join(parts).decode("utf-8", "replace")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class BlockbenchError(Exception):
    pass


class Blockbench:
    """A private Blockbench instance. Use as a context manager."""

    def __init__(self, exe=None, timeout=60):
        self.exe = exe or find_blockbench()
        if not self.exe:
            raise BlockbenchError("Blockbench is not installed (https://www.blockbench.net). The .bbmodel file is "
                                  "still valid; this check is optional. Set BLOCKBENCH_PATH if it lives elsewhere.")
        self.timeout = timeout
        self.proc = self.ws = self.profile = None
        self._id = 0

    def __enter__(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.profile = tempfile.mkdtemp(prefix="f3d_blockbench_")
        # The private instance is cut off from the network. Blockbench checks for updates at every start and, with
        # the default settings of a fresh profile, downloads a new version and installs it over the user's app when
        # the instance quits. Nothing this tool does needs the network, so no host name is allowed to resolve.
        args = ["--remote-debugging-port=%d" % self.port, "--userData", self.profile,
                "--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=http://127.0.0.1:9", "--proxy-bypass-list=<-loopback>"]
        if sys.platform == "darwin" and ".app/" in self.exe:
            app = self.exe.split(".app/")[0] + ".app"
            # -g -j: start in the background, hidden, without taking focus; -n: never reuse the user's instance
            self.proc = subprocess.Popen(["open", "-n", "-g", "-j", "-W", "-a", app, "--args"] + args,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self.proc = subprocess.Popen([self.exe] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        target = None
        while time.time() - t0 < self.timeout:
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/json" % self.port, timeout=2) as r:
                    pages = [p for p in json.loads(r.read().decode()) if p.get("type") == "page"]
                if pages:
                    target = pages[0]["webSocketDebuggerUrl"]
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.4)
        if not target:
            self.__exit__(None, None, None)
            raise BlockbenchError("Blockbench did not start within %d s" % self.timeout)
        self.ws = _WS(target, timeout=120)
        t0 = time.time()
        while time.time() - t0 < self.timeout:                   # wait until the app has finished booting
            try:
                if self.js("return typeof Codecs !== 'undefined' && !!Codecs.project && typeof Preview !== 'undefined' "
                           "&& Preview.all.length > 0 && typeof Formats !== 'undefined' && !!Formats.free"):
                    # second lock on the same door: this instance never agrees to download an update
                    self.js("try { settings.automatic_updates.value = false; } catch (e) {} return true;")
                    return self
            except BlockbenchError:
                pass
            time.sleep(0.4)
        self.__exit__(None, None, None)
        raise BlockbenchError("Blockbench started but its editor did not become ready")

    def js(self, body, **args):
        """Run an async function body inside Blockbench; `args` is available as the object ARGS."""
        self._id += 1
        expr = "(async (ARGS) => { %s })(%s)" % (body, json.dumps(args))
        self.ws.send(json.dumps({"id": self._id, "method": "Runtime.evaluate",
                                 "params": {"expression": expr, "awaitPromise": True, "returnByValue": True,
                                            "userGesture": True}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                res = msg.get("result", {})
                if "exceptionDetails" in res:
                    d = res["exceptionDetails"]
                    raise BlockbenchError("Blockbench script error: %s" % (
                        (d.get("exception") or {}).get("description") or d.get("text")))
                return res.get("result", {}).get("value")

    def __exit__(self, *exc):
        try:
            if self.ws is not None:
                try:
                    self._id += 1
                    self.ws.send(json.dumps({"id": self._id, "method": "Browser.close"}))
                    time.sleep(0.5)
                except Exception:  # noqa: BLE001
                    pass
                self.ws.close()
        finally:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
            if self.profile:
                shutil.rmtree(self.profile, ignore_errors=True)
        return False

    # ---------------------------------------------------------------- high level
    def open(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.js("""
            const model = JSON.parse(ARGS.text);
            Codecs.project.load(model, {path: ARGS.path, no_file: true});
            await new Promise(r => setTimeout(r, 400));
            Canvas.updateAll();
            const cubes = Outliner.elements.filter(e => e instanceof Cube);
            const meshes = Outliner.elements.filter(e => (typeof Mesh !== 'undefined') && e instanceof Mesh);
            const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
            const v = new THREE.Vector3();
            for (const c of cubes.concat(meshes)) { if (!c.mesh || !c.mesh.geometry || !c.mesh.geometry.attributes.position) continue; c.mesh.updateMatrixWorld(true);
              const p = c.mesh.geometry.attributes.position;
              for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i).applyMatrix4(c.mesh.matrixWorld);
                lo[0] = Math.min(lo[0], v.x); lo[1] = Math.min(lo[1], v.y); lo[2] = Math.min(lo[2], v.z);
                hi[0] = Math.max(hi[0], v.x); hi[1] = Math.max(hi[1], v.y); hi[2] = Math.max(hi[2], v.z); } }
            let problems = [];
            try { Validator.validate(); problems = (Validator.warnings || []).map(w => 'warning: ' + w.message)
                    .concat((Validator.errors || []).map(w => 'error: ' + w.message)); } catch (e) {}
            return {version: Blockbench.version, format: Format.id, format_name: Format.name, name: Project.name,
                    cubes: cubes.length, meshes: meshes.length,
                    mesh_faces: meshes.reduce((n, m) => n + Object.keys(m.faces).length, 0),
                    mesh_faces_untextured: meshes.reduce((n, m) => n + Object.values(m.faces).filter(f => !f.getTexture()).length, 0),
                    elements: Outliner.elements.length, groups: Group.all.length,
                    textures: Texture.all.map(t => ({name: t.name, width: t.width, height: t.height, error: t.error || 0})),
                    untextured_faces: cubes.reduce((n, c) => n + Object.values(c.faces).filter(f => f.texture === false).length, 0),
                    box: [lo, hi], problems: problems};
        """, text=text, path=os.path.abspath(path))

    def screenshot(self, out, angle="isometric_right", size=900):
        """Save what Blockbench's own viewport shows. angle: north (the model's front), south, east, west, top,
        isometric_right, isometric_left."""
        data = self.js("""
            const p = Preview.selected || Preview.all[0];
            const preset = DefaultCameraPresets.find(a => a.id === ARGS.angle);
            if (!preset) throw new Error('unknown angle ' + ARGS.angle);
            p.loadAnglePreset(preset);
            const box = new THREE.Box3();
            for (const c of Outliner.elements) { if (c.mesh) { c.mesh.updateMatrixWorld(true); box.expandByObject(c.mesh); } }
            if (!box.isEmpty()) {
              const centre = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
              const radius = Math.max(s.length() / 2, 1);
              const cam = p.camera, dir = new THREE.Vector3().subVectors(cam.position, p.controls.target).normalize();
              p.controls.target.copy(centre);
              if (p.isOrtho) { cam.position.copy(centre).addScaledVector(dir, radius * 4 + 32);
                const span = Math.min(Math.abs(cam.right - cam.left), Math.abs(cam.top - cam.bottom)); cam.zoom = span / (Math.max(s.x, s.y, s.z) * 1.25);
              } else { const fov = cam.fov * Math.PI / 180; cam.position.copy(centre).addScaledVector(dir, radius * 1.15 / Math.sin(fov / 2)); }
              cam.updateProjectionMatrix(); p.controls.update && p.controls.update();
            }
            p.render();
            return p.canvas.toDataURL('image/png');
        """, angle=angle)
        raw = base64.b64decode(data.split(",", 1)[1])
        import cv2
        import numpy as np
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        side = min(h, w)
        im = im[(h - side) // 2:(h - side) // 2 + side, (w - side) // 2:(w - side) // 2 + side]
        if side > size:
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out, im)
        return out

    def export(self, codec, out):
        """Write the open project with one of Blockbench's own exporters: bedrock (.geo.json), java_block (.json),
        modded_entity (.java), optifine_entity (.jem), gltf (.gltf), obj, fbx, collada (.dae), stl."""
        res = self.js("""
            const c = Codecs[ARGS.codec];
            if (!c) throw new Error('Blockbench has no exporter named ' + ARGS.codec + ' (have: ' + Object.keys(Codecs).join(', ') + ')');
            let content = await c.compile();
            if (content && typeof content === 'object' && !(content instanceof ArrayBuffer) && !ArrayBuffer.isView(content)) {
              if (content.obj) content = content.obj; else content = JSON.stringify(content);
            }
            if (content instanceof ArrayBuffer || ArrayBuffer.isView(content)) {
              const u8 = new Uint8Array(content.buffer || content); let s = '';
              for (let i = 0; i < u8.length; i += 32768) s += String.fromCharCode.apply(null, u8.subarray(i, i + 32768));
              return {b64: btoa(s)};
            }
            return {text: String(content)};
        """, codec=codec)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        if "b64" in res:
            with open(out, "wb") as f:
                f.write(base64.b64decode(res["b64"]))
        else:
            with open(out, "w", encoding="utf-8") as f:
                f.write(res["text"])
        return out
