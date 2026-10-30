from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections

OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "face_mesh.npz"
MODEL_URL = (
    "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)


def parse_obj(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns the depth and the (u, v) of each landmark. The file numbers them differently."""
    depth, texture, uv = [], [], {}
    for line in text.splitlines():
        kind, *fields = line.split() or [""]
        if kind == "v":
            depth.append(float(fields[2]))
        elif kind == "vt":
            texture.append((float(fields[0]), 1.0 - float(fields[1])))
        elif kind == "f":
            for corner in fields:
                vertex, tex = corner.split("/")[:2]
                uv[int(vertex) - 1] = texture[int(tex) - 1]
    return np.array(depth), np.array([uv[i] for i in range(len(depth))])


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else None
    text = Path(source).read_text() if source else urllib.request.urlopen(MODEL_URL).read().decode()
    depth, uv = parse_obj(text)

    # Eyes and mouth are holes in the tessellation, so the marks never paint over them.
    edges = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    triangles = np.array([[edge.start for edge in edges[i : i + 3]] for i in range(0, len(edges), 3)])

    # Same winding everywhere, so a triangle flipped on screen is one facing away from the camera.
    a, b, c = uv[triangles[:, 0]], uv[triangles[:, 1]], uv[triangles[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    triangles[cross < 0] = triangles[cross < 0][:, [0, 2, 1]]

    # Farthest first: drawn in this order, the nose covers the far cheek when the head turns.
    triangles = triangles[np.argsort(depth[triangles].mean(axis=1))]

    np.savez_compressed(OUTPUT, uv=uv.astype(np.float32), triangles=triangles.astype(np.int32))
    print(f"wrote {OUTPUT}: {len(uv)} points, {len(triangles)} triangles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
