"""Window output.

pyglet (OpenGL) is the default. Measured on macOS at 720p, per frame:
  OpenCV imshow + waitKey(1)       ~16.3 ms (waitKey alone ~15 ms: Cocoa event pump)
  pygame blit + flip, no vsync      ~2.9 ms
  pyglet texture upload + flip      ~2.3 ms
Upload is ~1.4 ms of that whatever the pixel format (BGR, BGRA, pyglet's ImageData or raw
glTexSubImage2D all measured within 0.1 ms): it is the cost of moving 2.7 MB to the GPU.
pyglet is also free of SDL. pygame ships its own SDL2, which clashes with the copy bundled
in every OpenCV wheel ("Class SDL... is implemented in both"). OpenCV is kept as a
dependency-light fallback (--display cv).
"""

from __future__ import annotations

import ctypes

import cv2
import numpy as np


class Display:
    def show(self, image: np.ndarray) -> list[str]:
        """Present a BGR frame. Returns the keys pressed since the last call (e.g. 'q', 'esc')."""
        raise NotImplementedError

    def toggle_fullscreen(self) -> None: ...

    def close(self) -> None: ...


def open_display(kind: str, title: str, size: tuple[int, int], vsync: bool = False) -> Display:
    if kind == "cv":
        return CvDisplay(title)
    return GlDisplay(title, size, vsync)


class GlDisplay(Display):
    def __init__(self, title: str, size: tuple[int, int], vsync: bool):
        import pyglet
        from pyglet import gl

        pyglet.options["debug_gl"] = False  # skips a glGetError after every GL call
        self.pyglet = pyglet
        self.gl = gl
        self.size = size
        self.window = pyglet.window.Window(*size, caption=title, vsync=vsync, resizable=True)
        self.texture = pyglet.image.Texture.create(*size)
        self._keys: list[str] = []
        self._closed = False

        @self.window.event
        def on_key_press(symbol, _modifiers):
            name = pyglet.window.key.symbol_string(symbol).lower()
            self._keys.append("esc" if name == "escape" else name)

        @self.window.event
        def on_close():
            self._closed = True
            return pyglet.event.EVENT_HANDLED  # we close it ourselves in close()

    def show(self, image: np.ndarray) -> list[str]:
        w, h = self.size
        if image.shape[1] != w or image.shape[0] != h:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        self.window.dispatch_events()
        # OpenGL's origin is bottom-left. Flipping with OpenCV (0.1 ms) is faster than letting
        # pyglet reorder rows for a negative pitch (+0.3 ms).
        flipped = cv2.flip(image, 0)
        gl = self.gl
        gl.glBindTexture(self.texture.target, self.texture.id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexSubImage2D(
            self.texture.target, 0, 0, 0, w, h, gl.GL_BGR, gl.GL_UNSIGNED_BYTE,
            flipped.ctypes.data_as(ctypes.c_void_p),
        )  # fmt: skip
        self.window.clear()
        self.texture.blit(0, 0, width=self.window.width, height=self.window.height)
        self.window.flip()

        keys, self._keys = self._keys, []
        if self._closed:
            keys.append("q")
        return keys

    def toggle_fullscreen(self) -> None:
        self.window.set_fullscreen(not self.window.fullscreen)

    def close(self) -> None:
        self.window.close()


class CvDisplay(Display):
    def __init__(self, title: str):
        self.title = title
        self.fullscreen = False
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(self, image: np.ndarray) -> list[str]:
        cv2.imshow(self.title, image)
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return []
        return ["esc" if key == 27 else chr(key)]

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        mode = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, mode)

    def close(self) -> None:
        cv2.destroyAllWindows()
