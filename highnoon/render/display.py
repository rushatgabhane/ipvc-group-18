from __future__ import annotations

import ctypes

import cv2
import numpy as np
import pyglet
from pyglet import gl

pyglet.options["debug_gl"] = False
SHAKE_ZOOM_PX = 32


class Display:
    def __init__(self, title: str, size: tuple[int, int], vsync: bool):
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
            return pyglet.event.EVENT_HANDLED

    def show(self, image: np.ndarray, shake: tuple[int, int] = (0, 0)) -> list[str]:
        self.window.dispatch_events()
        self._upload(image)
        self.window.clear()
        self._draw_texture(*shake)
        self.window.flip()
        keys, self._keys = self._keys, []
        if self._closed:
            keys.append("q")
        return keys

    def _draw_texture(self, shake_x: int, shake_y: int) -> None:
        width, height = self.window.width, self.window.height
        if not (shake_x or shake_y):
            self.texture.blit(0, 0, width=width, height=height)
            return
        margin = SHAKE_ZOOM_PX
        self.texture.blit(
            shake_x - margin, -shake_y - margin, width=width + 2 * margin, height=height + 2 * margin
        )

    def _upload(self, image: np.ndarray) -> None:
        width, height = self.size
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        bottom_up = cv2.flip(image, 0)
        gl.glBindTexture(self.texture.target, self.texture.id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexSubImage2D(
            self.texture.target,
            0,
            0,
            0,
            width,
            height,
            gl.GL_BGR,
            gl.GL_UNSIGNED_BYTE,
            bottom_up.ctypes.data_as(ctypes.c_void_p),
        )

    def toggle_fullscreen(self) -> None:
        self.window.set_fullscreen(not self.window.fullscreen)

    def close(self) -> None:
        self.window.close()
