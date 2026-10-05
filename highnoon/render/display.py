"""Window output.

pygame (SDL) is the default. Measured on macOS at 720p:
  OpenCV imshow + waitKey(1)   ~16.3 ms per frame (waitKey alone ~15 ms: Cocoa event pump)
  pygame blit + flip, no vsync  ~2.9 ms per frame
That is ~13 ms less capture-to-screen latency, and pygame also gives us sound and fullscreen
for the game. OpenCV is kept as a dependency-light fallback (--display cv).
"""

from __future__ import annotations

import os

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
    return PygameDisplay(title, size, vsync)


class PygameDisplay(Display):
    def __init__(self, title: str, size: tuple[int, int], vsync: bool):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.display.init()
        pygame.display.set_caption(title)
        self.size = size
        # SCALED lets SDL scale on the GPU when the window is resized or fullscreen.
        self.screen = pygame.display.set_mode(size, pygame.SCALED | pygame.RESIZABLE, vsync=int(vsync))

    def show(self, image: np.ndarray) -> list[str]:
        pg = self.pg
        h, w = image.shape[:2]
        if (w, h) != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
            h, w = image.shape[:2]
        # frombuffer wraps the numpy memory without copying; 'BGR' avoids a colour conversion.
        surface = pg.image.frombuffer(np.ascontiguousarray(image).data, (w, h), "BGR")
        self.screen.blit(surface, (0, 0))
        pg.display.flip()

        keys = []
        for event in pg.event.get():
            if event.type == pg.QUIT:
                keys.append("q")
            elif event.type == pg.KEYDOWN:
                keys.append("esc" if event.key == pg.K_ESCAPE else pg.key.name(event.key))
        return keys

    def toggle_fullscreen(self) -> None:
        self.pg.display.toggle_fullscreen()

    def close(self) -> None:
        self.pg.display.quit()


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
