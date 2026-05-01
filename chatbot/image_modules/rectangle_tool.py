"""Rectangle drawing tool using Gradio HTML with HTML5 Canvas.

This module provides a custom rectangle drawing interface built on top of
gr.HTML. Users draw rectangles by clicking and dragging on a canvas that
displays the background image. Rectangles are sent to Python as coordinate
data via Gradio's trigger mechanism.

Usage Example:
    >>> from chatbot.image_modules.rectangle_tool import RectangleTool
    >>> tool = RectangleTool(label="Image 1")
    >>> tool.set_background(image_array)
    >>> html_update = tool.get_html_update()
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image


@dataclass
class RectangleToolConfig:
    """Configuration for the rectangle drawing tool.

    Attributes:
        colors: Available rectangle outline colors.
        default_color: Default selected color.
        min_rect_size: Minimum drag distance in pixels to create a rectangle.
        canvas_width: Max display width of the canvas.
        canvas_height: Max display height of the canvas.
    """

    colors: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("#FF0000", "Red"),
            ("#00FF00", "Green"),
            ("#0000FF", "Blue"),
        ]
    )
    default_color: str = "#FF0000"
    min_rect_size: int = 5
    canvas_width: int = 800
    canvas_height: int = 600


def image_to_data_url(image_array: np.ndarray) -> str:
    """Convert a numpy RGB image array to a base64 data URL.

    Args:
        image_array: RGB numpy array (H x W x 3).

    Returns:
        Data URL string: "data:image/png;base64,..."
    """
    pil_img = Image.fromarray(image_array)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class RectangleTool:
    """Manages state for a rectangle drawing canvas.

    Attributes:
        label: Display label for the tool.
        visible: Whether the tool is visible.
        background_b64: Base64 data URL of the background image (or empty).
        rectangles: List of drawn rectangle dicts.
        config: Tool configuration.
    """

    def __init__(
        self,
        label: str = "Rectangle Tool",
        visible: bool = False,
        config: RectangleToolConfig | None = None,
    ) -> None:
        self.label = label
        self.visible = visible
        self.background_b64: str = ""
        self.rectangles: list[dict] = []
        self.config = config or RectangleToolConfig()

    def set_background(self, image_array: np.ndarray | None) -> None:
        """Set the background image from a numpy array.

        Args:
            image_array: RGB numpy array (H x W x 3), or None to clear.
        """
        if image_array is not None:
            self.background_b64 = image_to_data_url(image_array)
        else:
            self.background_b64 = ""

    def set_rectangles(self, rects: list[dict]) -> None:
        """Set the list of rectangles.

        Args:
            rects: List of dicts with x1, y1, x2, y2, color keys.
        """
        self.rectangles = list(rects) if rects else []

    def get_rectangles(self) -> list[dict]:
        """Return the current list of rectangle dicts."""
        return list(self.rectangles)

    def clear_rectangles(self) -> None:
        """Remove all drawn rectangles."""
        self.rectangles = []

    def _get_value(self) -> dict:
        """Return the component value dict."""
        return {
            "background_b64": self.background_b64,
            "rects": self.rectangles,
            "label": self.label,
        }

    def get_html_update(self) -> gr.update:
        """Return a gr.update with the full HTML for this rectangle tool.

        Returns:
            gr.update with html_template, css_template, js_on_load, value, and visibility.
        """
        return gr.update(
            value=self._get_value(),
            visible=self.visible,
            html_template=_build_html_template(),
            css_template=_build_css_template(),
            js_on_load=_build_js_template(self.config),
        )


def _build_html_template() -> str:
    """Build the HTML template string for the rectangle canvas."""
    return """
<div id="rect-tool">
    <div id="toolbar">
        <span id="label-display">${value.label || "Rectangle Tool"}</span>
        <select id="color-select">
            <option value="#FF0000" selected>Red</option>
            <option value="#00FF00">Green</option>
            <option value="#0000FF">Blue</option>
        </select>
        <button id="undo-btn" type="button">Undo</button>
        <button id="clear-btn" type="button">Clear All</button>
        <span id="rect-count">0 rects</span>
    </div>
    <div id="canvas-wrapper">
        <canvas id="rect-canvas"></canvas>
    </div>
    <p id="instructions">Click and drag to draw rectangles on the image.</p>
</div>
"""


def _build_css_template() -> str:
    """Build the CSS template string for the rectangle canvas."""
    return """
#rect-tool { margin: 8px 0; }
#rect-tool #toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    padding: 6px 8px;
    background: var(--block-background-fill, #f7f7f8);
    border: 1px solid var(--block-border-color, #e5e7eb);
    border-radius: 6px;
    flex-wrap: wrap;
}
#rect-tool #label-display {
    font-weight: 600;
    font-size: 0.9em;
    margin-right: auto;
}
#rect-tool #color-select {
    padding: 3px 6px;
    border-radius: 4px;
    border: 1px solid var(--block-border-color, #ccc);
}
#rect-tool #undo-btn, #rect-tool #clear-btn {
    padding: 4px 10px;
    border: 1px solid var(--block-border-color, #ccc);
    border-radius: 4px;
    background: var(--button-secondary-background-fill, #fff);
    cursor: pointer;
    font-size: 0.85em;
}
#rect-tool #undo-btn:hover, #rect-tool #clear-btn:hover {
    background: var(--button-secondary-background-fill-hover, #eee);
}
#rect-tool #rect-count {
    font-size: 0.8em;
    color: var(--body-text-color-subdued, #888);
}
#rect-tool #canvas-wrapper {
    position: relative;
    overflow: auto;
    border: 2px solid var(--block-border-color, #e5e7eb);
    border-radius: 6px;
    background: #000;
}
#rect-tool canvas {
    display: block;
    cursor: crosshair;
    max-width: 100%;
}
#rect-tool #instructions {
    font-size: 0.8em;
    color: var(--body-text-color-subdued, #888);
    margin: 4px 0 0 0;
}
"""


def _build_js_template(config: RectangleToolConfig) -> str:
    """Build the JavaScript template string for canvas interaction.

    Args:
        config: Tool configuration with colors, min size, etc.
    """
    min_size = config.min_rect_size
    default_color = config.default_color
    return (
        """
(function() {
    const canvas = element.querySelector('#rect-canvas');
    const ctx = canvas.getContext('2d');
    const colorSelect = element.querySelector('#color-select');
    const undoBtn = element.querySelector('#undo-btn');
    const clearBtn = element.querySelector('#clear-btn');
    const rectCount = element.querySelector('#rect-count');

    const MIN_SIZE = """
        + str(min_size)
        + """;

    let img = new Image();
    let imgLoaded = false;
    let drawing = false;
    let startX = 0, startY = 0;
    let currentRects = [];

    // Initialize from data
    function init() {
        currentRects = (props.value && props.value.rects) ? JSON.parse(JSON.stringify(props.value.rects)) : [];
        const bg = props.value ? props.value.background_b64 : '';
        updateCount();
        if (bg) {
            img = new Image();
            img.onload = function() {
                imgLoaded = true;
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                redrawAll();
            };
            img.onerror = function() {
                imgLoaded = false;
            };
            img.src = bg;
        } else {
            imgLoaded = false;
            canvas.width = 400;
            canvas.height = 300;
            redrawAll();
        }
    }

    function updateCount() {
        rectCount.textContent = currentRects.length + ' rect' + (currentRects.length !== 1 ? 's' : '');
    }

    function redrawAll(preview) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (imgLoaded) {
            ctx.drawImage(img, 0, 0);
        } else {
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#666';
            ctx.font = '16px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No image loaded', canvas.width / 2, canvas.height / 2);
        }
        // Draw stored rectangles
        currentRects.forEach(function(r) {
            drawRect(r.x1, r.y1, r.x2, r.y2, r.color || '"""
        + default_color
        + """');
        });
        // Draw preview rectangle
        if (preview) {
            drawRect(preview.x1, preview.y1, preview.x2, preview.y2, preview.color);
        }
    }

    function drawRect(x1, y1, x2, y2, color) {
        const rx = Math.min(x1, x2);
        const ry = Math.min(y1, y2);
        const rw = Math.abs(x2 - x1);
        const rh = Math.abs(y2 - y1);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.fillStyle = color.replace(')', ', 0.12)').replace('rgb', 'rgba').replace('#', '');
        // Semi-transparent fill using hex-to-rgba
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.12)';
        ctx.fillRect(rx, ry, rw, rh);
    }

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: Math.round((e.clientX - rect.left) * scaleX),
            y: Math.round((e.clientY - rect.top) * scaleY)
        };
    }

    // Mouse events
    canvas.addEventListener('mousedown', function(e) {
        e.preventDefault();
        const pos = getPos(e);
        drawing = true;
        startX = pos.x;
        startY = pos.y;
    });

    canvas.addEventListener('mousemove', function(e) {
        if (!drawing) return;
        const pos = getPos(e);
        const color = colorSelect.value;
        redrawAll({x1: startX, y1: startY, x2: pos.x, y2: pos.y, color: color});
    });

    canvas.addEventListener('mouseup', function(e) {
        if (!drawing) return;
        drawing = false;
        const pos = getPos(e);
        const color = colorSelect.value;
        const w = Math.abs(pos.x - startX);
        const h = Math.abs(pos.y - startY);
        if (w >= MIN_SIZE || h >= MIN_SIZE) {
            const newRect = {x1: startX, y1: startY, x2: pos.x, y2: pos.y, color: color};
            currentRects.push(newRect);
            props.value = {background_b64: props.value.background_b64, rects: currentRects, label: props.value.label};
            trigger('change');
        }
        redrawAll();
        updateCount();
    });

    canvas.addEventListener('mouseleave', function() {
        if (drawing) {
            drawing = false;
            redrawAll();
        }
    });

    // Touch events for mobile
    canvas.addEventListener('touchstart', function(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const pos = getPos(touch);
        drawing = true;
        startX = pos.x;
        startY = pos.y;
    }, {passive: false});

    canvas.addEventListener('touchmove', function(e) {
        if (!drawing) return;
        e.preventDefault();
        const touch = e.touches[0];
        const pos = getPos(touch);
        const color = colorSelect.value;
        redrawAll({x1: startX, y1: startY, x2: pos.x, y2: pos.y, color: color});
    }, {passive: false});

    canvas.addEventListener('touchend', function(e) {
        if (!drawing) return;
        drawing = false;
        const touch = e.changedTouches[0];
        const pos = getPos(touch);
        const color = colorSelect.value;
        const w = Math.abs(pos.x - startX);
        const h = Math.abs(pos.y - startY);
        if (w >= MIN_SIZE || h >= MIN_SIZE) {
            const newRect = {x1: startX, y1: startY, x2: pos.x, y2: pos.y, color: color};
            currentRects.push(newRect);
            props.value = {background_b64: props.value.background_b64, rects: currentRects, label: props.value.label};
            trigger('change');
        }
        redrawAll();
        updateCount();
    });

    // Buttons
    undoBtn.addEventListener('click', function() {
        if (currentRects.length > 0) {
            currentRects.pop();
            props.value = {background_b64: props.value.background_b64, rects: currentRects, label: props.value.label};
            trigger('change');
            redrawAll();
            updateCount();
        }
    });

    clearBtn.addEventListener('click', function() {
        if (currentRects.length > 0) {
            currentRects = [];
            props.value = {background_b64: props.value.background_b64, rects: currentRects, label: props.value.label};
            trigger('change');
            redrawAll();
            updateCount();
        }
    });

    // Initialize
    init();

    // Watch for value changes from Gradio updates (e.g., navigation)
    let lastBg = props.value ? props.value.background_b64 : '';
    const watchInterval = setInterval(function() {
        const currentBg = props.value ? props.value.background_b64 : '';
        if (currentBg !== lastBg) {
            lastBg = currentBg;
            currentRects = [];
            init();
        }
    }, 300);

    // Clean up interval if element is removed
    const observer = new MutationObserver(function(mutations) {
        for (const m of mutations) {
            for (const node of m.removedNodes) {
                if (node === element || element.contains(node)) {
                    clearInterval(watchInterval);
                    observer.disconnect();
                    return;
                }
            }
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});
})();
"""
    )
