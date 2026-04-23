# zone.py
# ----------------------------------------
# Static zone definitions for your project
# Coordinate system: 
# Top-left     = (0, 0)
# Top-right    = (1280, 0)
# Bottom-left  = (0, 720)
# Bottom-right = (1280, 720)
# ----------------------------------------

ZONES = {
    # Right half of frame = danger (rectangle, clockwise)
    "danger_zone": {
        "points": [[640, 0], [1280, 0], [1280, 720], [640, 720]],
        "color": (0, 0, 255)   # Red
    },

    # Upper-left safe area (rectangle, clockwise)
    "safe_zone": {
        "points": [[0, 0], [640, 0], [640, 360], [0, 360]],
        "color": (0, 255, 0)   # Green
    },

    # Lower-left orange area (rectangle, clockwise)
    "orange_zone": {
        "points": [[0, 360], [640, 360], [640, 720], [0, 720]],
        "color": (0, 165, 255)   # Orange (BGR)
    }
}
