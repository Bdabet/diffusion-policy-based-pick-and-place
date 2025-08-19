import re

ALLOWED_OBJECT_COLORS = {"red", "blue", "green", "yellow"}

def check_text_format(text: str) -> bool:
    """
    Returns True if the text matches the format:
    "put the {color} block at position {int}" 
    optionally followed by "at height {number}",
    and the color is in ALLOWED_COLORS. Otherwise False.
    """
    pattern = r'^put the (\w+) block at position (\d+)(?: at height (\d+(\.\d+)?))?$'
    match = re.match(pattern, text)
    
    if not match:
        return False
    
    color = match.group(1)
    return color in ALLOWED_OBJECT_COLORS