import re

ALLOWED_OBJECT_COLORS = {"--", "green"}
ALLOWED_OBJECTS = {"--", "block"}
ALLOWED_POSITIONS = {0, 1, 2, 3}
ALLOWED_HEIGHTS = {0, 1, 2, 3}

def check_text_format(text: str) -> bool:
    """
    Returns True if the text matches the format:
    "put the {color} block at position {int}" 
    optionally followed by "at height {number}",
    and the color is in ALLOWED_OBJECT_COLORS. Otherwise False.
    """
    pattern = r'^put the (\S+) (\S+) at position (\d+)(?: at height (\d+(\.\d+)?))?$'
    match = re.match(pattern, text)
    
    if not match:
        return False
    
    color = match.group(1)
    object = match.group(2)
    position = int(match.group(3))
    height = int(match.group(4)) if match.group(4) is not None else None
    return (color in ALLOWED_OBJECT_COLORS 
            and object in ALLOWED_OBJECTS 
            and position in ALLOWED_POSITIONS 
            and (height is None or height in ALLOWED_HEIGHTS))

if __name__ == "__main__":
    # Test cases
    test_cases = [
        "put the red block at position 1",
        "put the blue sphere at position 2 at height 3",
        "put the green cylinder at position 4 at height 5",
        "put the yellow block at position 0",
        "put the purple block at position 1",
        "put the red block at position 6",
        "put the blue block at position 1 at height 5",
        "put the -- -- at position 1 at height 1"
    ]

    for case in test_cases:
        print(f"Testing: {case}")
        result = check_text_format(case)
        print(f"Result: {result}")