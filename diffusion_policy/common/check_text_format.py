import re
import itertools
import json
import pathlib



ALLOWED_OBJECTS = {"green block", "orange cylinder", "blue semicircle"}
ALLOWED_POSITIONS = {1, 2, 3}
ALLOWED_HEIGHTS = {1, 2, 3}

def check_text_format(text: str) -> bool:
    """
    Returns True if the text matches the format:
    "put the {color} block at position {int}" 
    optionally followed by "at height {number}",
    and the color is in ALLOWED_OBJECT_COLORS. Otherwise False.
    """
    pattern = r'^put the (\w+ \w+) at position (\d+)(?: at height (\d+(\.\d+)?))?$'
    match = re.match(pattern, text)
    if not match:
        return False
    
    
    object = match.group(1)
    position = int(match.group(2))
    height = int(match.group(3)) if match.group(3) is not None else None
    return (object in ALLOWED_OBJECTS 
            and position in ALLOWED_POSITIONS 
            and (height is None or height in ALLOWED_HEIGHTS))



def generate_balanced_configurations(no_of_repetitions=1, save_as_json = True):
    """
    Generates all balanced configurations of placing objects at positions.
    Each configuration is a list of placement commands.
    """
    configs = []
    # All permutations of objects assigned to positions
    for perm in itertools.permutations(ALLOWED_OBJECTS, len(ALLOWED_POSITIONS)):
        commands = []
        for pos, obj in zip(ALLOWED_POSITIONS, perm):
            text = f"put the {obj} at position {pos}"
            commands.append(text)
        for _ in range(no_of_repetitions):
            configs.append(commands)

    if save_as_json:
        import os
        json_array = json.dumps(configs, indent=4)
        output_path = os.environ.get(
            "BALANCED_CONFIGS_PATH",
            str(pathlib.Path(__file__).parent / "balanced_configs.json")
        )
        with open(output_path, 'w') as f:
            f.write(json_array)
    return configs




if __name__ == "__main__":
    configs = generate_balanced_configurations()
    #print(f"Generated {len(configs)} balanced configurations.")
    #for cfg in configs:
        #print(cfg)


# if __name__ == "__main__":
#     # Test cases
#     test_cases = [
#         "put the green block at position 1",
#         "put the blue sphere at position 2 at height 3",
#         "put the green cylinder at position 4 at height 5",
#         "put the yellow block at position 0",
#         "put the purple block at position 1",
#         "put the red block at position 6",
#         "put the blue block at position 1 at height 5",
#     ]

#     for case in test_cases:
#         print(f"Testing: {case}")
#         result = check_text_format(case)
#         print(f"Result: {result}")