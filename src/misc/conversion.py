import os
from pathlib import Path

current_path = "."

for child in Path(current_path).iterdir():
    if child.is_dir():
        # Get prefix of the directory
        prefix = child.name.split(sep="_")[0]

        if prefix.isupper():
            # Print out the name
            print(f"{child.name}")

            # Replace the prefix with consistent naming
            new_name = child.name.replace(prefix, prefix.lower(), 1)

            os.rename(f"{current_path}/{child.name}", f"{current_path}/{new_name}")
