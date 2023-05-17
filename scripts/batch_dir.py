import os
import re
from PIL import Image

def save_image_dir(image, path, basename, extension='png'):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Generate the filename
    filename = f"{basename}.{extension}"
    full_path = os.path.join(path, filename)
    
    # Save the image
    image.save(full_path)

    return full_path
    
def modify_basename(basename):
    match = re.search(r'(\d+)(\.\w+)?$', basename)
    if match is not None:
        # If there is a sequence of digits followed by the file extension,
        # capture the prefix, the sequence number, and the extension separately.
        prefix = basename[:match.start()]
        sequence = match.group(1)
        extension = match.group(2) if match.group(2) else ''

        # If there's a hyphen or underscore just before the sequence number,
        # include it in the new name.
        if prefix and (prefix[-1] == '_' or prefix[-1] == '-'):
            separator = prefix[-1]
            return f"{prefix[:-1]}{separator}mask{separator}{sequence}{extension}"
        else:
            return f"{prefix}_mask{sequence}{extension}"
    else:
        # If there's no sequence number, use the last character of the string to decide the format.
        if basename and (basename[-1] == '_' or basename[-1] == '-'):
            return f"{basename}mask"
        else:
            return f"{basename}-mask"