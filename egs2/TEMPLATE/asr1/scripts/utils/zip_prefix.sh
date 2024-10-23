#!/bin/bash

# Check if the correct number of arguments is supplied
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/directory prefix"
    exit 1
fi

# Assign arguments to variables
DIRECTORY="$1"
PREFIX="$2"

# Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "The directory '$DIRECTORY' does not exist."
    exit 1
fi

# Navigate to the specified directory
cd "$DIRECTORY" || exit

# Find directories starting with the prefix
DIRS=$(find . -type d -name "${PREFIX}*")

# Check if any directories are found
if [ -z "$DIRS" ]; then
    echo "No directories with prefix '$PREFIX' found in '$DIRECTORY'."
    exit 1
fi

# Create a zip archive containing the found directories
ZIPFILE="${PREFIX}_folders.zip"
zip -r "$ZIPFILE" $DIRS

echo "Created zip file '$ZIPFILE' containing directories:"
echo "$DIRS"
