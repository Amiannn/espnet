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

# Find files and folders starting with the prefix
FILES=$(find . -maxdepth 1 -name "${PREFIX}*" -printf "%P\n")

# Check if any files or folders are found
if [ -z "$FILES" ]; then
    echo "No files or folders with prefix '$PREFIX' found in '$DIRECTORY'."
    exit 1
fi

# Create a zip archive containing the found files and folders
ZIPFILE="${PREFIX}_files.zip"
zip -r "$ZIPFILE" $FILES

echo "Created zip file '$ZIPFILE' containing:"
echo "$FILES"
