#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
  echo "Please provide a directory to process."
  echo "Usage: $0 <directory>"
  exit 1
fi

# Iterate through all Python files in the specified directory
find "$1" -type f -name "*.py" | while read -r file; do
  echo "Processing file: $file"
  
  # Remove unused imports
  autoflake --remove-all-unused-imports --in-place "$file"
  
  # Format the code using black
  black "$file"
done

echo "Unused imports removed and all Python files formatted."
