#!/bin/bash

# Usage: ./rename_images.sh <folder_path>

set -e

DIR="$1"

if [ -z "$DIR" ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

if [ ! -d "$DIR" ]; then
  echo "Error: Directory does not exist."
  exit 1
fi

cd "$DIR"

# Step 1: Find the highest existing index
max_id=-1

for f in img_[0-9][0-9][0-9][0-9].*; do
  [ -e "$f" ] || continue
  num=$(echo "$f" | sed -E 's/img_([0-9]{4})\..*/\1/')
  num=$((10#$num))  # handle leading zeros
  if [ "$num" -gt "$max_id" ]; then
    max_id=$num
  fi
done

# Start index
counter=$((max_id + 1))

# Step 2: Rename other images
for file in *; do
  # Skip directories
  [ -f "$file" ] || continue

  # Skip already renamed files
  if [[ "$file" =~ ^img_[0-9]{4}\..+ ]]; then
    continue
  fi

  # Extract extension
  ext="${file##*.}"

  # Generate new name
  new_name=$(printf "img_%04d.%s" "$counter" "$ext")

  # Avoid overwrite
  while [ -e "$new_name" ]; do
    counter=$((counter + 1))
    new_name=$(printf "img_%04d.%s" "$counter" "$ext")
  done

  mv "$file" "$new_name"

  echo "Renamed: $file -> $new_name"

  counter=$((counter + 1))

  # Stop if exceeding limit
  if [ "$counter" -gt 9999 ]; then
    echo "Reached maximum limit (img_9999). Stopping."
    break
  fi
done

echo "Done."
