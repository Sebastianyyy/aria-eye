#!/bin/bash

#TRAIN SET
TEST_FILE="./test_set.txt"

CONFIG_PATH="./aria_everyday_activities_dataset_download_urls.json"
OUTPUT_PATH="./downloads_test"
check_exists() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "Error: '$path' does not exist."
        exit 1
    fi
}

check_exists "$TEST_FILE"
check_exists "$CONFIG_PATH"

if [[ -e "$OUTPUT_PATH" && ! -d "$OUTPUT_PATH" ]]; then
    echo "Error: '$OUTPUT_PATH' exists but is not a directory."
    exit 1
fi

if [[ ! -d "$OUTPUT_PATH" ]]; then
    echo "'$OUTPUT_PATH' does not exist. Creating directory."
    mkdir -p "$OUTPUT_PATH"
fi


echo "Reading test data from $TEST_FILE:"
while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -n "$line" ]]; then  # Skip empty lines
        echo "Processing: $line"
        aea_dataset_downloader --cdn_file "$CONFIG_PATH" --output_folder "$OUTPUT_PATH" --sequence_names "$line" --data_types 0 1

    fi
done < "$TEST_FILE"

echo "All tasks completed successfully."

read -p "Press Enter to exit..."

