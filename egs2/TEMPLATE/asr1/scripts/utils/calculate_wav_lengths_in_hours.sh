#!/bin/bash

# Check if sox is installed
if ! command -v sox &> /dev/null; then
    echo "Error: sox is not installed. Please install it and try again."
    exit 1
fi

declare -A total_durations
grand_total=0

for dataset in dev test train; do
    scp_file="data/$dataset/wav.scp"
    dataset_total=0

    echo "Processing $scp_file..."

    while read -r line; do
        wav_path=$(echo "$line" | awk '{print $2}')

        # Get the duration of the WAV file using sox
        duration=$(sox --i -D "$wav_path" 2>/dev/null)

        if [ -z "$duration" ]; then
            echo "Warning: Failed to get duration for $wav_path"
            continue
        fi

        # Sum up the durations
        dataset_total=$(echo "$dataset_total + $duration" | bc)
    done < "$scp_file"

    # Convert total duration to hours
    duration_hours=$(echo "scale=4; $dataset_total / 3600" | bc)

    total_durations[$dataset]=$duration_hours
    grand_total=$(echo "$grand_total + $duration_hours" | bc)

    # Format and display the duration in hours
    printf "Total duration for %s: %.2f hours\n" "$dataset" "${total_durations[$dataset]}"
done

# Format and display the grand total duration
printf "Grand total duration: %.2f hours\n" "$grand_total"
