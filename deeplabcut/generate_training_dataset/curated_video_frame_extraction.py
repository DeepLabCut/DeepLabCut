import os
import cv2
import random
import argparse
import re

def time_to_seconds(time_str):
    """Convert time string in format HH:MM:SS to seconds"""
    if isinstance(time_str, (int, float)):
        return float(time_str)

    # Parse time in format HH:MM:SS
    match = re.match(r"(\d+):(\d+):(\d+)(?:\.(\d+))?", time_str)
    if match:
        hours, minutes, seconds, ms = match.groups()
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        if ms:
            total_seconds += float(f"0.{ms}")
        return total_seconds
    else:
        try:
            return float(time_str)
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS or seconds.")

def extract_curated_frames_from_video(video_path, start_time, end_time, min_gap, min_frames):
    video_obj = cv2.VideoCapture(video_path)

    # Get the video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print("Video Name: ", video_name)
    frames_folder = f"/labeled-data/{video_name}"
    output_image_folder = os.path.dirname(os.path.dirname(video_path)) + frames_folder
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    print("\n" + "="*50)

    # Get video properties
    frame_rate = video_obj.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / frame_rate
    print("Total Frames: ", total_frames)
    print(f"Duration: {duration:.2f} seconds")

    # Convert time strings to seconds
    start_time_sec = time_to_seconds(start_time)
    end_time_sec = time_to_seconds(end_time)

    print(f"Start time: {start_time} ({start_time_sec} seconds)")
    print(f"End time: {end_time} ({end_time_sec} seconds)")

    # Convert time to frames
    start_frame = max(0, round(start_time_sec * frame_rate))
    end_frame = min(total_frames - 1, round(end_time_sec * frame_rate))
    indices = list(range(start_frame, end_frame + 1))

    # Process min_gap
    if isinstance(min_gap, str) and '%' in min_gap:
        # If min_gap is a percentage
        try:
            percentage = float(min_gap.strip('%')) / 100
            min_gap_frames = max(1, round((end_frame - start_frame) * percentage))
            print(f"Using minimum gap of {percentage:.2%} ({min_gap_frames} frames)")
        except ValueError:
            print(f"Invalid percentage format: {min_gap}. Using default 1% gap.")
            min_gap_frames = max(1, round((end_frame - start_frame) * 0.01))
    else:
        # If min_gap is a number of frames
        try:
            min_gap_frames = int(min_gap)
            print(f"Using minimum gap of {min_gap_frames} frames")
        except ValueError:
            print(f"Invalid min_gap format: {min_gap}. Using default 1% gap.")
            min_gap_frames = max(1, round((end_frame - start_frame) * 0.01))

    # Select frames with minimum gap
    selected_frames = []
    attempts = 0
    max_attempts = len(indices) * 5  # Avoid infinite loop

    while len(selected_frames) < min_frames and indices and attempts < max_attempts:
        frame_number = random.choice(indices)
        if all(abs(frame_number - sf) >= min_gap_frames for sf in selected_frames):
            selected_frames.append(frame_number)
        indices.remove(frame_number)
        attempts += 1

    selected_frames.sort()  # Sort frames for readability
    print(f"Selected {len(selected_frames)} frames", "User selected", min_frames)

    # Extract the frames
    for frame_number in selected_frames:
        video_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video_obj.read()
        if ret:
            output_image_name = f"{video_name}_Frame_{frame_number}.jpg"
            cv2.imwrite(os.path.join(output_image_folder, output_image_name), frame)

    video_obj.release()
    print(f"\nFrames saved to: {output_image_folder}")

    # Log extraction details
    with open(os.path.join(output_image_folder, "extraction_log.txt"), "a") as f:
        f.write(f"Video: {video_name}\n")
        f.write(f"Video Path: {video_path}\n")
        f.write(f"Frame Rate: {round(frame_rate,0)}\n")
        f.write(f"Video Total Frames: {total_frames}\n")
        f.write(f"User Selected Time Range: {start_time} - {end_time}\n")
        if isinstance(min_gap, str) and '%' in min_gap:
            f.write(f"Min Gap: {min_gap} ({min_gap_frames} frames)\n")
        else:
            f.write(f"Min Gap: {min_gap_frames} frames\n")
        f.write(f"Extracted Curated Frames: {selected_frames}\n\n")
        f.write("Minimum Frames: {}\n".format(min_frames))
        f.write(f"Actual Selected Frames: {len(selected_frames)}\n")
        f.write("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument("--start_time", type=str, help="Start time in format HH:MM:SS or seconds.")
    parser.add_argument("--end_time", type=str, help="End time in format HH:MM:SS or seconds.")
    parser.add_argument("--min_gap", type=str, help="Minimum frame gap (number of frames or percentage with % sign).")
    parser.add_argument("--num_frames", type=int, default=15, help="Number of frames to extract.")

    args = parser.parse_args()

    extract_curated_frames_from_video(args.video_path, args.start_time, args.end_time, args.min_gap, args.num_frames)
