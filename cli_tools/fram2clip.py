import argparse
import os
import natsort
from moviepy import ImageSequenceClip


def create_video_from_frames(input_folder, output_file, fps=30, codec="libx265"):
    """
    Creates an MP4 video from image frames in a folder.

    Args:
        input_folder (str): Path to the folder containing image frames.
        output_file (str): Path to the output MP4 video file.
        fps (int): Frames per second for the output video.
        codec (str): Video codec to use (e.g., 'libx265' for HEVC, 'libx264' for H.264).
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    try:
        frame_files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(image_extensions)
        ]
    except OSError as e:
        print(f"Error reading input folder: {e}")
        return

    if not frame_files:
        print(f"Error: No image files found in {input_folder}")
        return

    # Sort files naturally to handle names like frame1.png, frame10.png correctly
    sorted_frame_files = natsort.natsorted(frame_files)

    print(f"Found {len(sorted_frame_files)} frames.")
    print(f"Creating video: {output_file} at {fps} FPS using {codec} codec...")

    try:
        clip = ImageSequenceClip(sorted_frame_files, fps=fps)
        ffmpeg_params = ["-pix_fmt", "yuv420p"]
        clip.write_videofile(output_file, codec=codec, logger="bar", ffmpeg_params=ffmpeg_params)  # Use 'bar' for progress
        clip.close()  # Close the clip to release resources
        print(f"Video successfully created: {output_file}")
    except Exception as e:
        print(f"Error creating video: {e}")
        print("Please ensure ffmpeg is installed and accessible by moviepy.")
        print("You might need to install codecs like libx265 (HEVC) or libx264 (H.264).")
        print("Try using codec='libx264' if 'libx265' fails.")


def main():
    parser = argparse.ArgumentParser(description="Create an MP4 video from a sequence of image frames.")
    parser.add_argument("input_folder", help="Path to the folder containing image frames.")
    parser.add_argument("output_file", help="Path for the output MP4 video file.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video (default: 30).")
    parser.add_argument("--codec", type=str, default="libx264", help="Video codec (default: 'libx265' for HEVC). Use 'libx264' for H.264 if needed.")

    args = parser.parse_args()

    # Ensure output directory exists if specified within a path
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    create_video_from_frames(args.input_folder, args.output_file, args.fps, args.codec)


if __name__ == "__main__":
    # Ensure natsort is installed: pip install natsort moviepy
    main()
