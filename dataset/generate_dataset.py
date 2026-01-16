
from huggingface_hub import HfApi, hf_hub_download
import argparse
from convert_mp4 import render_video
import hashlib


def fetch_csv_files(repo_id="p-doom/crowd-code-0.1", repo_type="dataset"):
    api = HfApi()
    print("Fetching file list from repository...")
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    csv_files = [f for f in files if f.endswith('.csv')]
    return csv_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Render coding traces to MP4.")
    parser.add_argument("--speed", type=float, default=20.0, help="Playback speed multiplier.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--num_videos", type=int, default=350, help="Number of videos to generate from the dataset.")

    args = parser.parse_args()

    repo_id = "p-doom/crowd-code-0.1"
    repo_type = "dataset"
    csv_files = fetch_csv_files(repo_id=repo_id, repo_type=repo_type)
    print(f"Total CSV files found: {len(csv_files)}")

    for idx, csv_file in enumerate(csv_files, 1):
        if idx > args.num_videos:
            break

        print(f"Processing {idx}/{args.num_videos}: {csv_file}")
        
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=csv_file,
            repo_type=repo_type
        )

        output_filename = f"/data/vid_{hashlib.sha256(csv_file.replace('/', '_').replace('.csv', '').encode()).hexdigest()[:15]}.mp4"
        render_video(file_path, output_filename, args.speed, args.width, args.height, labels_only=True)

    print(f"\nCompleted processing {args.num_videos} CSV files")
