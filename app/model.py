import os
from random import shuffle

from app.config import raw_path, train_path, test_path
from app.img import copy

if __name__ == "__main__":
    # each mood is represented as a subfolder within the mood directory
    moods = [d for d in os.listdir(f"{raw_path}/mood") if not d.startswith(".")]
    for mood in moods:
        # get all the files for each mood
        files = [f for f in os.listdir(f"{raw_path}/mood/{mood}")]
        # randomize the files
        shuffle(files)
        for i, f in enumerate(files, 1):
            # move 70% of files into the training directory
            dest_path = f"{train_path}/mood/{mood}"
            if i > len(files) * 0.70:
                # the remaining 30% of files will go in to the test
                dest_path = f"{test_path}/mood/{mood}"
            # ensure dest path exists
            os.makedirs(dest_path, mode=0o775, exist_ok=True)
            # copy the files over
            source_file = f"{raw_path}/mood/{mood}/{f}"
            dest_file = f"{dest_path}/{f}"
            print(f"copying {source_file} to {dest_file}")
            copy(source_file, dest_file)
