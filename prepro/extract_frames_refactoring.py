import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FrameExtractDataset(Dataset):
    def __init__(
        self,
        video_root_dir: str,
        target_fps: int = 3,
    ) -> None:
        super().__init__()
        self.path_list = self.get_mp4_path_list(video_root_dir)
        self.target_fps = target_fps

    @staticmethod
    def get_mp4_path_list(video_root_dir: str) -> List[Path]:
        root_dir = Path(video_root_dir).absolute()
        path_list = list(root_dir.glob("**/*.mp4"))
        path_list = sorted(path_list)
        return path_list

    def get_frame_dir(self, video_path: Path) -> Path:
        frame_dir = video_path.parent / video_path.stem
        frame_dir.mkdir(exist_ok=True)
        return frame_dir

    def get_frame_info(self, video) -> Tuple[float, int]:
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, total_frame

    def extract_frame(self, frame_idx: int, frame_dir: str, video) -> None:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            print(f"Error read video - {frame_dir} / frame {frame_idx:06d}")

        frame_path = frame_dir / f"{frame_idx:06d}.jpg"
        self.save_frame(str(frame_path), frame)

    def extract_frames(self, video_path: str) -> None:
        frame_dir = self.get_frame_dir(video_path)
        video_path = str(video_path)
        video = cv2.VideoCapture(video_path)

        fps, total_frame = self.get_frame_info(video)
        step = round(fps / self.target_fps)

        frame_idx_list = list(range(0, total_frame, step))
        for frame_idx in frame_idx_list:
            self.extract_frame(frame_idx, frame_dir, video)

    @staticmethod
    def save_frame(path: str, frame):
        return cv2.imwrite(path, frame)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        video_path = self.path_list[idx]
        self.extract_frames(video_path)
        return str(video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root_dir", type=str, help="video root dir")
    parser.add_argument("--fps", type=str, default="3")
    args = parser.parse_args()

    dataset = FrameExtractDataset(
        video_root_dir=args.video_root_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=10,
    )
    for _ in tqdm(dataloader):
        pass
