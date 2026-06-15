import cv2
from pathlib import Path


def video_to_images(
    video_path: str,
    output_dir: str,
    every_n_frames: int = 30,
    max_frames: int = None,
    prefix: str = "frame",
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[Video] {video_path}")
    print(f"[Video] total_frames={total_frames}, fps={fps:.2f}")
    print(f"[Extract] every {every_n_frames} frames")

    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % every_n_frames == 0:
            out_name = f"{prefix}_{frame_id:06d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if max_frames is not None and saved >= max_frames:
                break

        frame_id += 1

    cap.release()
    print(f"[Done] saved {saved} images to {output_dir}")


if __name__ == "__main__":
    # ======= 配置区域（按需修改） =======
    video_path = r"E:\detected sys\highway_perception_v2\calib_videos\MP4\D3.mp4"
    output_dir = r"E:\detected sys\highway_perception_v2\assets\images"

    every_n_frames = 30      # 30fps 视频 ≈ 每秒 1 张
    max_frames = None        # 不限制

    video_to_images(
        video_path=video_path,
        output_dir=output_dir,
        every_n_frames=every_n_frames,
        max_frames=max_frames,
        prefix="calib",
    )
