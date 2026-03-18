from pathlib import Path

from utils.general import (
    cv2,
)
def save_results(dataset, vid_path, vid_writer, vid_cap, i, im0, save_path):
    if dataset.mode == "image":
        cv2.imwrite(save_path, im0)
    else:  # 'video' or 'stream'
        if vid_path[i] != save_path:  # new video
            vid_path[i] = save_path
            if isinstance(vid_writer[i], cv2.VideoWriter):
                vid_writer[i].release()  # release previous video writer
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, im0.shape[1], im0.shape[0]
            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        vid_writer[i].write(im0) # update model (to fix SourceChangeWarning)
