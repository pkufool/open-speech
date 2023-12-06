# Copied and modified from https://github.com/wenet-e2e/wesubtitle/blob/main/wesubtitle/main.py

import argparse
import datetime
import glob
import logging
import os
import random
import subprocess

from multiprocessing import Pool

import cv2
import numpy as np
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity
import srt


def get_args():
    parser = argparse.ArgumentParser(description="simple subtitle")
    parser.add_argument(
        "--subsampling",
        type=int,
        default=8,
        help="subsampling rate, means processing the video every `subsampling` frame, just for accelerate the extraction.",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.8,
        help="similarity threshold, when the similarity less than this value, means the subtitle changes.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="""The number of workers.
        """,
    )
    parser.add_argument(
        "--ext", type=str, default="mp4", help="""The extension of video."""
    )
    parser.add_argument("video_dir", help="Directory containing video file")
    args = parser.parse_args()
    return args


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def extract_wav(file, output_file):
    logging.info(f"Extract wav : {file}")
    dir_name = os.path.dirname(output_file)
    os.makedirs(dir_name, exist_ok=True)
    if os.path.isfile(output_file):
        logging.warning(f"File exists: {output_file}")
        return
    command = (
        "ffmpeg -i '"
        + str(file)
        + f"' -af atempo=1.001 -ar 16000 -ac 1 '"
        + str(output_file)
        + "'"
    )
    print(command)
    subprocess.run(command, shell=True)


def toint(box):
    for i in range(len(box)):
        for j in range(len(box[i])):
            box[i][j] = int(box[i][j])
    return box


def detect_subtitle(ocr_results, image_size):
    """
    Args:
        ocr_results:
          The return object of PaddleOcr
    """
    ocr_results = ocr_results[0]  # 0, the first image result
    if ocr_results is None:
        return False, None, None
    expected_x = image_size[0] // 4 * 3
    # Merge text areas
    candidates = []
    max_width = 0
    max_idx = -1
    for idx, ocr_result in enumerate(ocr_results):
        boxes, text = ocr_result
        if (boxes[1][0] + boxes[0][0]) // 2 > expected_x:
            continue
        b_width = boxes[1][0] - boxes[0][0]
        if b_width > max_width:
            max_width = b_width
            max_idx = idx
    if max_idx == -1:
        return False, None, None
    max_boxes = ocr_results[max_idx][0]
    max_text = ocr_results[max_idx][1][0].strip()
    if len(max_text) == 0:
        return False, None, None
    width_diff = max_width / len(max_text) * 1.5  # width of one and half characters
    height_diff = (max_boxes[3][1] - max_boxes[0][1]) / 6
    idx = max_idx + 1
    while idx < len(ocr_results):
        boxes, (text, _) = ocr_results[idx]
        if (
            boxes[0][0] - max_boxes[1][0] < width_diff
            and abs(boxes[0][1] - max_boxes[1][1]) < height_diff
            and abs(boxes[3][1] - max_boxes[2][1]) < height_diff
        ):
            max_boxes[1] = boxes[1]
            max_boxes[2] = boxes[2]
            max_text += text.strip()
            logging.info(f"Add right extra text : {text}.")
            idx += 1
        else:
            break

    idx = max_idx - 1
    while idx >= 0:
        boxes, (text, _) = ocr_results[idx]
        if (
            max_boxes[0][0] - boxes[1][0] < width_diff
            and abs(boxes[1][1] - max_boxes[0][1]) < height_diff
            and abs(boxes[2][1] - max_boxes[3][1]) < height_diff
        ):
            max_boxes[0] = boxes[0]
            max_boxes[3] = boxes[3]
            max_text = text.strip() + max_text
            logging.info(f"Add left extra text : {text}.")
            idx -= 1
        else:
            break

    return True, toint(max_boxes), max_text.strip()


def locate_subtitle(videoCap, ocr, start_frames, total_frames, image_size):
    centers = []
    max_width = 0
    max_height = 0
    expected_x = image_size[0] // 2
    expected_y = image_size[1] // 5 * 4
    num_tries = 0
    while not len(centers) and num_tries < 5:
        num_tries += 1
        select_frames = [random.randint(start_frames, total_frames) for x in range(10)]
        for i in select_frames:
            videoCap.set(cv2.CAP_PROP_POS_FRAMES, i)
            res, frame = videoCap.read()
            if not res:
                continue
            result = ocr.ocr(frame, cls=False)
            if result[0] is None:
                continue
            for res in result[0]:
                box = res[0]
                middle_x = (box[0][0] + box[1][0]) // 2
                middle_y = (box[0][1] + box[3][1]) // 2
                if middle_x < expected_x and middle_y > expected_y:
                    centers.append((middle_x, middle_y))
                    width = box[1][0] - middle_x
                    height = box[3][1] - middle_y
                    if width > max_width:
                        max_width = width
                    if height > max_height:
                        max_height = height
    if len(centers) == 0:
        return False, (0, 0, 0, 0)
    X = sum([x[0] for x in centers]) // len(centers)
    Y = sum([x[1] for x in centers]) // len(centers)
    scale = 1.5
    start_x = image_size[0] // 8 * 1
    if X - max_width * 1.5 < start_x:
        start_x = max(0, X - max_width * 1.5)
    end_x = image_size[0] // 8 * 7
    return True, (
        int(start_x),
        int(end_x),
        int(Y - max_height * scale),
        int(Y + max_height * scale),
    )


def extract_srt(file, output_file):
    logging.info(f"Extract srt : {file}")
    dir_name = os.path.dirname(output_file)
    os.makedirs(dir_name, exist_ok=True)
    if os.path.isfile(output_file):
        logging.warning(f"File exists: {output_file}")
        return
    command = "ffmpeg -i '" + str(file) + f"' -y -c:s text '" + str(output_file) + "'"
    print(command)
    subprocess.run(command, shell=True)


def extract_video(args, input_video):
    wav_file = input_video[0 : -len(args.ext)] + "wav"
    extract_wav(input_video, wav_file)
    srt_file = input_video[0 : -len(args.ext)] + "srt"

    logging.info(f"srt file : {srt_file}")

    extract_srt(input_video, srt_file)

    dir_name = os.path.dirname(srt_file)
    os.makedirs(dir_name, exist_ok=True)
    if os.path.isfile(srt_file):
        logging.warning(f"File exists: {srt_file}")
        return

    ocr = PaddleOCR(
        lang="ch",
        show_log=False,
        use_gpu=False,
        use_angle_cls=False,
    )
    cap = cv2.VideoCapture(input_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Video info width: {w}, height: {h}, frames: {count}, fps: {fps}")

    cur = 0
    detected = False
    box = None
    content = ""
    start = 0
    ref_gray_image = None
    subs = []

    def _add_subs(end):
        if len(subs):
            pre_content = subs[-1].content
            if (
                (len(pre_content) == len(content) and pre_content == content)
                or (
                    len(pre_content) - len(content) == 1
                    and (pre_content[1:] == content or pre_content[0:-1] == content)
                )
                or (
                    len(content) - len(pre_content) == 1
                    and (pre_content == content[1:] or pre_content == content[0:-1])
                )
            ):
                subs[-1].end = datetime.timedelta(seconds=end / fps)
                subs[-1].content = (
                    content if len(content) > len(pre_content) else pre_content
                )
                return
        logging.info(f"New subtitle {start/fps} {end/fps} {content}")
        subs.append(
            srt.Subtitle(
                index=0,
                start=datetime.timedelta(seconds=start / fps),
                end=datetime.timedelta(seconds=end / fps),
                content=content.strip(),
            )
        )

    subtitle_start = min(int(fps * 60 * 4), count // 2)
    ret, (x1, x2, y1, y2) = locate_subtitle(cap, ocr, subtitle_start, count, (w, h))

    if ret:
        logging.info(f"subtitle area (xmin, xmax, ymin, ymax) : {(x1, x2, y1, y2)}")
    else:
        logging.error(f"No subtitle found for {input_video}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if detected:
                _add_subs(cur)
            break
        frame = frame[y1:y2, x1:x2, :]
        cur += 1
        if cur % args.subsampling != 0:
            continue
        if detected:
            # Compute similarity to reference subtitle area, if the result is
            # bigger than thresh, it's the same subtitle, otherwise, there is
            # changes in subtitle area
            hyp_gray_image = frame[box[1][1] : box[2][1], box[0][0] : box[1][0], :]
            hyp_gray_image = cv2.cvtColor(hyp_gray_image, cv2.COLOR_BGR2GRAY)
            similarity = structural_similarity(hyp_gray_image, ref_gray_image)
            if similarity > args.similarity_thresh:  # the same subtitle
                continue
            else:
                # Record current subtitle
                _add_subs(cur)
                detected = False
        else:
            # Detect subtitle area
            ocr_results = ocr.ocr(frame, cls=False)
            detected, box, content = detect_subtitle(ocr_results, (x2 - x1, y2 - y1))
            if detected:
                # the structural_similarity requires the input images larger than 7 * 7
                if box[2][1] - box[1][1] > 10 and box[1][0] - box[0][0] > 10:
                    start = cur
                    ref_gray_image = frame[
                        box[1][1] : box[2][1], box[0][0] : box[1][0], :
                    ]
                    ref_gray_image = cv2.cvtColor(ref_gray_image, cv2.COLOR_BGR2GRAY)
                else:
                    detected = False
    cap.release()

    # Write srt file
    with open(srt_file, "w", encoding="utf8") as fout:
        fout.write(srt.compose(subs))
    logging.info(f"Finish {input_video}.")


def main():
    args = get_args()
    videos = glob.glob(args.video_dir + f"/**/*.{args.ext}", recursive=True)

    pool = Pool(args.num_workers)
    params = [(args, os.path.abspath(x)) for x in videos]
    aync_results = pool.starmap_async(extract_video, params)
    results = aync_results.get()
    pool.close()
    pool.join()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
