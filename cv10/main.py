import cv2

from background_subtraction import BackgroundSubtraction
from cumshift import CumShift

SEGMENTATION_THRESHOLD = 0.7


def main():
    cap = cv2.VideoCapture("data/pvi_cv10_video_in.mp4")
    out = cv2.VideoWriter(
        "pvi_cv10_output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    prev_hist = None
    current_segment = 0

    bs = BackgroundSubtraction()
    cumshift = CumShift("data/pvi_cv10_vzor_pomeranc.bmp")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hist = cv2.calcHist(
            [frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )

        if prev_hist is None:
            prev_hist = hist
            continue

        hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

        if hist_diff > SEGMENTATION_THRESHOLD:
            current_segment += 1

        if current_segment in (0, 2):
            rect = bs.bbox(frame)

            if rect is not None:
                x_min, y_min, x_max, y_max = rect
                draw_bbox(frame, (x_min, y_min, x_max, y_max))

        elif current_segment == 1:
            x1,y1,x2,y2 = cumshift.detect(frame)

            draw_bbox(frame, (x1, y1, x2, y2))

        embed_segment_label(frame, current_segment)

        prev_hist = hist
        out.write(frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


def embed_segment_label(frame, segment):
    cv2.putText(
        frame,
        f"Segment {segment}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

if __name__ == "__main__":
    main()
