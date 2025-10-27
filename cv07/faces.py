import cv2


def iou(boxA, boxB):
    # box = (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    return interArea / float(boxAArea + boxBArea - interArea)

def evaluate_detections(detected_boxes, ground_truth_boxes, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0

    matched_gt = set()

    for det in detected_boxes:
        match_found = False
        for i, gt in enumerate(ground_truth_boxes):
            if i in matched_gt:
                continue
            if iou(det, gt) >= iou_threshold:
                TP += 1
                matched_gt.add(i)
                match_found = True
                break
        if not match_found:
            FP += 1

    FN = len(ground_truth_boxes) - len(matched_gt)
    return TP, FP, FN


def get_boxes():
    boxes = []
    with open("data/pvi_cv07_boxes_01.txt") as f:
        lines = f.read().splitlines()
        for line in lines:
            vec = line.split(" ")
            vec = [int(x) for x in vec]
            boxes.append(vec)
    return boxes


def main():
    people = cv2.imread("data/pvi_cv07_people.jpg")
    faces_gray = cv2.cvtColor(people, cv2.COLOR_BGR2GRAY)

    boxes = get_boxes()

    for x, y, w, h in boxes:
        cv2.rectangle(people, (x, y), (x + w, y + h), (0, 0, 255), 2)

    face_cascade = cv2.CascadeClassifier(
        "data/pvi_cv07_haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        faces_gray,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for x, y, w, h in faces:
        cv2.rectangle(people, (x, y), (x + w, y + h), (0, 255, 0), 2)

    TP, FP, FN = evaluate_detections(faces, boxes)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = TP / len(boxes) if len(boxes) > 0 else 0

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    cv2.imshow("Detected Faces", people)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
