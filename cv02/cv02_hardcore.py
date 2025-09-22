import collections
import cv2

colors = {
    "black":  lambda h, s, v: v < 50,
    "white":  lambda h, s, v: v > 200 and s < 30,
    "gray":   lambda h, s, v: 50 <= v <= 200 and s < 30,
    "red":    lambda h, s, v: (h < 10 or h >= 170) and s > 100 and v > 100,
    "orange": lambda h, s, v: 10 <= h < 20 and s > 100 and v > 100,
    "yellow": lambda h, s, v: 20 <= h < 35 and s > 100 and v > 100,
    "green":  lambda h, s, v: 35 <= h < 85 and s > 50 and v > 50,
    "blue":   lambda h, s, v: 90 <= h < 130 and s > 50 and v > 50,
    "purple": lambda h, s, v: 140 <= h < 160 and s > 100 and v > 50,
    "pink":   lambda h, s, v: 160 <= h < 179 and s > 50 and v > 150,
}

images = (
    "data/cv01_auto.jpg",
    "data/cv01_jablko.jpg",
    "data/cv01_mesic.jpg"
)

def detect_color(h, s, v):
    for color_name, condition in colors.items():
        if condition(h, s, v):
            return color_name
    return "unknown"

def main():
    for path in images:
        image_data = cv2.imread(path)
        image_hsv = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)

        total_pixels = image_hsv.shape[0] * image_hsv.shape[1]

        color_counter = collections.Counter()
        for y in range(image_hsv.shape[0]):
            for x in range(image_hsv.shape[1]):
                h, s, v = image_hsv[y, x]
                color_name = detect_color(h, s, v)
                color_counter[color_name] += 1

        print(color_counter.most_common(3))
        for i, (color_name, count) in enumerate(color_counter.most_common(3)):
            cv2.putText(image_data, f"{color_name}: {count/total_pixels:.2%}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(f"Image: {path}", image_data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

