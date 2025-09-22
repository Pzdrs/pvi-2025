import cv2

def get_centers():
    width = height = 600
    rows = cols = 3
    cell = width / cols  # 200

    centers = [
        ((c + 0.5) * cell, (r + 0.5) * cell)
        for r in range(rows)
        for c in range(cols)
    ]
    return centers

colors = {
    "black":  lambda h, s, v: v < 50,
    "white":  lambda h, s, v: v > 200 and s < 30,
    "gray":   lambda h, s, v: 50 <= v <= 200 and s < 30,
    "red":    lambda h, s, v: (h < 10 or h >= 170) and s > 100 and v > 100,
    "orange": lambda h, s, v: 10 <= h < 20 and s > 100 and v > 100,
    "yellow": lambda h, s, v: 20 <= h < 35 and s > 100 and v > 100,
    "green":  lambda h, s, v: 35 <= h < 85 and s > 50 and v > 50,
    "blue":   lambda h, s, v: 90 <= h < 130 and s > 50 and v > 50,
    "purple": lambda h, s, v: 130 <= h < 160 and s > 100 and v > 50,
    "pink":   lambda h, s, v: 160 <= h < 179 and s > 50 and v > 150,
}


def main():
    image_data = cv2.imread("data/cv02_01.bmp")
    image_hsv = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
    
    centers = get_centers()

    for (x, y) in centers:
        h, s, v = image_hsv[int(y), int(x)]
        for color_name, condition in colors.items():
            if condition(h, s, v):
                cv2.putText(image_data, color_name, (int(x) - 50, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) if color_name == "black" else (0, 0, 0), 2)
                break
    
    cv2.imshow("Image with Colors", image_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()