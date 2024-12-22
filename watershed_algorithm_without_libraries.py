import numpy as np
import heapq
import cv2
from matplotlib import pyplot as plt


def show_image(title, image, cmap_type=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis("off")
    plt.show()


def watershed_algorithm(image, markers, mask=None):
    shape = image.shape
    labels = np.zeros_like(image, dtype=np.int32)

    # Инициализируем очередь приоритетов
    priority_queue = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            if markers[x, y] > 0:
                heapq.heappush(priority_queue, (image[x, y], x, y))
                labels[x, y] = markers[x, y]

    # Направления соседей (восьмисвязные)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Основной цикл алгоритма
    while priority_queue:
        current_height, x, y = heapq.heappop(priority_queue)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1]:  # Проверка границ
                if mask is not None and not mask[nx, ny]:
                    continue
                if labels[nx, ny] == 0:  # Если точка еще не помечена
                    labels[nx, ny] = labels[x, y]
                    heapq.heappush(priority_queue, (image[nx, ny], nx, ny))

    return labels



image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ** Этап 1: Гауссово размытие **
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ** Этап 2: Пороговая обработка **
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ** Этап 3: Уточнение маркеров **
# Явное выделение фона
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations=3)

# Определение точных объектов
distance_map = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(distance_map, 0.2 * distance_map.max(), 255, 0)

# Определение "неопределенных" областей
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# ** Этап 4: Создание маркеров **
markers = cv2.connectedComponents(sure_fg)[1]
markers += 1  # Увеличиваем, чтобы фон был 1, а объекты >= 2
markers[unknown == 255] = 0  # Неопределенные области помечаем как 0


labels = watershed_algorithm(-distance_map, markers, mask=thresh)


output = np.zeros_like(image)
for label in np.unique(labels):
    if label <= 0:
        continue
    mask = (labels == label).astype("uint8")
    color = np.random.randint(0, 255, size=(3,), dtype="uint8")
    output[mask > 0] = color


show_image("Исходное изображение", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
show_image("Гауссово размытие", blurred, cmap_type="gray")
show_image("Пороговая обработка", thresh, cmap_type="gray")
show_image("Карта расстояний", distance_map, cmap_type="viridis")
show_image("Результат Watershed", output)
