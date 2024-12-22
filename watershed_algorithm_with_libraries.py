import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(title, image, cmap_type=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Загрузка изображения и преобразование в оттенки серого
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Гауссово размытие для устранения шумов
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Пороговая обработка
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Уточнение фона и объектов
# Увеличение областей фона
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations=3)

# Определение точных объектов с помощью карты расстояний
distance_map = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(distance_map, 0.2 * distance_map.max(), 255, 0)

# Вычисление "неопределенных" областей
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Создание маркеров
markers = cv2.connectedComponents(sure_fg)[1]
markers += 1  # Увеличиваем, чтобы фон был 1, а объекты >= 2
markers[unknown == 255] = 0  # Неопределенные области помечаем как 0

# Применение метода водораздела
markers = cv2.watershed(image, markers)

# Визуализация результатов
output = image.copy()
for label in np.unique(markers):
    if label == -1:  # Границы
        output[markers == label] = [0, 0, 255]  # Красные границы
    elif label == 1:  # Фон
        continue
    else:
        mask = (markers == label).astype("uint8")
        color = np.random.randint(0, 255, size=(3,), dtype="uint8")
        output[mask > 0] = color

# Отображение изображений
show_image("Исходное изображение", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
show_image("Пороговая обработка", thresh, cmap_type="gray")
show_image("Карта расстояний", distance_map, cmap_type="viridis")
show_image("Результат Watershed (библиотечный)", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
