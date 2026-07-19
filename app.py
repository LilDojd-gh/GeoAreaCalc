import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

# Импортируем архитектуры и функции расчета из нашего пакета src
from src import UNetSegmentation, get_vehicle_detection_model, calculate_gsd_from_cars


# ==========================================
# 1. Загрузка моделей с кэшированием
# ==========================================
@st.cache_resource
def load_models():
    """
    Загружает веса моделей один раз и сохраняет их в памяти Streamlit,
    чтобы приложение не перезагружало тяжелые файлы при каждом клике.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Инициализация и загрузка U-Net
    unet = UNetSegmentation(in_channels=3, out_channels=1)
    unet_path = "weights/unet_best.pth"
    if os.path.exists(unet_path):
        unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.to(device)
    unet.eval()  # Перевод в режим инференса

    # 2. Инициализация и загрузка детектора Faster R-CNN
    detector = get_vehicle_detection_model(num_classes=2)
    detector_path = "weights/detector_best.pth"
    if os.path.exists(detector_path):
        detector.load_state_dict(torch.load(detector_path, map_location=device))
    detector.to(device)
    detector.eval()  # Перевод в режим инференса

    return unet, detector, device


# ==========================================
# 2. Пообъектный анализ маски и нумерация
# ==========================================
def process_instances_with_numbers(binary_mask, gsd):
    """
    Разбивает сплошную маску сегментации на изолированные строения,
    вычисляет площадь каждого здания и наносит номера прямо на изображение.
    """
    # Переводим маску в формат uint8 для работы OpenCV
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)

    # Метод связанных компонент вычисляет статистику и центры масс (centroids) объектов
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask_uint8, connectivity=8
    )

    individual_areas = {}
    colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)

    # Генерируем фиксированную случайную палитру для раскраски зданий
    np.random.seed(10)
    colors = np.random.randint(50, 230, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [15, 15, 20]  # Темно-серый фон вместо агрессивного черного

    # Индекс 0 — это фон, поэтому циклом идем со здания №1
    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        area_sqm = area_px * (gsd ** 2)

        # Защита от мелкого шума сегментации: отсекаем объекты меньше 10 кв.м.
        if area_sqm > 10.0:
            individual_areas[i] = round(area_sqm, 2)

            # Закрашиваем пиксели текущего здания его случайным цветом
            colored_mask[labels == i] = colors[i]

            # Извлекаем координаты центра масс здания для отрисовки текста
            cx, cy = int(centroids[i][0]), int(centroids[i][1])

            # Трюк: Сначала рисуем толстый черный текст (обводка), чтобы номер читался на любом цвете
            cv2.putText(colored_mask, str(i), (cx - 7, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                        cv2.LINE_AA)
            # Поверх него рисуем тонкий белый текст
            cv2.putText(colored_mask, str(i), (cx - 7, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

    return colored_mask, individual_areas


# ==========================================
# 3. Верстка интерфейса Streamlit
# ==========================================
st.set_page_config(page_title="Спутниковый Калькулятор", layout="wide")

st.title("🛰️ Расчет площади застройки по спутниковым снимкам")
st.markdown(
    "Система автоматически сегментирует здания, детектирует автомобили для вычисления масштаба и считает точную площадь объектов в квадратных метрах.")

# --- Боковая панель (Настройки порогов моделей) ---
st.sidebar.header("Настройки моделей")
seg_threshold = st.sidebar.slider("Порог сегментации (Здания)", 0.1, 1.0, 0.5,
                                  help="Чем выше порог, тем жестче модель отбирает контуры зданий.")
det_threshold = st.sidebar.slider("Порог детекции (Машины)", 0.1, 1.0, 0.5,
                                  help="Минимальная уверенность детектора для учета автомобиля.")
real_car_length = st.sidebar.number_input("Длина эталонного авто (метров)", value=4.5, step=0.1,
                                          help="Средняя длина легковой машины для калибровки масштаба.")

# --- Окно загрузки файла ---
uploaded_file = st.file_uploader("Шаг 1: Загрузите снимок местности (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Декодируем изображение в NumPy массив
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    # Подготовка тензора для PyTorch
    unet, detector, device = load_models()
    img_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # ==========================================
    # ИСПРАВЛЕНИЕ: Динамический паддинг для U-Net
    # ==========================================
    _, _, h, w = img_tensor.shape
    # Считаем, сколько пикселей не хватает до кратности 32
    pad_bottom = (32 - h % 32) % 32
    pad_right = (32 - w % 32) % 32

    # Добавляем временные полосы (left, right, top, bottom)
    img_tensor_padded = F.pad(img_tensor, (0, pad_right, 0, pad_bottom))

    with st.spinner("⏳ Идет глубокий анализ изображения нейросетями..."):
        # ---- ШАГ 1: Детекция машин и расчет масштаба ----
        with torch.no_grad():
            # Детектору паддинг не нужен, отдаем оригинал
            det_outputs = detector(img_tensor)[0]

        valid_boxes = []
        for box, score, label in zip(det_outputs['boxes'], det_outputs['scores'], det_outputs['labels']):
            if score > det_threshold and label == 1:
                valid_boxes.append(box.cpu().numpy())

        gsd = calculate_gsd_from_cars(valid_boxes, real_car_length)

        if gsd is None:
            st.sidebar.warning("⚠️ Автомобили не обнаружены. Применен стандартный масштаб по умолчанию: 0.3 м/px.")
            gsd = 0.3

        # ---- ШАГ 2: Сегментация строений ----
        with torch.no_grad():
            # Отправляем в U-Net тензор с кратными сторонами
            seg_output_padded = unet(img_tensor_padded)[0, 0].cpu().numpy()

        # ОБРЕЗАЕМ добавленные полосы, возвращая маске строгий оригинальный размер
        seg_output = seg_output_padded[:h, :w]

        # Формируем бинарную маску по установленному пользователем порогу
        binary_mask = (seg_output > seg_threshold).astype(np.float32)

        # ---- ШАГ 3: Пообъектный обсчет геометрии ----
        colored_mask, individual_areas_dict = process_instances_with_numbers(binary_mask, gsd)
        total_area = sum(individual_areas_dict.values())

    # ==========================================
    # 4. Визуализация результатов в UI
    # ==========================================
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Результаты поиска масштаба")

        # Наносим зеленые рамки детектированных машин поверх копии картинки
        img_with_cars = image_np.copy()
        for box in valid_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_with_cars, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(img_with_cars, caption=f"Детектировано опорных объектов (машин): {len(valid_boxes)}",
                 use_container_width=True)

    with col2:
        st.subheader("🗺️ Пообъектная карта застройки")
        st.image(colored_mask, caption="Номера на зданиях соответствуют списку в таблице статистики.",
                 use_container_width=True)

    # --- Секция числовых метрик и сводной таблицы ---
    st.markdown("### 📊 Сводный аналитический отчет")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Рассчитанный масштаб снимка (GSD)", value=f"{gsd:.3f} м/пиксель")
    with metric_col2:
        st.metric(label="Общая площадь строений", value=f"{total_area:,.2f} м²")
    with metric_col3:
        st.metric(label="Количество идентифицированных зданий", value=f"{len(individual_areas_dict)}")

    if individual_areas_dict:
        st.markdown("#### Детализированный реестр площадей строений")

        # Превращаем данные словаря в датафрейм pandas для интерактивной таблицы
        df = pd.DataFrame({
            "Номер здания на карте": list(individual_areas_dict.keys()),
            "Вычисленная площадь (м²)": list(individual_areas_dict.values())
        })

        # Сортируем строки по убыванию площади зданий для наглядности
        df = df.sort_values(by="Вычисленная площадь (м²)", ascending=False).reset_index(drop=True)

        # Выводим красивую интерактивную таблицу с возможностью поиска и сортировки в Streamlit
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Модель не обнаружила строений на данном снимке при текущем пороге уверенности.")
