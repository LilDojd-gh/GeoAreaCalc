import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

from src import UNetSegmentation, get_vehicle_detection_model, calculate_gsd_from_cars


# ==========================================
# 1. Загрузка моделей с кэшированием
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNetSegmentation(in_channels=3, out_channels=1)
    unet_path = "weights/unet_best.pth"
    if os.path.exists(unet_path):
        unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.to(device)
    unet.eval()

    detector = get_vehicle_detection_model(num_classes=2)
    detector_path = "weights/detector_best.pth"
    if os.path.exists(detector_path):
        detector.load_state_dict(torch.load(detector_path, map_location=device))
    detector.to(device)
    detector.eval()

    return unet, detector, device


# ==========================================
# 2. Пообъектный анализ маски и нумерация
# ==========================================
def process_instances_with_numbers(binary_mask, gsd):
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask_uint8, connectivity=8
    )

    individual_areas = {}
    instance_colors = {}
    colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)

    np.random.seed(10)
    colors = np.random.randint(50, 230, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [15, 15, 20]

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        area_sqm = area_px * (gsd ** 2)

        if area_sqm > 10.0:
            individual_areas[i] = round(area_sqm, 2)

            r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
            instance_colors[i] = f"#{r:02x}{g:02x}{b:02x}"

            colored_mask[labels == i] = colors[i]

            cx, cy = int(centroids[i][0]), int(centroids[i][1])

            cv2.putText(colored_mask, str(i), (cx - 7, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(colored_mask, str(i), (cx - 7, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

    return colored_mask, individual_areas, instance_colors


# ==========================================
# 3. Верстка интерфейса Streamlit
# ==========================================
st.set_page_config(page_title="Спутниковый Калькулятор", layout="wide")

st.title("🛰️ Расчет площади застройки по спутниковым снимкам")

# --- Боковая панель ---
st.sidebar.header("🎛️ Настройки алгоритмов")
auto_seg = st.sidebar.checkbox("Автоматический порог зданий (Оцу)", value=True)
seg_threshold = st.sidebar.slider("Порог сегментации (Здания)", 0.1, 1.0, 0.5, disabled=auto_seg)

st.sidebar.markdown("---")
st.sidebar.header("📏 Настройки масштаба (GSD)")
gsd_mode = st.sidebar.radio(
    "Метод расчета:",
    ["Автоматически (по машинам)", "Ручной ввод GSD", "По параметрам дрона (высота)"]
)

if gsd_mode == "Автоматически (по машинам)":
    det_threshold = st.sidebar.slider("Порог детекции (Машины)", 0.1, 1.0, 0.5)
    real_car_length = st.sidebar.number_input("Длина эталонного авто (метров)", value=4.5, step=0.1)
elif gsd_mode == "Ручной ввод GSD":
    manual_gsd = st.sidebar.number_input("Укажите GSD (м/пиксель)", value=0.300, step=0.010, format="%.3f")
else:
    altitude = st.sidebar.number_input("Высота полета (метры)", value=100.0, step=5.0)
    focal_length = st.sidebar.number_input("Фокусное расстояние (мм)", value=8.8, step=0.1)
    sensor_width = st.sidebar.number_input("Ширина матрицы (мм)", value=13.2, step=0.1)
    st.sidebar.caption("По умолчанию указаны параметры DJI Phantom 4 Pro.")

# --- Окно загрузки файла ---
uploaded_file = st.file_uploader("Шаг 1: Загрузите снимок местности (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    unet, detector, device = load_models()
    img_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Динамический паддинг для U-Net (кратность 32)
    _, _, h, w = img_tensor.shape
    pad_bottom = (32 - h % 32) % 32
    pad_right = (32 - w % 32) % 32
    img_tensor_padded = F.pad(img_tensor, (0, pad_right, 0, pad_bottom))

    valid_boxes = []

    with st.spinner("⏳ Идет глубокий анализ изображения нейросетями..."):

        # ---- ШАГ 1: Расчет масштаба (GSD) ----
        if gsd_mode == "Автоматически (по машинам)":
            with torch.no_grad():
                det_outputs = detector(img_tensor)[0]

            for box, score, label in zip(det_outputs['boxes'], det_outputs['scores'], det_outputs['labels']):
                if score > det_threshold and label == 1:
                    valid_boxes.append(box.cpu().numpy())

            gsd = calculate_gsd_from_cars(valid_boxes, real_car_length)

            if gsd is None:
                st.sidebar.warning("⚠️ Автомобили не обнаружены. Применен стандартный масштаб по умолчанию: 0.3 м/px.")
                gsd = 0.3

        elif gsd_mode == "Ручной ввод GSD":
            gsd = manual_gsd

        else:  # По параметрам дрона (высота)
            image_width_px = image_np.shape[1]
            gsd = (altitude * sensor_width) / (focal_length * image_width_px)

        # ---- ШАГ 2: Сегментация строений ----
        with torch.no_grad():
            seg_output_padded = torch.sigmoid(unet(img_tensor_padded))[0, 0].cpu().numpy()

        seg_output = seg_output_padded[:h, :w]

        if auto_seg:
            seg_uint8 = (seg_output * 255).astype(np.uint8)
            optimal_val, binary_mask_uint8 = cv2.threshold(seg_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_mask = binary_mask_uint8.astype(np.float32)
            st.sidebar.success(f"🤖 Применен порог Оцу: {optimal_val / 255.0:.2f}")
        else:
            binary_mask = (seg_output > seg_threshold).astype(np.float32)

        # ---- ШАГ 3: Пообъектный обсчет геометрии ----
        colored_mask, individual_areas_dict, instance_colors_dict = process_instances_with_numbers(binary_mask, gsd)
        total_area = sum(individual_areas_dict.values())

    # ==========================================
    # 4. Визуализация результатов в UI
    # ==========================================
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Оригинал и детекция масштаба")
        img_with_cars = image_np.copy()

        if gsd_mode == "Автоматически (по машинам)":
            for box in valid_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_with_cars, (x1, y1), (x2, y2), (0, 255, 0), 2)
            st.image(img_with_cars, caption=f"Детектировано машин: {len(valid_boxes)}", use_container_width=True)
        else:
            st.image(img_with_cars, caption=f"Режим: {gsd_mode}", use_container_width=True)

    with col2:
        st.subheader("🗺️ Пообъектная карта застройки")
        st.image(colored_mask, caption="Номера на зданиях соответствуют списку в таблице статистики.",
                 use_container_width=True)

    st.markdown("### 📊 Сводный аналитический отчет")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Рассчитанный масштаб снимка (GSD)", value=f"{gsd:.3f} м/пиксель")
    with metric_col2:
        st.metric(label="Общая площадь строений", value=f"{total_area:,.2f} м²")
    with metric_col3:
        st.metric(label="Количество идентифицированных зданий", value=f"{len(individual_areas_dict)}")

    if individual_areas_dict:
        st.markdown("#### 🏢 Детализированный реестр площадей строений")

        df = pd.DataFrame({
            "Номер здания на карте": list(individual_areas_dict.keys()),
            "Метка цвета": list(instance_colors_dict.values()),
            "Вычисленная площадь (м²)": list(individual_areas_dict.values())
        })

        df = df.sort_values(by="Вычисленная площадь (м²)", ascending=False).reset_index(drop=True)


        def color_square(val):
            return f'background-color: {val}; color: {val}'


        try:
            styled_df = df.style.map(color_square, subset=["Метка цвета"])
        except AttributeError:
            styled_df = df.style.applymap(color_square, subset=["Метка цвета"])

        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("Модель не обнаружила строений на данном снимке при текущем пороге уверенности.")
