# Превращаем папку src в пакет
from .dataset import SatelliteSegmentationDataset, VehicleDetectionDataset
from .models import UNetSegmentation, get_vehicle_detection_model
from .losses import BCEDiceLoss
from .utils import calculate_metrics, calculate_gsd_from_cars