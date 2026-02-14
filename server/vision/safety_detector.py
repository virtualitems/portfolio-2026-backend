import base64
import cv2
import numpy as np
from ultralytics import YOLO
from ..shared.logger import get_logger
from ..shared.env import env

logger = get_logger(__name__)

class VisionSafetyDetector:
    """Detector de equipos de seguridad usando YOLO"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat',
            'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone',
            'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
            'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer',
            'truck', 'van', 'vehicle', 'wheel loader'
        ]

    def load_model(self):
        """Carga el modelo YOLO si no está cargado"""
        if self.model is None:
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load YOLO model from {self.model_path}: {e}")
                raise

    def _get_detection_color(self, class_name: str) -> tuple:
        """Determina el color del bounding box basado en la clase detectada"""
        if class_name in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
            return (0, 0, 255)
        elif class_name in ['Hardhat', 'Safety Vest', 'Mask', 'Gloves']:
            return (0, 255, 0)
        else:
            return (255, 0, 0)

    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Decodifica una imagen en formato base64 a formato OpenCV"""
        try:
            if base64_string.startswith('data:image'):
                base64_data = base64_string.split(',')[1]
            else:
                base64_data = base64_string

            img_bytes = base64.b64decode(base64_data.encode('utf-8'))

            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return img
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            return None

    def process_frame(self, img: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
        """Procesa un frame con el modelo YOLO y dibuja las detecciones"""
        if img is None:
            logger.warning("Attempted to process None image frame")
            return None

        if self.model is None:
            logger.error("YOLO model not loaded when trying to process frame")
            return img

        try:
            results = self.model(img, stream=True)

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.class_names[cls]

                    if conf <= confidence_threshold:
                        continue

                    color = self._get_detection_color(class_name)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(img, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return img
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return img

    def encode_image_to_base64(self, img: np.ndarray) -> str:
        """Codifica una imagen OpenCV a base64 con formato para enviar al cliente"""
        try:
            _, buffer = cv2.imencode('.jpg', img)
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            return f'data:image/jpeg;base64,{processed_base64}'
        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            return None

    def process_base64_frame(self, base64_string: str, confidence_threshold: float = 0.5) -> str:
        """Proceso completo: decodifica, procesa y codifica una imagen en base64"""
        img = self.decode_base64_image(base64_string)
        if img is None:
            return None

        processed_img = self.process_frame(img, confidence_threshold)
        if processed_img is None:
            return None

        return self.encode_image_to_base64(processed_img)


safety_detector = VisionSafetyDetector(env['YOLO_SAFETY_DETECTOR_MODEL_PATH'])
