"""Envoltorio del modelo YOLO, con un modo mock para desarrollar sin GPU ni pesos.

Ambos modos implementan la misma interfaz y viven detrás de los mismos endpoints,
así que el contrato que ve la app es idéntico en mock y en real. Esa era la
debilidad del `api_mock.py` anterior: era un segundo servidor que se desincronizó
del real.
"""

import base64
import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

# Ancho máximo de la imagen anotada que se devuelve a la app. Acota el base64
# sin perder legibilidad en un teléfono.
ANNOTATED_MAX_WIDTH = 720
ANNOTATED_JPEG_QUALITY = 80


@dataclass(frozen=True)
class Detection:
	class_name: str
	confidence: float
	box: tuple[int, int, int, int]  # x1, y1, x2, y2


class Detector(Protocol):
	mode: str
	class_names: set[str]

	def detect(self, image: np.ndarray) -> list[Detection]: ...


class YoloDetector:
	"""Detector real. Carga los pesos una sola vez, al arrancar el proceso."""

	mode = "yolo"

	def __init__(self, model_path: str, device: str | None = None):
		# Import perezoso: en modo mock no queremos pagar la carga de torch.
		import torch
		from ultralytics import YOLO

		resolved = device or _autodetect_device(torch)
		if resolved == "cuda" and not torch.cuda.is_available():
			print("[detector] CUDA no disponible, se usará CPU")
			resolved = "cpu"
		elif resolved == "mps" and not torch.backends.mps.is_available():
			print("[detector] MPS no disponible, se usará CPU")
			resolved = "cpu"

		self.device = resolved
		self.model = YOLO(model_path).to(resolved)
		self.class_names = {str(name) for name in self.model.names.values()}
		print(f"[detector] YOLO '{model_path}' cargado en {resolved}")

	def detect(self, image: np.ndarray) -> list[Detection]:
		results = self.model(image, verbose=False)
		names = self.model.names

		detections: list[Detection] = []
		for box in results[0].boxes:
			x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
			detections.append(
				Detection(
					class_name=str(names[int(box.cls[0])]),
					confidence=float(box.conf[0]),
					box=(x1, y1, x2, y2),
				)
			)
		return detections


class MockDetector:
	"""Detector sintético para desarrollar la app sin modelo entrenado.

	Devuelve las clases que la configuración conoce, alternando entre una
	detección de alta y de baja confianza en llamadas sucesivas, de modo que el
	flujo de la app pase por el camino de éxito y por el de fallo sin trucos en
	el cliente.
	"""

	mode = "mock"

	def __init__(self, class_names: set[str]):
		self.class_names = class_names
		self._calls = 0

	def detect(self, image: np.ndarray) -> list[Detection]:
		self._calls += 1
		# Una de cada tres capturas se degrada, para poder ver la pantalla de fallo.
		confidence = 0.22 if self._calls % 3 == 0 else 0.93

		height, width = image.shape[:2]
		ordered = sorted(self.class_names)
		detections: list[Detection] = []

		for position, class_name in enumerate(ordered):
			# Cajas en diagonal, solo para que la imagen anotada sea legible.
			offset = 24 + position * 18
			x1 = min(offset, max(width - 60, 0))
			y1 = min(offset, max(height - 60, 0))
			detections.append(
				Detection(
					class_name=class_name,
					confidence=confidence,
					box=(x1, y1, min(x1 + width // 3, width), min(y1 + height // 3, height)),
				)
			)
		return detections


def _autodetect_device(torch) -> str:
	"""Elige el acelerador disponible.

	`mps` es la GPU integrada de los Mac con Apple Silicon: sin esta rama, un
	Mac caería a CPU y la inferencia sería varias veces más lenta.
	"""
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def build_detector(class_names: set[str]) -> Detector:
	"""Crea el detector según las variables de entorno.

	DETECTOR_MODE=mock  -> detector sintético, sin torch ni pesos
	MODEL_PATH          -> ruta de los pesos (default: yolov8n.pt)
	DEVICE              -> cuda | mps | cpu (default: el mejor disponible)
	"""
	if os.getenv("DETECTOR_MODE", "").lower() == "mock":
		print("[detector] modo mock: no se cargará el modelo YOLO")
		return MockDetector(class_names)

	return YoloDetector(
		model_path=os.getenv("MODEL_PATH", "yolov8n.pt"),
		device=os.getenv("DEVICE") or None,
	)


def annotate(image: np.ndarray, detections: list[Detection], labels: dict[str, str]) -> str:
	"""Dibuja las detecciones y devuelve un data-URI JPEG listo para un <img>.

	Las cajas se rotulan con el nombre legible, no con la clase del modelo: es lo
	que el operador está leyendo en el resto de la pantalla.
	"""
	canvas = image.copy()

	for detection in detections:
		x1, y1, x2, y2 = detection.box
		label = labels.get(detection.class_name, detection.class_name)
		caption = f"{label} {detection.confidence:.0%}"

		cv2.rectangle(canvas, (x1, y1), (x2, y2), (216, 95, 14), 2)

		(text_width, text_height), baseline = cv2.getTextSize(
			caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
		)
		# La etiqueta va sobre la caja, salvo que no quepa arriba.
		text_top = y1 - text_height - baseline - 4
		if text_top < 0:
			text_top = y1 + 4

		cv2.rectangle(
			canvas,
			(x1, text_top),
			(x1 + text_width + 8, text_top + text_height + baseline + 4),
			(216, 95, 14),
			cv2.FILLED,
		)
		cv2.putText(
			canvas,
			caption,
			(x1 + 4, text_top + text_height + 2),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)

	height, width = canvas.shape[:2]
	if width > ANNOTATED_MAX_WIDTH:
		scale = ANNOTATED_MAX_WIDTH / width
		canvas = cv2.resize(
			canvas, (ANNOTATED_MAX_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA
		)

	ok, buffer = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, ANNOTATED_JPEG_QUALITY])
	if not ok:
		raise RuntimeError("No se pudo codificar la imagen anotada")

	return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
