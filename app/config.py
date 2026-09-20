"""Carga y validación de config/sequences.json.

El archivo lo edita una persona a mano, así que la estrategia es fallar al
arrancar con un mensaje que nombre secuencia, paso y clase. Un typo en una clase
que solo se notara a mitad de una validación en piso sería mucho peor.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "sequences.json"
FALLBACK_MIN_CONFIDENCE = 0.45


def config_path() -> Path:
	"""Archivo de configuración a usar.

	`CONFIG_PATH` permite arrancar contra otro archivo sin tocar el de
	producción, por ejemplo para probar el modelo preentrenado con las clases
	de COCO.
	"""
	override = os.getenv("CONFIG_PATH")
	return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


class ConfigError(Exception):
	"""Configuración inválida. El mensaje ya viene listo para leerse en consola."""


@dataclass(frozen=True)
class Expectation:
	class_name: str
	label: str
	min_count: int


@dataclass(frozen=True)
class Step:
	instruction: str
	expect: tuple[Expectation, ...]
	min_confidence: float


@dataclass(frozen=True)
class Sequence:
	id: str
	name: str
	description: str
	steps: tuple[Step, ...]


@dataclass(frozen=True)
class Config:
	classes: dict[str, str]
	sequences: tuple[Sequence, ...]

	def sequence(self, sequence_id: str) -> Sequence | None:
		return next((s for s in self.sequences if s.id == sequence_id), None)


def load_config(path: Path | None = None, model_classes: set[str] | None = None) -> Config:
	"""Lee el JSON, lo valida y lo convierte en objetos inmutables.

	`model_classes` son los nombres que realmente conoce el modelo YOLO cargado.
	Cuando se pasa, se verifica que cada clase de la config exista en el modelo.
	"""
	path = path or config_path()
	where = path.name

	try:
		raw = json.loads(path.read_text(encoding="utf-8"))
	except FileNotFoundError:
		raise ConfigError(f"No se encontró el archivo de configuración: {path}")
	except json.JSONDecodeError as exc:
		raise ConfigError(f"{where}: JSON inválido en la línea {exc.lineno}: {exc.msg}")

	if not isinstance(raw, dict):
		raise ConfigError(f"{where}: la raíz del archivo debe ser un objeto JSON")

	classes = _parse_classes(raw.get("classes"), where)
	default_confidence = _parse_default_confidence(raw.get("defaults") or {}, where)

	raw_sequences = raw.get("sequences")
	if not isinstance(raw_sequences, list) or not raw_sequences:
		raise ConfigError(f"{where}: 'sequences' debe ser una lista con al menos una secuencia")

	sequences: list[Sequence] = []
	seen_ids: set[str] = set()

	for position, raw_sequence in enumerate(raw_sequences, start=1):
		sequence = _parse_sequence(
			raw_sequence,
			position=position,
			where=where,
			classes=classes,
			model_classes=model_classes,
			default_confidence=default_confidence,
		)
		if sequence.id in seen_ids:
			raise ConfigError(f"{where}: el id de secuencia '{sequence.id}' está repetido")
		seen_ids.add(sequence.id)
		sequences.append(sequence)

	return Config(classes=classes, sequences=tuple(sequences))


def _parse_classes(raw_classes: object, where: str) -> dict[str, str]:
	if not isinstance(raw_classes, dict) or not raw_classes:
		raise ConfigError(
			f"{where}: 'classes' debe ser un objeto que mapee cada clase del modelo "
			f"a su nombre legible, por ejemplo {{\"cable\": \"Cable de alimentación\"}}"
		)

	for class_name, label in raw_classes.items():
		if not isinstance(label, str) or not label.strip():
			raise ConfigError(f"{where}: la clase '{class_name}' necesita un nombre legible no vacío")

	return {str(k): str(v) for k, v in raw_classes.items()}


def _parse_default_confidence(raw_defaults: object, where: str) -> float:
	if not isinstance(raw_defaults, dict):
		raise ConfigError(f"{where}: 'defaults' debe ser un objeto")
	return _parse_confidence(
		raw_defaults.get("min_confidence", FALLBACK_MIN_CONFIDENCE),
		context=f"{where}: defaults",
	)


def _parse_confidence(value: object, context: str) -> float:
	if not isinstance(value, (int, float)) or isinstance(value, bool):
		raise ConfigError(f"{context}: 'min_confidence' debe ser un número entre 0 y 1")
	if not 0 < float(value) <= 1:
		raise ConfigError(f"{context}: 'min_confidence' debe estar entre 0 y 1, se recibió {value}")
	return float(value)


def _parse_sequence(
	raw_sequence: object,
	position: int,
	where: str,
	classes: dict[str, str],
	model_classes: set[str] | None,
	default_confidence: float,
) -> Sequence:
	if not isinstance(raw_sequence, dict):
		raise ConfigError(f"{where}: la secuencia #{position} debe ser un objeto")

	sequence_id = raw_sequence.get("id")
	if not isinstance(sequence_id, str) or not sequence_id.strip():
		raise ConfigError(f"{where}: la secuencia #{position} necesita un 'id' de texto no vacío")

	label = f"{where}: secuencia '{sequence_id}'"

	name = raw_sequence.get("name")
	if not isinstance(name, str) or not name.strip():
		raise ConfigError(f"{label}: necesita un 'name' de texto no vacío")

	raw_steps = raw_sequence.get("steps")
	if not isinstance(raw_steps, list) or not raw_steps:
		raise ConfigError(f"{label}: 'steps' debe ser una lista con al menos un paso")

	steps = tuple(
		_parse_step(
			raw_step,
			number=number,
			label=label,
			classes=classes,
			model_classes=model_classes,
			default_confidence=default_confidence,
		)
		for number, raw_step in enumerate(raw_steps, start=1)
	)

	return Sequence(
		id=sequence_id,
		name=name,
		description=str(raw_sequence.get("description") or ""),
		steps=steps,
	)


def _parse_step(
	raw_step: object,
	number: int,
	label: str,
	classes: dict[str, str],
	model_classes: set[str] | None,
	default_confidence: float,
) -> Step:
	context = f"{label}, paso {number}"

	if not isinstance(raw_step, dict):
		raise ConfigError(f"{context}: debe ser un objeto")

	instruction = raw_step.get("instruction")
	if not isinstance(instruction, str) or not instruction.strip():
		raise ConfigError(
			f"{context}: necesita una 'instruction' no vacía; es el texto que lee el operador"
		)

	raw_expect = raw_step.get("expect")
	if not isinstance(raw_expect, list) or not raw_expect:
		raise ConfigError(f"{context}: 'expect' debe listar al menos un objeto a detectar")

	expectations: list[Expectation] = []
	for raw_expectation in raw_expect:
		if not isinstance(raw_expectation, dict):
			raise ConfigError(f"{context}: cada entrada de 'expect' debe ser un objeto")

		class_name = raw_expectation.get("class")
		if not isinstance(class_name, str) or not class_name.strip():
			raise ConfigError(f"{context}: cada entrada de 'expect' necesita una 'class'")

		if class_name not in classes:
			raise ConfigError(
				f"{context}: la clase '{class_name}' no está en 'classes'. "
				f"Agrégala con su nombre legible para poder informar al operador."
			)

		if model_classes is not None and class_name not in model_classes:
			raise ConfigError(
				f"{context}: la clase '{class_name}' no existe en el modelo cargado. "
				f"Clases disponibles: {', '.join(sorted(model_classes))}"
			)

		min_count = raw_expectation.get("min_count", 1)
		if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
			raise ConfigError(f"{context}: 'min_count' de '{class_name}' debe ser un entero >= 1")

		expectations.append(
			Expectation(class_name=class_name, label=classes[class_name], min_count=min_count)
		)

	min_confidence = (
		_parse_confidence(raw_step["min_confidence"], context)
		if "min_confidence" in raw_step
		else default_confidence
	)

	return Step(
		instruction=instruction,
		expect=tuple(expectations),
		min_confidence=min_confidence,
	)
