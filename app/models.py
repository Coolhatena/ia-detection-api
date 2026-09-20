"""Esquemas del contrato HTTP entre la API y la app.

Los tipos de `app/config.py` describen el archivo de configuración; los de aquí
describen lo que viaja por la red. Se mantienen separados a propósito: la config
puede crecer (umbrales, metadatos internos) sin cambiar lo que ve la app.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExpectedObject(BaseModel):
	"""Un objeto que el paso exige ver en la foto, ya traducido a lenguaje humano."""

	# `class` es palabra reservada en Python, así que el campo se llama distinto
	# y se serializa con el nombre del contrato.
	class_name: str = Field(serialization_alias="class")
	label: str
	min_count: int


class Step(BaseModel):
	index: int
	number: int  # index + 1, para no hacer la aritmética en la vista
	instruction: str
	expect: list[ExpectedObject]


class SequenceSummary(BaseModel):
	id: str
	name: str
	description: str
	total_steps: int


class SequenceRef(BaseModel):
	id: str
	name: str


class SessionState(BaseModel):
	session_id: str
	sequence: SequenceRef
	total_steps: int
	step: Step


class CreateSessionRequest(BaseModel):
	sequence_id: str


class ObjectResult(BaseModel):
	"""Veredicto de un objeto esperado. Se devuelven todos, no solo los faltantes."""

	class_name: str = Field(serialization_alias="class")
	label: str
	expected: int
	found: int
	ok: bool
	best_confidence: Optional[float] = None


class StepResult(BaseModel):
	status: Literal["passed", "failed", "completed"]
	message: str
	step_number: int
	total_steps: int
	next_step: Optional[Step] = None
	results: list[ObjectResult]
	annotated_image: Optional[str] = None


class HealthResponse(BaseModel):
	# `model_classes` choca con el namespace protegido `model_` de Pydantic v2.
	model_config = ConfigDict(protected_namespaces=())

	status: str
	detector: Literal["yolo", "mock"]
	model_classes: list[str]
	sequences: int
