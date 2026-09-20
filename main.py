"""API de validación de producto por secuencia de detecciones.

Flujo: la app pide las secuencias disponibles, abre una sesión sobre una de
ellas y envía una foto por paso. El backend decide si el paso se aprueba y,
cuando lo hace, devuelve el siguiente paso.

Ejecución en LAN:
    uvicorn main:app --host 0.0.0.0 --port 8000

Sin GPU ni pesos entrenados:
    DETECTOR_MODE=mock uvicorn main:app --host 0.0.0.0 --port 8000
"""

import sys
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import Config, ConfigError, Step as StepConfig, load_config
from app.detector import Detection, Detector, annotate, build_detector
from app.models import (
	CreateSessionRequest,
	ExpectedObject,
	HealthResponse,
	ObjectResult,
	SequenceRef,
	SequenceSummary,
	SessionState,
	Step,
	StepResult,
)
from app.sessions import Session, SessionStore

# TODO: Add login

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
	try:
		# Primera pasada sin el modelo: necesitamos el catálogo de clases para
		# poder construir el detector mock.
		config = load_config()
		detector = build_detector({c for c in config.classes})
		# Segunda pasada con las clases reales del modelo, para cazar cualquier
		# clase configurada que el modelo no conozca.
		config = load_config(model_classes=detector.class_names)
	except ConfigError as exc:
		print(f"\n[config] {exc}\n", file=sys.stderr)
		raise SystemExit(1)

	state["config"] = config
	state["detector"] = detector
	state["sessions"] = SessionStore()
	print(f"[api] {len(config.sequences)} secuencia(s) cargada(s)")
	yield
	state.clear()


app = FastAPI(title="Validación por secuencia de detecciones", lifespan=lifespan)

# LAN cerrada y la app corre bajo androidScheme 'http': sin restricción de origen.
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)


def get_config() -> Config:
	return state["config"]  # type: ignore[return-value]


def get_detector() -> Detector:
	return state["detector"]  # type: ignore[return-value]


def get_sessions() -> SessionStore:
	return state["sessions"]  # type: ignore[return-value]


# --- Endpoints -------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health(config: Config = Depends(get_config), detector: Detector = Depends(get_detector)):
	return HealthResponse(
		status="ok",
		detector=detector.mode,
		model_classes=sorted(detector.class_names),
		sequences=len(config.sequences),
	)


@app.get("/sequences", response_model=list[SequenceSummary])
def list_sequences(config: Config = Depends(get_config)):
	return [
		SequenceSummary(
			id=sequence.id,
			name=sequence.name,
			description=sequence.description,
			total_steps=len(sequence.steps),
		)
		for sequence in config.sequences
	]


@app.post("/sessions", response_model=SessionState, status_code=201)
def create_session(
	request: CreateSessionRequest,
	config: Config = Depends(get_config),
	sessions: SessionStore = Depends(get_sessions),
):
	sequence = config.sequence(request.sequence_id)
	if sequence is None:
		raise HTTPException(
			status_code=404,
			detail={
				"code": "sequence_not_found",
				"message": f"No existe la secuencia '{request.sequence_id}'",
			},
		)

	session = sessions.create(sequence)
	return _session_state(session)


@app.post("/sessions/{session_id}/steps", response_model=StepResult)
async def submit_step(
	session_id: str,
	file: UploadFile = File(...),
	detector: Detector = Depends(get_detector),
	config: Config = Depends(get_config),
	sessions: SessionStore = Depends(get_sessions),
):
	session = sessions.get(session_id)
	if session is None:
		raise HTTPException(status_code=404, detail=_session_not_found())

	if session.is_complete:
		raise HTTPException(
			status_code=409,
			detail={
				"code": "sequence_already_completed",
				"message": "Esta validación ya terminó. Inicia una nueva.",
			},
		)

	image = _decode_image(await file.read())
	step = session.current_step
	step_number = session.step_index + 1
	total_steps = len(session.sequence.steps)

	detections = [d for d in detector.detect(image) if d.confidence >= step.min_confidence]
	# La imagen se anota con todo lo que el modelo vio por encima del umbral,
	# incluidos objetos que este paso no pedía: ayuda a entender un encuadre malo.
	annotated = annotate(image, detections, config.classes)
	results = _evaluate(step, detections)

	if all(result.ok for result in results):
		session.step_index += 1
		if session.is_complete:
			sessions.discard(session.id)
			return StepResult(
				status="completed",
				message="Producto validado",
				step_number=step_number,
				total_steps=total_steps,
				next_step=None,
				results=results,
				annotated_image=annotated,
			)

		return StepResult(
			status="passed",
			message="Paso aprobado",
			step_number=step_number,
			total_steps=total_steps,
			next_step=_step_model(session.current_step, session.step_index),
			results=results,
			annotated_image=annotated,
		)

	return StepResult(
		status="failed",
		message=_failure_message([r for r in results if not r.ok]),
		step_number=step_number,
		total_steps=total_steps,
		next_step=None,
		results=results,
		annotated_image=annotated,
	)


@app.delete("/sessions/{session_id}", status_code=204)
def abort_session(session_id: str, sessions: SessionStore = Depends(get_sessions)):
	# Abortar una sesión ya desaparecida es el resultado que el cliente quería.
	sessions.discard(session_id)
	return None


# --- Lógica de evaluación --------------------------------------------------


def _evaluate(step: StepConfig, detections: list[Detection]) -> list[ObjectResult]:
	"""Compara lo detectado contra lo que el paso exige.

	Se devuelven todos los objetos esperados, aprobados y faltantes, para que la
	app pinte la misma lista de siempre con el veredicto de cada uno.
	"""
	results: list[ObjectResult] = []

	for expectation in step.expect:
		matches = [d for d in detections if d.class_name == expectation.class_name]
		found = len(matches)
		best = max((d.confidence for d in matches), default=None)

		results.append(
			ObjectResult(
				class_name=expectation.class_name,
				label=expectation.label,
				expected=expectation.min_count,
				found=found,
				ok=found >= expectation.min_count,
				best_confidence=round(best, 3) if best is not None else None,
			)
		)

	return results


def _failure_message(missing: list[ObjectResult]) -> str:
	if len(missing) == 1:
		item = missing[0]
		if item.expected > 1:
			return (
				f"Se esperaban {item.expected} de «{item.label}» y se detectaron {item.found}. "
				f"Ajusta el encuadre y vuelve a tomar la foto."
			)
		return f"No se detectó «{item.label}». Ajusta el encuadre y vuelve a tomar la foto."

	labels = ", ".join(f"«{item.label}»" for item in missing)
	return f"No se detectaron: {labels}. Ajusta el encuadre y vuelve a tomar la foto."


def _decode_image(payload: bytes) -> np.ndarray:
	image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
	if image is None:
		raise HTTPException(
			status_code=400,
			detail={
				"code": "invalid_image",
				"message": "No se pudo leer la imagen. Vuelve a tomar la foto.",
			},
		)
	return image


def _session_not_found() -> dict:
	return {
		"code": "session_not_found",
		"message": "La sesión expiró. Vuelve a empezar la validación.",
	}


def _step_model(step: StepConfig, index: int) -> Step:
	return Step(
		index=index,
		number=index + 1,
		instruction=step.instruction,
		expect=[
			ExpectedObject(
				class_name=expectation.class_name,
				label=expectation.label,
				min_count=expectation.min_count,
			)
			for expectation in step.expect
		],
	)


def _session_state(session: Session) -> SessionState:
	return SessionState(
		session_id=session.id,
		sequence=SequenceRef(id=session.sequence.id, name=session.sequence.name),
		total_steps=len(session.sequence.steps),
		step=_step_model(session.current_step, session.step_index),
	)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
