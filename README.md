# API de validación por secuencia de detecciones

Valida un producto paso a paso: la app envía una foto por paso y esta API decide,
con YOLO, si se ven los objetos que ese paso exige. Si falta algo, responde qué
objeto no apareció usando el nombre legible configurado; si todo está, devuelve
el siguiente paso.

El backend es la autoridad del flujo: la app solo renderiza el paso que recibe.

## Dependencias

```
pip install fastapi uvicorn python-multipart opencv-python numpy ultralytics torch torchvision
```

En modo mock (ver abajo) bastan `fastapi uvicorn python-multipart opencv-python numpy`:
el detector real y `torch` se importan solo cuando se usan.

## Ejecución

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` no es opcional para probar desde el teléfono: con el default
(`127.0.0.1`) la API solo escucha en la propia máquina y la app falla con
"connection refused". El teléfono y el servidor deben estar en la misma WiFi, y
la IP de la máquina es la que va en `environment.ts` de la app.

### Sin GPU ni pesos entrenados

```
DETECTOR_MODE=mock uvicorn main:app --host 0.0.0.0 --port 8000
```

El detector mock devuelve detecciones sintéticas y degrada una de cada tres
capturas, para poder recorrer el camino de éxito y el de fallo desde la app. Usa
los mismos endpoints y las mismas respuestas que el modo real, así que el
contrato no puede divergir.

### Variables de entorno

| Variable | Default | Para qué |
|---|---|---|
| `DETECTOR_MODE` | *(vacío)* | `mock` para no cargar el modelo |
| `MODEL_PATH` | `yolov8n.pt` | Ruta de los pesos |
| `DEVICE` | auto | `cuda`, `mps` (GPU de Apple Silicon) o `cpu`; si se pide un acelerador que no existe, cae a cpu |
| `CONFIG_PATH` | `config/sequences.json` | Otro archivo de configuración, para probar sin tocar el de producción |

### Pasar del mock al modelo real

```
.venv/bin/pip install ultralytics torch torchvision
MODEL_PATH=ruta/a/tus/pesos.pt uvicorn main:app --host 0.0.0.0 --port 8000
```

Basta con omitir `DETECTOR_MODE`. Las clases de `config/sequences.json` deben
existir en los pesos que cargues; si no, el proceso no arranca y te dice cuál
falta y cuáles hay disponibles.

Para comprobar que torch y la GPU funcionan **antes** de tener pesos propios,
`config/sequences.coco.json` trae una secuencia de humo con clases del modelo
preentrenado (teclado, ratón, teléfono):

```
CONFIG_PATH=config/sequences.coco.json uvicorn main:app --host 0.0.0.0 --port 8000
```

La primera ejecución descarga `yolov8n.pt` (~6 MB) al directorio de trabajo.

## Configuración: `config/sequences.json`

Es el único archivo que se edita a mano para definir qué se valida.

```json
{
  "defaults": { "min_confidence": 0.45 },
  "classes": {
    "cable": "Cable de alimentación"
  },
  "sequences": [
    {
      "id": "empaque-teclado",
      "name": "Empaque de teclado",
      "description": "Validación del contenido de la caja antes de sellarla.",
      "steps": [
        {
          "instruction": "Acomoda el cable en su compartimento y toma la foto.",
          "expect": [{ "class": "cable", "min_count": 1 }],
          "min_confidence": 0.6
        }
      ]
    }
  ]
}
```

- `classes` mapea cada clase del modelo YOLO a su nombre legible. Es el único
  lugar donde se traduce: cambiar cómo se llama un objeto de cara al operador es
  una sola edición.
- `instruction` es literalmente el texto que lee el operador antes de disparar.
- `min_count` exige varias piezas del mismo objeto (default 1).
- `min_confidence` se define por paso, con fallback a `defaults`.

La configuración se valida al arrancar. Si una clase no existe en `classes` o no
la conoce el modelo cargado, el proceso no levanta y escribe qué secuencia, qué
paso y qué clase están mal.

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| `GET` | `/health` | Estado, modo del detector y clases del modelo |
| `GET` | `/sequences` | Secuencias disponibles para elegir en la app |
| `POST` | `/sessions` | Abre una validación: `{ "sequence_id": "..." }` → paso 1 |
| `POST` | `/sessions/{id}/steps` | Envía la foto del paso actual (multipart `file`) |
| `DELETE` | `/sessions/{id}` | Aborta la validación en curso |

`POST /sessions/{id}/steps` responde con `status`:

- `passed` — paso aprobado, `next_step` trae el siguiente.
- `failed` — se repite el paso; `results` dice qué objeto faltó y con qué
  confianza se vio (si se vio).
- `completed` — producto validado, la sesión se cierra.

Siempre incluye `annotated_image`, un data-URI JPEG con las cajas dibujadas y
rotuladas con el nombre legible, para que el operador vea qué reconoció el
modelo.

Las sesiones viven en memoria con TTL de 2 horas. Si el proceso reinicia, la app
recibe `404` con `{"detail": {"code": "session_not_found"}}` y reinicia el flujo.

## Pendientes

- Login (`TODO` en `main.py`).
- Persistencia de las validaciones: hoy no queda registro de los productos
  validados, todo vive en memoria.
