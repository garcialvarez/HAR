# Datos

El proyecto usa el **UCI HAR Dataset** (*Human Activity Recognition Using Smartphones*,
Anguita et al., 2013). El dataset **no se incluye en el repositorio**; se descarga solo.

## Descarga automática (recomendada)

Todos los notebooks ejecutan en su celda de arranque:

```python
from data_loader import download_dataset
download_dataset()
```

Esto descarga el `.zip` (~61 MB) desde UCI a `data/raw/UCI_HAR_Dataset.zip` y lo descomprime en
`data/raw/UCI_HAR_Dataset/`. Si el archivo ya existe, no se vuelve a descargar. Funciona igual en
Google Colab y en local.

URL de origen:
<https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip>

## Descarga manual (si la URL de UCI no responde)

1. Descargar el `.zip` desde <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>.
2. Guardarlo como `data/raw/UCI_HAR_Dataset.zip`.
3. Ejecutar cualquier notebook: `data_loader.ensure_extracted()` lo descomprime.

## Contenido relevante

- `train/` y `test/`: partición oficial por sujeto (21 y 9 sujetos).
- `X_*.txt`: 561 características; `y_*.txt`: actividad (1–6); `subject_*.txt`: sujeto (1–30).
- `Inertial Signals/`: 9 canales crudos de 128 muestras por ventana (50 Hz).
