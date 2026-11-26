# Reconocimiento de Actividades Humanas (HAR) a partir de sensores inerciales de smartphones

Proyecto final — curso *Fundamentos de Deep Learning*, Ingeniería de Sistemas, Universidad de Antioquia.

**Integrantes:** Juan José García Álvarez · Valentina Buelvas Martínez · Tomás Cadavid Martínez

| Entregable | Enlace |
|---|---|
| Informe ejecutivo (Entrega 2) | [`INFORME_PROYECTO.PDF`](INFORME_PROYECTO.PDF) |
| Propuesta (Entrega 1) | [`ENTREGA1.PDF`](ENTREGA1.PDF) |
| **Video de presentación (YouTube)** | _Pendiente: el enlace se agrega aquí al publicar el video._ |

## 1. El problema

Clasificar la actividad que realiza una persona (**caminar, subir escaleras, bajar escaleras, sentado, de pie, acostado**) a partir del acelerómetro y el giroscopio de un smartphone en la cintura, en ventanas de 2.56 s (128 muestras a 50 Hz). Dataset: [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) — 30 voluntarios, 10,299 ventanas, partición oficial por sujeto (21 para entrenar, 9 para prueba).

La retroalimentación de la Entrega 1 pidió tres cosas, y cada una tiene su lugar en el repositorio:

| Retroalimentación | Cómo se atendió | Dónde |
|---|---|---|
| Comparar **dos familias de modelos con modalidades de información distintas** (tiempo vs. tiempo-frecuencia) | Redes sobre la **serie temporal** (CNN 1D, LSTM/DRNN, CNN-LSTM) frente a una **CNN 2D sobre espectrogramas STFT**, más una línea base clásica sobre las 561 features | Notebooks 03–07 |
| **Integración numérica** de la aceleración y su deriva por el ruido | Se integra la aceleración, se mide la deriva en actividades estáticas (donde la velocidad real es ≈ 0), se corrige (quitar media + pasa-altos) y se prueba la velocidad integrada como entrada de la red | Notebook 02 (sección 4) y experimento A3 del 04 |
| **Validación sin ventanas del mismo sujeto en conjuntos distintos** | Todo se parte por sujeto (`GroupShuffleSplit`, `GroupKFold`); se demuestra que repartir ventanas al azar infla la accuracy en 6.5 puntos | Notebook 02 (sección 6), 03 y 07 |

## 2. Notebooks

Se ejecutan **directamente en Google Colab** (cada uno clona el repositorio, instala dependencias y descarga el dataset). Están numerados en orden de lectura y llevan sus salidas guardadas.

| # | Notebook | Contenido | Colab |
|---|---|---|---|
| 01 | [`01_exploracion_datos.ipynb`](01_exploracion_datos.ipynb) | EDA: tamaños, clases, señales por actividad, ventanas por sujeto, espectrogramas | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garcialvarez/HAR/blob/main/01_exploracion_datos.ipynb) |
| 02 | [`02_preprocesado.ipynb`](02_preprocesado.ipynb) | Normalización, magnitudes, integración numérica y deriva, STFT, fuga de información, esquema de validación | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garcialvarez/HAR/blob/main/02_preprocesado.ipynb) |
| 03 | [`03_linea_base_features.ipynb`](03_linea_base_features.ipynb) | Regresión Logística, Random Forest y SVM sobre las 561 features | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garcialvarez/HAR/blob/main/03_linea_base_features.ipynb) |
| 04 | [`04_cnn1d_dominio_tiempo.ipynb`](04_cnn1d_dominio_tiempo.ipynb) | CNN 1D: canales crudos, magnitudes, velocidad integrada, aumento de datos | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garcialvarez/HAR/blob/main/04_cnn1d_dominio_tiempo.ipynb) |
| 05 | [`05_recurrentes_lstm.ipynb`](05_recurrentes_lstm.ipynb) | LSTM, DRNN de 4 capas (artículo de referencia), BiLSTM, CNN-LSTM | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garcialvarez/HAR/blob/main/05_recurrentes_lstm.ipynb) |
| 06 | [`06_cnn2d_tiempo_frecuencia.ipynb`](06_cnn2d_tiempo_frecuencia.ipynb) | CNN 2D sobre espectrogramas (dos resoluciones, con y sin magnitudes) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garcialvarez/HAR/blob/main/06_cnn2d_tiempo_frecuencia.ipynb) |
| 07 | [`07_comparacion_y_validacion_sujetos.ipynb`](07_comparacion_y_validacion_sujetos.ipynb) | Finalistas de cada familia: test oficial y validación cruzada por sujeto sobre los 30 voluntarios | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garcialvarez/HAR/blob/main/07_comparacion_y_validacion_sujetos.ipynb) |

**Tiempos aproximados en CPU** (sin contar ≈ 1 min de descarga la primera vez)**:** 01 ≈ 15 s · 02 ≈ 1 min · 03 ≈ 3 min · 04 ≈ 5 min · 05 ≈ 15 min · 06 ≈ 6 min · 07 ≈ 17 min. En Colab, los notebooks 05 y 07 se aceleran mucho con GPU (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*).

## 3. Resultados

Cada familia se resume con su finalista, elegido por la pérdida de validación (no por el test). Las redes se evalúan sobre 2 semillas (media ± desviación) en el test oficial y sobre 1 entrenamiento por fold en la validación cruzada.

| Modelo | Familia | Test oficial (9 sujetos) | CV por sujeto (30 sujetos, 5 folds) | Latencia* | Tamaño** |
|---|---|---|---|---|---|
| **SVM RBF** (561 features) | clásica | **96.5 %** | **96.0 ± 1.1 %** | 0.4 ms | 5.0 MB |
| CNN 1D (A4) | tiempo | 91.6 ± 0.4 % | 93.1 ± 1.5 % | 0.9 ms | 230 KB |
| CNN-LSTM (R4) | tiempo | 92.1 ± 0.4 % | 92.8 ± 2.3 % | 2.1 ms | 241 KB |
| CNN 2D sobre STFT (B3) | tiempo-frecuencia | 93.3 ± 0.1 % | 91.5 ± 3.5 % | 1.2 ms | 1.1 MB |

\* Mediana por ventana, lote de 1, CPU de escritorio, sobre el grafo compilado.  
\** Redes: pesos en float32. SVM: modelo serializado con `pickle` (guarda los vectores de soporte).

**Lo que muestran los resultados**

- Las **561 features con un SVM ganan** en ambos protocolos (en línea con el 96.0 % de Anguita et al., 2013). Ninguna red superó a la línea base clásica.
- Entre las redes **el ranking depende de la partición**: la CNN 2D parece la mejor en el test oficial (9 sujetos) y es la peor en la validación cruzada. Las diferencias entre modalidades (1–2 puntos) son del orden del ruido entre sujetos y semillas, así que **no concluimos que una modalidad sea superior**.
- El error se concentra en **sentado ↔ de pie**; hay sujetos difíciles para casi todos los modelos.
- **No reproducimos el 96.7 %** de la DRNN de 4 capas de Murad & Pyun (2017): con nuestro protocolo estricto por sujeto obtiene ≈ 88.9 % (notebook 05). No determinamos la causa.
- La integración numérica corregida reduce la deriva (el cociente señal/deriva pasa de 6.9 a 25.2, notebook 02), pero **como entrada de la red no mejoró la accuracy** (experimento A3).

## 4. Datos

El dataset **no está en el repositorio**: los notebooks lo descargan solos (`src/data_loader.download_dataset()`). Instrucciones y alternativa manual en [`data/README.md`](data/README.md).

## 5. Reproducir

**En Colab (recomendado):** abrir cualquier notebook con su botón y ejecutar todas las celdas.

**En local (probado con Python 3.12):**

```bash
git clone https://github.com/garcialvarez/HAR.git
cd HAR
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter lab
```

Las semillas están fijadas (42, 43, 44); los resultados se reproducen con variaciones mínimas entre CPU y GPU.

## 6. Estructura

```
.
├─ 01_exploracion_datos.ipynb … 07_comparacion_y_validacion_sujetos.ipynb
├─ ENTREGA1.PDF
├─ INFORME_PROYECTO.PDF
├─ src/
│  ├─ data_loader.py     # descarga y carga del dataset (561 features y señales crudas)
│  ├─ preprocessing.py   # normalización, magnitudes, integración, STFT, split por sujeto
│  ├─ training.py        # preparación de entradas, entrenamiento y evaluación Keras
│  └─ evaluation.py      # métricas, latencia, tamaño, gráficas
├─ data/README.md        # cómo obtener los datos
├─ requirements.txt
├─ LICENSE · LICENSE-DATASET.txt
└─ README.md
```

Los notebooks escriben métricas en `results/` y figuras en `figures/` al ejecutarse; esas carpetas no se versionan.

## 7. Referencias

1. Anguita, D.; Ghio, A.; Oneto, L.; Parra, X.; Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013. — SVM: 96.0 %.
2. Murad, A.; Pyun, J.-Y. (2017). *Deep Recurrent Neural Networks for Human Activity Recognition.* Sensors, 17(11), 2556. [doi:10.3390/s17112556](https://doi.org/10.3390/s17112556). — DRNN unidireccional de 4 capas: 96.7 %.
3. Jiang, W.; Yin, Z. (2015). *Human Activity Recognition using Wearable Sensors by Deep Convolutional Neural Networks.* ACM Multimedia. — CNN: 95.2 %.
4. Chandan Kumar, R. et al. (2016). *Human activity recognition in cognitive environments using sequential ELM.* — ELM: 93.3 %.

## Licencia

Código bajo licencia MIT ([`LICENSE`](LICENSE)). El dataset UCI HAR tiene sus propios términos ([`LICENSE-DATASET.txt`](LICENSE-DATASET.txt)).
