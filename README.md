# Reconocimiento de Actividades Humanas a partir de Sensores Inerciales de Smartphones (HAR)

Entrega 1 (propuesta y línea base).

Reporte completo de la propuesta: [`docs/ENTREGA1.PDF`](docs/ENTREGA1.PDF)

Análisis exploratorio que sustenta el reporte: [`notebooks/01_analisis_exploratorio.ipynb`](notebooks/01_analisis_exploratorio.ipynb) ([abrir en Colab](https://colab.research.google.com/github/garcialvarez/HAR/blob/main/notebooks/01_analisis_exploratorio.ipynb))

## 1. Contexto de aplicación

El reconocimiento de actividad humana (Human Activity Recognition, HAR) a partir de sensores inerciales embebidos en smartphones es un problema central en computación ubicua, salud digital y bienestar. A diferencia de sistemas basados en cámaras, un smartphone portado por el usuario permite monitoreo continuo, de bajo costo y no intrusivo.

Aplicaciones prácticas: apps de fitness que cuantifican actividad física diaria, sistemas de asistencia a adultos mayores (detección de caídas / inactividad prolongada), interfaces adaptativas según contexto físico del usuario, y telemetría de salud en poblaciones clínicas (Por ejemplo detección de congelamiento de la marcha en pacientes con Parkinson).

## 2. Objetivo

**Predecir la actividad física que realiza una persona (6 clases) a partir de señales inerciales de un smartphone (acelerómetro y giroscopio de 3 ejes) capturadas en ventanas de 2.56 s.**

Es un problema de clasificación multiclase supervisada: dado un vector de entrada `x` (señal cruda de 9 canales × 128 pasos, o el vector de 561 features ya extraídas), aprender `f(x) → y`, con:

```
y ∈ {WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING}
```

## 3. Dataset

**[UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)** — 30 voluntarios (19-48 años), Samsung Galaxy S II en la cintura, acelerómetro + giroscopio a 50 Hz, ventanas de 128 muestras (2.56 s, 50% traslape).

| Partición | # Sujetos | # Ventanas | # Features |
|---|---|---|---|
| Train | 21 | 7,352 | 561 |
| Test | 9 | 2,947 | 561 |
| **Total** | **30** | **10,299** | **561** |

- **Tipo de datos:** señales inerciales crudas (9 canales × 128 pasos por ventana) **y** un vector de 561 características de dominio tiempo/frecuencia ya normalizadas en `[-1, 1]`.
- **Distribución de clases:** razonablemente balanceada (ratio máx/mín ≈ 1.43); ver detalle en el reporte.
- **Partición:** *subject-wise* (70%/30% de sujetos, no de ventanas), evitando fuga de información entre train y test.


## 4. Métricas de desempeño

Machine learning: accuracy global, precisión/recall/F1 por clase (macro y ponderado), matriz de confusión, y validación cruzada por sujeto (leave-subjects-out) para estimar generalización a usuarios nuevos.

Negocio: latencia de inferencia por ventana (ms) para viabilidad de on-device inference, consumo computacional/energético, tasa de falsos negativos en clases de riesgo (p. ej. detección de inactividad/caídas), y robustez ante sujetos/dispositivos no vistos en entrenamiento.

## 5. Referencias y resultados previos

1. Anguita, D. et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013. — SVM: **96.0%** accuracy.
2. Murad, A.; Pyun, J.-Y. (2017). *Deep Recurrent Neural Networks for Human Activity Recognition*. **Sensors**, 17(11), 2556. [doi:10.3390/s17112556](https://doi.org/10.3390/s17112556). — Referencia principal; DRNN unidireccional (4 capas): **96.7%** accuracy, 96.8% precisión, 96.7% recall, F1 = 0.96.
3. Jiang, W.; Yin, Z. (2015). CNN baseline usada en (2): **95.2%** accuracy.
4. Chandan Kumar, R. et al. (2016). ELM secuencial baseline usada en (2): **93.3%** accuracy.

---

## Estructura del repositorio

```
.
├─ docs/
│  └─ ENTREGA1.PDF
├─ notebooks/
|	└─ 01_analisis_exploratorio.ipynb
├─ LICENSE
├─ LICENSE-DATASET.txt
├─ README.md
├─ requirements.txt
├─ .gitignore
```


