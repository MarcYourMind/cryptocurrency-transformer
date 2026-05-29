# Prompt: Crear Notebook Educativo — Laboratorio de Transformers para Trading con Criptomonedas

## Instrucciones para Claude

Crea un **Jupyter Notebook** completo en español llamado `Laboratorio_Transformers.ipynb` que funcione como un laboratorio universitario paso a paso. Un analista de datos junior debe poder ejecutar cada celda, entender cada concepto, y reconstruir este proyecto entero desde cero mientras aprende.

---

## Contexto del Proyecto

Este proyecto es un **sistema de trading algorítmico** que usa un **Transformer Encoder** (tipo GPT) para predecir si una operación long alcanzará su take-profit antes que su stop-loss, operando **exclusivamente en mercados laterales (sideways)** de criptomonedas, con ratio riesgo:beneficio de 1:1 basado en ATR.

La arquitectura central es un `VolumeProfileTransformer` que toma un perfil de volumen de 64 bins como secuencia de entrada y 5 características contextuales, produciendo una probabilidad de éxito para cada posible punto de entrada dentro del rango lateral.

El proyecto logró un **~60% de win rate realista** después de corregir tres fugas de datos críticas que inicialmente producían un falso 85%.

---

## Estructura Requerida del Notebook

El notebook debe tener las siguientes secciones, cada una con **texto explicativo rico**, **código ejecutable**, y **preguntas de reflexión** al final de cada sección. El tono debe ser el de un profesor universitario que explica con paciencia pero rigor.

---

### Sección 0: Introducción y Objetivos del Laboratorio

- Explicar qué van a aprender: arquitectura de transformers, ingeniería de features financieros, detección de regímenes de mercado, backtesting, y los errores comunes que causan resultados falsos.
- Listar los prerrequisitos: Python, conceptos básicos de ML, nociones de trading.
- Mostrar el mapa general del proyecto: desde la descarga de datos hasta el trading en vivo.
- Incluir un diagrama ASCII del pipeline completo:

```
Datos Binance → Detección Lateral → Perfil de Volumen + Contexto
       ↓                                        ↓
   División Train/Val/Test          VolumeProfileTransformer
       ↓                                        ↓
   Entrenamiento con Early Stop      Probabilidad por bin
       ↓                                        ↓
   Backtest cronológico            Mejor entrada → Señal
       ↓
   Bot de Trading (BingX)
```

---

### Sección 1: ¿Qué es un Transformer? De cero a entendimiento

**Esta es la sección más importante. Debe ser extensa y pedagógica.**

- **1.1** Explicar la motivación: ¿Por qué transformers y no RNNs/LSTMs? Problemas de secuencia larga, vanish gradients, paralelización.
- **1.2** El mecanismo de atención (Self-Attention): explicar con un ejemplo numérico concreto paso a paso.
  - Mostrar una secuencia de 4 vectores de dimensión 3
  - Calcular Q, K, V con matrices de peso inventadas
  - Mostrar la fórmula: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
  - Calcular los scores paso a paso en una tabla
- **1.3** Multi-Head Attention: por qué múltiples cabezas, cómo se concatenan, la proyección final.
- **1.4** Positional Encoding: por qué los transformers necesitan posición, la fórmula sinusoidal, y por qué aquí usamos **positional encoding aprendido** en vez de sinusoidal.
- **1.5** Feed-Forward Network dentro de cada capa del encoder.
- **1.6** Layer Normalization y conexiones residuales.
- **1.6** Stack de capas: cómo se apilan N capas de encoder.
- **1.7** Implementación desde cero de un mini-transformer con PyTorch (sin usar `nn.TransformerEncoder`), paso a paso, para que el estudiante vea cada componente.
- **1.8** Luego mostrar cómo PyTorch lo simplifica con `nn.TransformerEncoderLayer` y `nn.TransformerEncoder`.
- **Preguntas de reflexión:**
  - ¿Qué pasaría si eliminamos el positional encoding?
  - ¿Por qué dividimos por sqrt(d_k) en la atención?
  - ¿Cuál es la diferencia entre encoder-only y decoder-only?

---

### Sección 2: El Problema — Predecir en Mercados Laterales

- **2.1** ¿Qué es un mercado lateral (sideways/range-bound)? Explicar con ejemplos visuales (generar un gráfico con matplotlib que muestre un segmento sideways vs trending).
- **2.2** ¿Por qué solo operar en laterales? La lógica de la reversión a la media. Por qué los趋势 (trends) son más difíciles de predecir con este enfoque.
- **2.3** Detección de regímenes laterales: implementar `is_sideways()` paso a paso.
  - Mostrar la regresión lineal sobre precios normalizados
  - Explicar el umbral de pendiente (`SlopeThreshold = 0.0002`)
  - Visualizar ventanas que pasan y no pasan el filtro
- **2.4** Ratio riesgo:beneficio 1:1 con ATR: explicar qué es ATR, cómo se calcula, y por qué usar ATR en vez de porcentajes fijos.
- **Preguntas de reflexión:**
  - ¿Por qué es importante filtrar solo mercados laterales?
  - ¿Qué sucede si el umbral de pendiente es muy estricto o muy laxo?

---

### Sección 3: Ingeniería de Features — El Perfil de Volumen

- **3.1** ¿Qué es un perfil de volumen (Volume Profile)? Explicar el concepto de mercado con analogías visuales.
- **3.2** Implementar `compute_volume_profile()` paso a paso:
  - Tomar una ventana de 288 velas (24 horas)
  - Calcular min y max del rango de precios
  - Dividir en 64 bins iguales
  - Asignar volumen a cada bin usando el precio medio de cada vela
  - Normalizar dividiendo por el valor máximo del bin
- **3.3** Visualizar un perfil de volumen real: generar un gráfico horizontal de barras donde el eje Y son los bins de precio y el eje X es el volumen normalizado.
- **3.4** ¿Por qué 64 bins? Explicar el trade-off entre resolución y ruido.
- **3.5** La decisión de diseño clave: **por qué usamos perfil de volumen y no precios brutos**. Esto hace que el modelo sea price-agnostic — un rango de BTC a $40k y uno de ADA a $0.50 se vuelven el mismo vector de 64 dimensiones.
- **3.6** Las 5 características contextuales: explicar cada una:
  - ATR (Average True Range): volatilidad del rango
  - Volatilidad: desviación estándar de los retornos
  - Volumen promedio: actividad del mercado
  - Pendiente (trend slope): direccionalidad de la ventana
  - Bin normalizado de entrada: cuál de los 64 bins es el candidato a entrada
- **3.7** Implementar `get_context_features()` y mostrar un ejemplo con datos reales.
- **Preguntas de reflexión:**
  - ¿Por qué es importante que el modelo sea "price-agnostic"?
  - ¿Qué otras características podrías añadir al vector de contexto?

---

### Sección 4: Etiquetado — Simulación Forward

- **4.1** Explicar el etiquetado: para cada una de las 64 posibles entradas (bins de precio), ¿alcanzará el take-profit antes que el stop-loss?
- **4.2** Implementar la lógica de simulación forward paso a paso:
  - Para cada bin, calcular el precio de entrada (centro del bin)
  - Verificar si el precio de entrada fue alcanzado en la ventana futura
  - Si fue alcanzado, simular la operación: ¿TP o SL primero?
  - Etiqueta: 1 si TP primero, 0 si SL primero
  - Entradas no alcanzadas → se descartan (no se incluyen en el dataset)
- **4.3** Visualizar un ejemplo con una ventana real: mostrar el gráfico de velas, los 64 bins, y cuáles resultan en win/loss.
- **4.4** La decisión de evaluar **todos los 64 bins** en vez de uno solo: esto permite buscar el mejor punto de entrada dentro del rango.
- **Preguntas de reflexión:**
  - ¿Por qué descartamos muestras donde el precio de entrada nunca se alcanza?
  - ¿Qué sesgo introduciría incluir esas muestras con etiqueta 0?

---

### Sección 5: Descarga y Preparación de Datos

- **5.1** Conectar con la API de Binance para obtener los top 100 pares USDT por volumen.
- **5.2** Descargar datos OHLCV de 5 minutos (~2.7 años por símbolo).
- **5.3** Explorar los datos: mostrar shape, head, estadísticas descriptivas.
- **5.4** Implementar la ventana deslizante: 288 velas lookback, 36 velas forward, stride de 4.
- **5.5** Aplicar el filtro sideways y mostrar cuántas ventanas pasan vs cuántas se rechazan.
- **5.6** Generar el dataset completo: perfiles de volumen + contexto + etiquetas.

---

### Sección 6: Fugas de Datos — Las Tres Fallas Críticas

**Esta sección es crucial. Debe explicar en detalle cada fuga, por qué es un error, y cómo se corrigió.**

- **6.1 Fuga #1 — División de datos incorrecta (concatenar antes de dividir)**
  - El error: concatenar todos los símbolos y luego dividir train/val/test
  - Por qué es una fuga: datos del mismo símbolo en diferentes splits comparten patrones temporales
  - La corrección: dividir **por símbolo** cronológicamente (70/15/15), luego concatenar
  - Mostrar visualmente la diferencia con un diagrama

- **6.2 Fuga #2 — Sesgo de selección (solo contar trades donde se alcanzó la entrada)**
  - El error: solo incluir muestras donde el precio de entrada fue alcanzado
  - Por qué es un sesgo: las entradas que nunca se alcanzan serían pérdidas (operación nunca se ejecuta), pero el modelo nunca ve esos casos
  - La corrección: marcar las entradas no alcanzadas como "No Trade" y excluirlas del entrenamiento, pero tenerlas en cuenta en el backtest
  - Mostrar el impacto numérico

- **6.3 Fuga #3 — Escalado global (fit scaler en todos los datos)**
  - El error: ajustar el StandardScaler en train+val+test juntos
  - Por qué es una fuga: el scaler "ve" estadísticas del set de test
  - La corrección: fit solo en train, transform en val y test
  - Mostrar código comparativo

- **6.4** Resultado: con las fugas, el modelo reportaba ~85% de win rate. Sin fugas, ~60%. Esto es un **lesson** sobre la importancia de la integridad de datos en ML financiero.
- **Preguntas de reflexión:**
  - ¿Puedes pensar en otras posibles fugas de datos en pipelines financieros?
  - ¿Por qué es tan fácil engañarse con resultados falsos en trading?

---

### Sección 7: El Modelo — VolumeProfileTransformer

- **7.1** Arquitectura detallada con diagrama ASCII:

```
Input 1: Volume Profile (64 bins) → Linear(1,128) + Positional Encoding(1,64,128)
                                        ↓
                              TransformerEncoder(4 layers, 4 heads, d=128)
                                        ↓
                                   Mean Pool → 128-dim

Input 2: Context Features (5 dims) → Linear(5,128) → 128-dim

Combined: [profile_pool ; context] (256-dim) → MLP(256→64→1) → Logit → Sigmoid → P(win)
```

- **7.2** Implementar el modelo paso a paso:
  - `bin_embedding`: proyectar cada bin de 1 dimensión a 128
  - `pos_encoder`: encoding posicional aprendido
  - `transformer`: stack de 4 encoder layers
  - `context_projection`: proyectar contexto de 5 a 128
  - `head`: MLP de clasificación (256→64→1)
  - Explicar por qué no hay sigmoid en el modelo (se usa BCEWithLogitsLoss)
- **7.3** Explicar cada hiperparámetro y por qué se eligió:
  - EmbedDim=128: trade-off entre capacidad y overfitting
  - NumHeads=4: suficiente para capturar relaciones entre bins
  - NumLayers=4: profundidad suficiente sin exceso
  - Dropout=0.1: regularización moderada
  - ContextDim=5: las 5 features definidas
  - NumVolumeBins=64: resolución del perfil de volumen
- **7.4** Mostrar el modelo instanciado y un forward pass con datos dummy para verificar dimensiones.
- **7.5** Contar los parámetros totales del modelo.
- **Preguntas de reflexión:**
  - ¿Por qué mean-pooling y no usar el token [CLS]?
  - ¿Qué pasaría si aumentamos/diminuimos el número de capas?

---

### Sección 8: Entrenamiento

- **8.1** Preparar los datasets y dataloaders con el split por símbolo.
- **8.2** Balance de clases: calcular pos_weight para BCEWithLogitsLoss.
  - Explicar por qué el desbalance de clases es problemático en trading
  - Mostrar el cálculo: `pos_weight = neg_count / pos_count`
- **8.3** Configurar el optimizador Adam con learning rate 1e-3.
- **8.4** El loop de entrenamiento:
  - Forward pass
  - Cálculo de pérdida con BCEWithLogitsLoss
  - Backward pass
  - Gradient accumulation (si aplica)
  - Logging cada N batches
- **8.5** Early Stopping: explicar por qué es crucial en trading (evitar overfitting a ruido del mercado).
  - Paciencia de 5 epochs
  - Guardar el mejor modelo basado en validation loss
- **8.6** Mixed precision training: explicar qué es y por qué ahorra memoria GPU.
- **8.7** Ejecutar el entrenamiento (o cargar modelo pre-entrenado si está disponible).
- **8.8** Graficar las curvas de training loss y validation loss.
- **Preguntas de reflexión:**
  - ¿Por qué early stopping y no simplemente menos epochs?
  - ¿Qué indicaría un validation loss que sube mientras el training loss baja?

---

### Sección 9: Evaluación y Curva de Calibración

- **9.1** Cargar el mejor modelo y evaluar en el set de test.
- **9.2** Mostrar el classification report (precision, recall, f1).
- **9.3** Curva de calibración: explicar qué es y por qué importa en trading.
  - Un modelo bien calificado: cuando dice 60%, gana ~60% de las veces
  - Mostrar la curva de calibración y comparar con la línea diagonal perfecta
- **9.4** Distribución de predicciones: histograma de probabilidades predichas.
- **9.5** Analizar casos donde el modelo está más confiado y dónde se equivoca.
- **Preguntas de reflexión:**
  - ¿Por qué la calibración es más importante que la accuracy en trading?
  - ¿Cómo usarías el umbral de probabilidad para filtrar señales?

---

### Sección 10: Backtesting

- **10.1** Explicar qué es backtesting y por qué es esencial antes de operar en vivo.
- **10.2** El proceso de backtesting paso a paso:
  - Para cada símbolo en el set de test
  - Deslizar ventanas que pasen el filtro sideways
  - Para cada ventana, evaluar los 64 bins de entrada
  - Seleccionar el bin con mayor probabilidad predicha
  - Si la probabilidad supera el umbral, ejecutar la operación
  - Simular el resultado: ¿TP o SL primero?
- **10.3** Cálculo de la curva de equity con capital compuesto:
  - Riesgo del 1% del balance por operación
  - Tamaño de posición basado en ATR
  - Capital compuesto (los resultados de trades cerrados afectan el balance)
- **10.4** Métricas de backtesting:
  - Win rate
  - F1 Score
  - AUC-ROC
  - Profit factor
  - Maximum drawdown
  - Sharpe ratio (si es posible)
- **10.5** Generar la curva de equity con matplotlib.
- **10.6** Advertencias sobre backtesting: overfitting, survivorship bias, slippage, latency.
- **Preguntas de reflexión:**
  - ¿Por qué un backtest rentable no garantiza rentabilidad en vivo?
  - ¿Qué es el sesgo de supervivencia y cómo afecta este sistema?

---

### Sección 11: Inferencia en Vivo — La Clase Strategy

- **11.1** Explicar cómo se usa el modelo en producción:
  - Obtener las últimas 288 velas del exchange (BingX)
  - Verificar si el mercado está en régimen sideways
  - Calcular el perfil de volumen y contexto
  - Inferir para los 64 bins de entrada
  - Retornar la señal con mayor probabilidad si supera el umbral
- **11.2** Mostrar el código de `Strategy.get_signal()` y ejecutarlo con datos de ejemplo.
- **11.3** Umbral de probabilidad: explicar cómo elegirlo y su impacto en la frecuencia de señales vs la win rate.

---

### Sección 12: El Bot de Trading

- **12.1** Explicar la arquitectura del bot:
  - Loop principal que ejecuta cada 5 minutos (al cierre de vela)
  - Verificación de señales para cada símbolo
  - Gestión de órdenes (open → filled → closed)
  - Persistencia de estado en `trades_state.json`
- **12.2** Gestión de riesgo:
  - 1% del balance por operación
  - Máximo 10% del equity en una sola posición
  - Distancia mínima de SL del 0.1% del precio
  - Apalancamiento configurable (default: 5x)
- **12.3** El cliente BingX: explicar cómo ccxt facilita la conexión con exchanges.
- **12.4** Modo DRY_RUN: paper trading seguro sin dinero real.
- **12.5** Dashboard web: FastAPI + HTML para monitoreo en tiempo real.
- **Preguntas de reflexión:**
  - ¿Qué riesgos adicionales existen al operar en vivo vs backtesting?
  - ¿Por qué es importante empezar con paper trading?

---

### Sección 13: Conclusiones y Lecciones Aprendidas

- Resumir lo aprendido:
  1. Los transformers pueden procesar datos secuenciales financieros (perfiles de volumen)
  2. La ingeniería de features es tan importante como la arquitectura del modelo
  3. Las fugas de datos en ML financiero son fáciles de cometer y devastadoras
  4. La calibración importa más que la accuracy en trading
  5. Operar solo en regímenes laterales reduce el problema a algo más manejable
  6. El backtesting es necesario pero no suficiente
- Enumerar limitaciones y posibles mejoras:
  - Solo opera long (añadir shorts)
  - Solo usa perfil de volumen (añadir order flow, indicadores técnicos)
  - Ventana fija de 24h (ventanas adaptativas)
  - 1:1 R:R fijo (optimizar dinámicamente)
  - Más capas/features para el modelo
- Preguntas de reflexión finales:
  - ¿Cómo mejorarías este sistema?
  - ¿Qué otros mercados podrían beneficiarse de este enfoque?

---

## Estilo y Formato

- **Idioma:** Todo el texto explicativo en español. Los nombres de variables y código en inglés (como el proyecto original).
- **Tono:** Profesoral, paciente, riguroso. Como un laboratorio universitario de posgrado.
- **Código:** Cada celda de código debe ser ejecutable de forma independiente (asumiendo que las celdas anteriores se ejecutaron en orden). Incluir imports necesarios al inicio de cada sección relevante.
- **Visualizaciones:** Generar gráficos con matplotlib en cada sección relevante (distribución de datos, arquitectura del modelo, curvas de entrenamiento, curva de equity, etc.).
- **Preguntas:** Al final de cada sección, incluir 2-3 preguntas de reflexión en una celda Markdown con formato de cita (`>`).
- **Advertencias:** En secciones donde puedan ocurrir errores comunes, incluir callouts con `⚠️`.
- **Datos:** Usar los datos locales del directorio `data/` si existen. Si no, descargar un subconjunto pequeño (5-10 símbolos) para demostración.
- **Hardware:** Asumir que el estudiante puede no tener GPU. Todo el notebook debe funcionar en CPU (con tiempos razonables usando un subconjunto de datos).
- **Celdas Markdown vs Code:** Las explicaciones teóricas en Markdown, el código ejecutable en celdas Code. No mezclar grandes bloques de texto en celdas de código.

---

## Requisitos Técnicos

- Python 3.9+
- PyTorch (CPU es suficiente)
- numpy, pandas, matplotlib, scikit-learn
- requests (para datos de Binance)
- joblib (para guardar/cargar scaler)
- El notebook debe poder ejecutarse desde el directorio raíz del proyecto (`C:\Users\Nexus\code\transformer`)
- Las imports del proyecto deben usar `from src.config import Config` y `from src.train_gpt import ...` para reutilizar el código existente donde sea posible, pero también reimplementar funciones clave desde cero para que el estudiante entienda cada paso.

---

## Nota Final

Este notebook debe ser una **experiencia educativa completa**. Un estudiante que lo complete debería ser capaz de:
1. Explicar cómo funciona un transformer desde sus componentes fundamentales
2. Diseñar un pipeline de ML para trading desde cero
3. Identificar y prevenir fugas de datos en cualquier proyecto de ML financiero
4. Entender cada decisión de diseño de este proyecto y justificarla
5. Modificar y extender el sistema con nuevas ideas

Escribe el notebook completo, no dejes secciones como "ejercicio para el lector". Cada sección debe tener el código funcional y las respuestas provistas.