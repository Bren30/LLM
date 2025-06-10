import os
import logging
import pandas as pd
import camelot
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import shutil # Para limpiar directorio de FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURACIÓN DE ARCHIVOS Y CARPETAS ---
FOLDER_INPUT_PDFS = r"C:/Users/BRENDA/Desktop/BCP/PROYECTO" # ¡¡¡ACTUALIZA ESTA RUTA!!!

PDF_NUESTRO_BANCO_FILENAME = "CREDICORP 2025 20F.pdf"
NOMBRE_NUESTRO_BANCO_PROMPT = "Credicorp" # Constante global para nuestro banco

FAISS_INDEX_PATH = "faiss_index_bancos_v11_final_output" # Nombre de índice para esta versión
REBUILD_FAISS_INDEX = True # PONER EN TRUE PARA LA PRIMERA EJECUCIÓN O SI CAMBIAS PDFs

# --- LISTA DE PARÁMETROS CLAVE Y QUERIES ---
PARAMETROS_CLAVE = [
    {
        "nombre_parametro": "Probability of Default (PD)",
        "query_rag_nuestro_banco_template": f"Metodología actual de cálculo y estimación de Probabilidad de Incumplimiento (PD) {NOMBRE_NUESTRO_BANCO_PROMPT}",
        "query_rag_banco_externo_template": "Metodología, cambios o actualizaciones en cálculo y estimación de Probabilidad de Incumplimiento (PD) {nombre_banco_externo}, justificación y factores específicos de segmentación y modelado",
        "aspectos_parametro": "definición de default, segmentación (ej. por tipo de cliente, producto, rating), variables clave (listar si se mencionan), modelos (scoring, rating, machine learning), recalibraciones, datos históricos, forward-looking (factores específicos), impacto cuantitativo del cambio si lo hay, fechas."
    },
    {
        "nombre_parametro": "Loss Given Default (LGD)",
        "query_rag_nuestro_banco_template": f"Metodología actual de estimación de Pérdida Dado el Incumplimiento (LGD) {NOMBRE_NUESTRO_BANCO_PROMPT}, tratamiento de colaterales y recuperaciones",
        "query_rag_banco_externo_template": "Metodología (explicar detalladamente términos como 'Closed Recoveries', 'Open Recoveries', 'flujos de caja descontados' si aparecen y cómo se aplican), cambios o actualizaciones en estimación de Pérdida Dado el Incumplimiento (LGD) {nombre_banco_externo}, tipos de colateral, tasas de recuperación, descuentos de flujos, costos de recuperación, justificación",
        "aspectos_parametro": "metodologías (ej. workout, ELBE, Flujos de Caja Descontados, modelos estadísticos), segmentación (ej. por tipo de garantía, producto), tipos de colateral valorados, tasas de recuperación observadas o estimadas, tratamiento de costos de recuperación, impacto cuantitativo del cambio si lo hay, fechas."
    },
    {
        "nombre_parametro": "Exposure at Default (EAD)",
        "query_rag_nuestro_banco_template": f"Metodología actual de estimación de Exposición al Incumplimiento (EAD) {NOMBRE_NUESTRO_BANCO_PROMPT}, factores de conversión crediticia (CCF)",
        "query_rag_banco_externo_template": "Metodología (explicar Factores de Conversión Crediticia - CCF si se mencionan y cómo se segmentan o aplican), cambios o actualizaciones en estimación de Exposición al Incumplimiento (EAD) {nombre_banco_externo}, tratamiento de saldos dispuestos y no dispuestos, justificación y detalles de segmentación",
        "aspectos_parametro": "tratamiento de saldos dispuestos y no dispuestos, líneas de crédito (ej. rotativas, no rotativas), productos específicos, Factores de Conversión Crediticia (CCF) y cómo se estiman o aplican (ej. segmentados, fijos, por comportamiento), prepagos, impacto cuantitativo del cambio si lo hay, fechas."
    },
    {
        "nombre_parametro": "Expected Credit Loss (ECL) y Staging",
        "query_rag_nuestro_banco_template": f"Metodología actual de Pérdida Crediticia Esperada (ECL) {NOMBRE_NUESTRO_BANCO_PROMPT} y criterios de clasificación por etapas (Stage 1, 2, 3), transferencias",
        "query_rag_banco_externo_template": "Metodología de Pérdida Crediticia Esperada (ECL) {nombre_banco_externo}, criterios de etapas (Stage 1, 2, 3), transferencias entre etapas, justificación, factores y cualquier cambio o actualización",
        "aspectos_parametro": "fórmula PDxLGDxEAD, ponderación de escenarios macroeconómicos (listar escenarios y ponderaciones si se mencionan), definición y criterios de Stages (cuantitativos y cualitativos), umbrales de transferencia, impacto cuantitativo del cambio en ECL total o por Stage si lo hay, fechas."
    },
    {
        "nombre_parametro": "Significant Increase in Credit Risk (SICR)",
        "query_rag_nuestro_banco_template": f"Criterios actuales para Aumento Significativo en el Riesgo de Crédito (SICR) {NOMBRE_NUESTRO_BANCO_PROMPT}",
        "query_rag_banco_externo_template": "Metodología, cambios o actualizaciones en criterios para Aumento Significativo en el Riesgo de Crédito (SICR) {nombre_banco_externo}, justificación, umbrales específicos cuantitativos y cualitativos y cómo se combinan",
        "aspectos_parametro": "criterios cuantitativos (listar variaciones PD absoluta/relativa y umbrales si se mencionan), criterios cualitativos (listas de vigilancia, refinanciamientos, días de mora, alertas tempranas), backtesting, impacto cuantitativo del cambio si lo hay, fechas."
    },
    {
        "nombre_parametro": "Forward-Looking Information (FLI)",
        "query_rag_nuestro_banco_template": f"Uso actual de información prospectiva (FLI) en modelos de riesgo {NOMBRE_NUESTRO_BANCO_PROMPT}, escenarios macroeconómicos",
        "query_rag_banco_externo_template": "Metodología, cambios o actualizaciones en uso de información prospectiva (FLI) {nombre_banco_externo}, escenarios macroeconómicos (listar variables específicas, ponderaciones, fuentes si se mencionan), justificación y cómo se integran en PD, LGD o ECL",
        "aspectos_parametro": "variables macroeconómicas clave (listar si se mencionan, ej. PBI, desempleo, tasas de interés, inflación sectorial), número y ponderación de escenarios (ej. base, optimista, pesimista, severo), fuentes de proyecciones, frecuencia de actualización, overlays de gestión basados en FLI, impacto cuantitativo del cambio si lo hay, fechas."
    }
]

# --- PLANTILLAS DE PROMPT (CORREGIDAS Y REFINADAS) ---
TEMPLATE_DESCRIBIR_METODOLOGIA_NUESTRO_BANCO_POR_PARAMETRO = f"""
Actúa como un experto Principal de Metodologías de Riesgo de '{NOMBRE_NUESTRO_BANCO_PROMPT}'.
Describe de forma DETALLADA y PRECISA la METODOLOGÍA ACTUAL que '{NOMBRE_NUESTRO_BANCO_PROMPT}' utiliza específicamente para el parámetro: "{{nombre_parametro}}", basándote **estrictamente en el CONTEXTO proporcionado**.
Este análisis es para CONTEXTO INTERNO. Enfócate en los aspectos mencionados en: {{aspectos_relevantes_parametro}}.
CITA PÁGINAS (ej: "pág. X") para información clave.
Si un aspecto no se detalla explícitamente en el contexto, indícalo claramente como "No detallado en el contexto proporcionado".
**FORMATO DE SALIDA:** Utiliza Markdown. Usa títulos (`### Metodología...`), negritas (`**Aspecto**`) y viñetas (`* Detalle`) para claridad.

CONTEXTO DE '{NOMBRE_NUESTRO_BANCO_PROMPT}' (PARA {{nombre_parametro}}):
---
{{context}}
---

### Descripción Contextual de la Metodología de '{NOMBRE_NUESTRO_BANCO_PROMPT}' para {{nombre_parametro}}
*(Basado estrictamente en el contexto proporcionado)*
"""

TEMPLATE_EXTRACCION_CAMBIOS_BANCO_EXTERNO_POR_PARAMETRO = """
Actúa como un Consultor Senior experto en benchmarking de metodologías de riesgo IFRS 9 para bancos.
Tu tarea es analizar el CONTEXTO del reporte financiero del BANCO EXTERNO ('{nombre_banco_externo_prompt}') para identificar, detallar y EXPLICAR su metodología y cualquier CAMBIO METODOLÓGICO relevante para el parámetro: "{nombre_parametro}".
El objetivo es entender A FONDO qué está haciendo el BANCO EXTERNO ('{nombre_banco_externo_prompt}') para que """ + NOMBRE_NUESTRO_BANCO_PROMPT + """ pueda evaluar oportunidades de mejora.
Considera los siguientes aspectos: {aspectos_a_buscar_en_cambios}.

CONTEXTO DEL BANCO EXTERNO ('{nombre_banco_externo_prompt}') (PARA {nombre_parametro}):
---
{context}
---

INSTRUCCIONES ESTRICTAS PARA TU ANÁLISIS (Enfócate en la EXPLICACIÓN y DETALLE de '{nombre_banco_externo_prompt}'):
Utiliza formato MARKDOWN. Emplea encabezados, negritas, itálicas y viñetas. CITA PÁGINAS (ej: `(pág. X)`).

**Prioridad 1: Extraer y Explicar desde el Contexto del PDF del Competidor.**
*   Para cada práctica, metodología o término técnico relevante (ej: "flujos de caja descontados", "CCF", "segmentación específica PD/LGD/EAD", "datos alternativos específicos", "sistemas de monitoreo proactivo"), **primero busca exhaustivamente en el CONTEXTO proporcionado una explicación o detalle de cómo lo aplica el BANCO EXTERNO ('{nombre_banco_externo_prompt}').**
*   Si el contexto describe la segmentación (ej. "segmentación de PD por rating interno y tipo de cliente industrial (pág. Z)"), detállala citando la página. Si menciona "utilizan un enfoque paramétrico para PD en la cartera mayorista y pérdida histórica para minorista (pág. W)", explícalo.
*   Si el contexto menciona "factores específicos para EAD en créditos revolventes como [X, Y, Z] (pág. W)", lístalos.
*   Si el contexto explica un término como "Closed Recoveries aplicados a cartera hipotecaria con un horizonte de 72 meses (pág. V)", inclúyelo.

**Prioridad 2: Explicación General como Respaldo (SI Y SOLO SI el contexto es insuficiente Y el concepto es CLAVE para una oportunidad).**
*   Si el CONTEXTO menciona un término o práctica clave (ej. "uso de Flujos de Caja Descontados para LGD", "segmentación granular de PD", "CCF para EAD") pero **NO EXPLICA DETALLADAMENTE** cómo lo implementa '{nombre_banco_externo_prompt}' o qué significa el término en sí, y consideras que es una **oportunidad significativa** para """ + NOMBRE_NUESTRO_BANCO_PROMPT + """, entonces:
    1.  Indica claramente: *"El contexto de '{nombre_banco_externo_prompt}' menciona [TÉRMINO/PRÁCTICA CLAVE] (pág. Y) pero no detalla su implementación específica o la definición del término. Basado en el conocimiento general de la industria bancaria, [TÉRMINO/PRÁCTICA CLAVE] típicamente implica: [EXPLICACIÓN GENERAL CONCISA y relevante para riesgo crediticio, 2-4 frases máximo, enfocada en QUÉ ES y POR QUÉ ES ÚTIL]."*
    2.  **No inventes detalles específicos sobre la implementación del competidor que no estén en el texto.** La explicación general debe ser sobre el concepto en sí, para que el equipo de """ + NOMBRE_NUESTRO_BANCO_PROMPT + """ entienda la idea.

### Análisis de Metodología y Cambios en '{nombre_banco_externo_prompt}' para el parámetro: "{nombre_parametro}"

#### 1. Metodología Actual de '{nombre_banco_externo_prompt}' para "{nombre_parametro}"
*   (Describe CLARAMENTE la metodología actual basándote en el CONTEXTO. Detalla segmentaciones, variables, modelos, etc., si el contexto los provee. Si un término técnico es mencionado y explicado en el contexto, inclúyelo. Si es mencionado pero no explicado y es CLAVE para una oportunidad, sigue la Prioridad 2 para explicar el concepto general).
*   (Ejemplo con Prioridad 2 para PD: "Para PD, {nombre_banco_externo_prompt} menciona una 'segmentación granular' para su cartera minorista (pág. X) pero no especifica los criterios. Basado en conocimiento de la industria, la segmentación granular de PD podría implicar dividir la cartera en subgrupos más pequeños y homogéneos usando variables como tipo de producto, score crediticio, comportamiento de uso de cuenta, datos sociodemográficos, etc., para asignar probabilidades de default más precisas a cada subgrupo.")
*   (Ejemplo con Prioridad 2 para LGD/FCD: "Para LGD, {nombre_banco_externo_prompt} indica que utiliza 'Flujos de Caja Descontados (FCD)' para ciertos portafolios (pág. Y), pero no explica el método en detalle. Basado en conocimiento de la industria, FCD para LGD implica proyectar los flujos de efectivo futuros esperados de la recuperación de un activo en default (ej. pagos parciales, venta de colateral) y descontarlos a valor presente usando una tasa apropiada que refleje el riesgo y el tiempo. Esto busca una valoración más económica de la pérdida.")
*   (Ejemplo con Prioridad 2 para EAD/CCF: "{nombre_banco_externo_prompt} menciona el uso de 'Factores de Conversión Crediticia (CCF)' para estimar la EAD de exposiciones fuera de balance (pág. Z), sin detallar su derivación o segmentación. En la industria, los CCF son porcentajes que estiman qué porción de un compromiso no dispuesto (ej. línea de crédito no utilizada) se espera que sea utilizada por el cliente al momento del default. Segmentar estos CCF por tipo de producto, cliente o comportamiento puede mejorar la precisión del EAD.")

#### 2. Cambios Metodológicos Identificados en '{nombre_banco_externo_prompt}' para "{nombre_parametro}" (Si los hay según el contexto)
*   (Para CADA cambio metodológico identificado en el CONTEXTO):
    *   **Descripción Detallada del Cambio:** (¿En qué consistió EXACTAMENTE el cambio según el contexto? CITA PÁGINA).
    *   **Justificación/Razón del Cambio:** (Según el contexto. CITA PÁGINA).
    *   **Impacto Específico del Cambio:** (Según el contexto. CITA PÁGINA).
*   (Si no se mencionan cambios explícitos, indícalo: "No se identificaron cambios metodológicos explícitos para '{nombre_parametro}' en el contexto proporcionado de {nombre_banco_externo_prompt}.")

#### 3. Observaciones Relevantes Adicionales y Potenciales Oportunidades para """ + NOMBRE_NUESTRO_BANCO_PROMPT + """"
*   (Si la metodología actual del BANCO EXTERNO, tal como se describe en el contexto, presenta prácticas interesantes, diferentes, o más avanzadas que podrían ser oportunidades para """ + NOMBRE_NUESTRO_BANCO_PROMPT + """", destácalo aquí. Aplica la Prioridad 1 y 2 para explicar los conceptos si es necesario para entender la oportunidad y su relevancia para """ + NOMBRE_NUESTRO_BANCO_PROMPT + """.)
*   (Ejemplo: "La práctica de {nombre_banco_externo_prompt} de recalibrar sus modelos de PD trimestralmente usando datos recientes (pág. A) podría ser una oportunidad para """ + NOMBRE_NUESTRO_BANCO_PROMPT + """ si nuestra frecuencia es menor, ya que permitiría una adaptación más rápida a cambios en el entorno.")

*(Si después de un análisis riguroso NO encuentras información relevante para "{nombre_parametro}" en el CONTEXTO de '{nombre_banco_externo_prompt}', responde ÚNICAMENTE:
"ANÁLISIS FINAL: No se encontró información específica sobre metodología o cambios para '{nombre_parametro}' en el contexto proporcionado de {nombre_banco_externo_prompt}.")*
"""

TEMPLATE_INFORME_BENCHMARK_COMPETIDOR_V2 = """
Actúa como un Director de Estrategia de Riesgos presentando un informe de benchmarking DETALLADO Y ACCIONABLE al Comité de Riesgos de '{nombre_nuestro_banco_prompt}'.
Este informe se centra en el BANCO COMPETIDOR: '{nombre_banco_externo_prompt}' (archivo: {pdf_banco_externo}).
Se te proporciona un ANÁLISIS DETALLADO POR PARÁMETRO (generado previamente a partir del PDF del competidor y tu conocimiento general) sobre sus metodologías de riesgo y cambios identificados. La metodología de '{nombre_nuestro_banco_prompt}' (archivo: {pdf_nuestro_banco}) es para referencia contextual.
El **FOCO PRINCIPAL ES EL BANCO COMPETIDOR Y LAS OPORTUNIDADES CONCRETAS, COMPRENSIBLES Y ACCIONABLES PARA '{nombre_nuestro_banco_prompt}'**.

ANÁLISIS DETALLADO POR PARÁMETRO (Foco en {nombre_banco_externo_prompt} - generado en paso previo):
---
{contexto_completo_analisis} 
---

TU TAREA: Generar un informe de benchmarking en formato MARKDOWN. Sé preciso, estratégico y utiliza una estructura clara. **Asegúrate de que las oportunidades y recomendaciones sean claras, expliquen conceptos clave (utilizando las explicaciones del análisis previo que se te proporciona), y sean ACCIONABLES para '{nombre_nuestro_banco_prompt}'. Evita recomendaciones genéricas como "crear equipos multidisciplinarios" o "solicitar más información al competidor". Enfócate en análisis internos, pilotos o exploraciones que '{nombre_nuestro_banco_prompt}' PUEDA realizar.**

---
## Análisis Comparativo: {nombre_nuestro_banco_prompt} vs. {nombre_banco_externo_prompt}
*(Basado en el archivo del competidor: {pdf_banco_externo} y el análisis previo de sus prácticas)*

### 1. Resumen Ejecutivo (vs. {nombre_banco_externo_prompt})
*   **Visión General del Competidor:** (2-3 frases sobre el perfil de {nombre_banco_externo_prompt} si es relevante a partir del análisis detallado por parámetro).
*   **Hallazgos Clave del Análisis de '{nombre_banco_externo_prompt}':** (3-4 viñetas con los descubrimientos más impactantes o diferenciadores de {nombre_banco_externo_prompt}. **Si un hallazgo implica un concepto técnico, asegúrate de que se entienda qué es, utilizando la explicación del análisis previo.**)
    *   (Ej. Hallazgo: "{nombre_banco_externo_prompt} utiliza Flujos de Caja Descontados (FCD) para la LGD de su cartera corporativa (pág. X). Como se explicó, los FCD implican [breve recordatorio de la explicación de FCD del análisis previo], lo que les permite una valoración económica de la recuperación.")
    *   (Ej. Hallazgo: "{nombre_banco_externo_prompt} reporta una segmentación de EAD para productos rotativos usando Factores de Conversión Crediticia (CCF) específicos por [detalle si lo hubo] (pág. Y). Los CCF, como se describió, estiman la porción de una línea no dispuesta que se usará al default.")
*   **Principal Propuesta de Valor para {nombre_nuestro_banco_prompt} (derivada de este competidor):** (1-2 frases sobre la oportunidad más grande y accionable que este análisis específico revela para {nombre_nuestro_banco_prompt}).

### 2. Prácticas y Cambios Metodológicos Destacados en '{nombre_banco_externo_prompt}' Relevantes para '{nombre_nuestro_banco_prompt}'
*(Para cada parámetro principal, si hay información relevante en el análisis detallado del competidor):*

#### 2.1. Parámetro: [Nombre del Parámetro]
*   **Metodología/Práctica Destacada de '{nombre_banco_externo_prompt}':** 
    *   (Describe concisamente lo que hace el competidor, ej: "Utilizan un modelo 'Closed Recoveries' para LGD en hipotecarios con un horizonte de X meses (pág. A).").
    *   (Ej: "Mencionan una segmentación de PD para minoristas que considera [X factor si se detalló] (pág. B).")
*   **Explicación Adicional del Concepto (basada en el análisis previo, si es necesaria para la comprensión):**
    *   (Ej: "Como se detalló en el análisis previo, los 'Flujos de Caja Descontados (FCD)' utilizados por {nombre_banco_externo_prompt} para [cartera] implican proyectar flujos de efectivo futuros de un activo y traerlos a valor presente. Esto ofrece una estimación económica de la recuperación y puede ser más sensible a las características específicas del activo y del entorno que una tasa de recuperación histórica simple.")
    *   (Ej: "Los 'Factores de Conversión Crediticia (CCF)' que {nombre_banco_externo_prompt} segmenta para [producto] (pág. B) son porcentajes que estiman qué porción de una exposición no dispuesta se convertirá en una exposición real al momento del default. Una segmentación más fina de CCFs puede llevar a una EAD más precisa.")
*   **Diferencia Clave vs. Práctica Estándar o Posible Práctica Actual de '{nombre_nuestro_banco_prompt}' (si se infiere):** 
    *   (Ej: "Esto podría diferir de nuestro enfoque actual en {nombre_nuestro_banco_prompt} que podría no segmentar CCFs con tanto detalle o no utilizar FCD para esta cartera específica.")
*   **Potencial Relevancia/Aprendizaje para '{nombre_nuestro_banco_prompt}':** 
    *   (Ej: "La aplicación de FCD podría inspirar una evaluación interna en {nombre_nuestro_banco_prompt} para mejorar la precisión de nuestra LGD para carteras similares." o "Analizar la segmentación de CCF de {nombre_banco_externo_prompt} podría llevarnos a revisar nuestros propios CCF para identificar oportunidades de mayor granularidad en {nombre_nuestro_banco_prompt}.")

### 3. Oportunidades de Mejora Detalladas para '{nombre_nuestro_banco_prompt}' (Inspiradas por '{nombre_banco_externo_prompt}')
*(Identifica 2-3 áreas o prácticas específicas. **Asegúrate de que la oportunidad sea comprensible, explicando el 'qué' y el 'cómo' tanto como sea posible basado en el análisis previo. Las sugerencias deben ser ACCIONABLES para {nombre_nuestro_banco_prompt}.**)*

#### 3.1. Oportunidad: [Nombre Corto de la Oportunidad, ej: "Evaluar Implementación de Flujos de Caja Descontados (FCD) para LGD en Cartera Corporativa No Retail de {nombre_nuestro_banco_prompt}"]
*   **Observado en '{nombre_banco_externo_prompt}':** (Detalle de la práctica del competidor, ej: "{nombre_banco_externo_prompt} utiliza FCD para estimar el monto recuperable en activos no financieros de su cartera corporativa (pág. X).")
*   **Explicación del Concepto (basada en el análisis previo):** (Ej: "Los Flujos de Caja Descontados (FCD), como se explicó anteriormente, consisten en proyectar los flujos de efectivo futuros que se espera recuperar de un préstamo o activo en default (ej. pagos parciales, venta de colateral), y luego descontar esos flujos a su valor presente utilizando una tasa que refleje el riesgo y el valor del dinero en el tiempo. Esto proporciona una valoración económica de la recuperación esperada, potencialmente más precisa que métodos basados solo en tasas históricas promedio, especialmente para exposiciones complejas o con colaterales específicos.")
*   **Beneficio Potencial para '{nombre_nuestro_banco_prompt}':** (Ej: "Mayor precisión en la estimación de LGD para la cartera corporativa no retail, provisiones más ajustadas a la realidad económica de las recuperaciones, posible optimización de capital al reflejar mejor el valor de las garantías y estrategias de recuperación individualizadas.")
*   **Consideraciones para Implementación en '{nombre_nuestro_banco_prompt}':** (Ej: "Requiere capacidad interna para proyectar flujos de recuperación (ej. cronogramas de pago esperados, valor de realización de colaterales por tipo), definir tasas de descuento apropiadas por segmento o riesgo, y posiblemente adaptar sistemas para los cálculos y el almacenamiento de datos de estas proyecciones.")
*   **Sugerencia de Próximo Paso para '{nombre_nuestro_banco_prompt}':** (Ej: "**Iniciar un proyecto piloto (Proof of Concept) en {nombre_nuestro_banco_prompt}** para aplicar FCD en una sub-cartera seleccionada de préstamos corporativos deteriorados (ej. Stage 3 con garantías reales) para evaluar su impacto, complejidad y viabilidad operativa comparado con el método actual.")

#### 3.2. Oportunidad: [Nombre Corto, ej: "Analizar Viabilidad de Segmentación Avanzada de CCF para Líneas de Crédito Rotativas en {nombre_nuestro_banco_prompt}"]
*   **Observado en '{nombre_banco_externo_prompt}':** (Ej: "{nombre_banco_externo_prompt} segmenta EAD para productos rotativos utilizando Factores de Conversión de Crédito (CCF) diferenciados por [ej. tipo de producto, comportamiento del cliente si se mencionó en el análisis previo] (pág. Y).")
*   **Explicación del Concepto (basada en el análisis previo):** (Ej: "Un Factor de Conversión de Crédito (CCF), como se explicó, es un porcentaje que estima qué porción de una exposición no dispuesta (ej. una línea de crédito no utilizada o una garantía contingente) se espera que sea dispuesta o utilizada por el cliente al momento del default. Aplicar CCFs diferenciados por tipo de producto rotativo, segmento de cliente (ej. PYME vs. Corporativo), o incluso comportamiento de uso histórico de la línea, puede hacer la estimación de EAD más precisa que un CCF único o genérico para todas las exposiciones fuera de balance.")
*   **Beneficio Potencial para '{nombre_nuestro_banco_prompt}':** (Ej: "Estimaciones de EAD más precisas para la cartera de líneas rotativas de {nombre_nuestro_banco_prompt}, lo que impacta directamente en el cálculo de ECL y en los requerimientos de capital. Puede llevar a una mejor tarificación del riesgo y gestión de límites para estos productos.")
*   **Consideraciones para Implementación en '{nombre_nuestro_banco_prompt}':** (Ej: "Requiere un análisis de datos históricos de disposición de líneas de crédito y otros productos fuera de balance en {nombre_nuestro_banco_prompt} para identificar patrones y segmentos significativos. Desarrollo y validación de modelos de CCF segmentados, asegurando la disponibilidad y calidad de los datos necesarios.")
*   **Sugerencia de Próximo Paso para '{nombre_nuestro_banco_prompt}':** (Ej: "**Realizar un análisis exploratorio de los datos históricos de disposición de líneas de crédito rotativas en {nombre_nuestro_banco_prompt}** para determinar si se pueden derivar CCFs empíricos más granulares (ej. por producto específico, segmento de cliente, nivel de utilización previo al estrés) y evaluar su potencial impacto en la EAD frente a los CCFs actuales.")

#### 3.3. Oportunidad: [Nombre Corto, ej: "Explorar Segmentación Adicional de PD para Cartera Minorista en {nombre_nuestro_banco_prompt}"]
*   **Observado en '{nombre_banco_externo_prompt}':** (Ej: "{nombre_banco_externo_prompt} mencionó una 'segmentación granular' para PD en su cartera minorista (pág. X), aunque los factores exactos no fueron detallados, sugiere un enfoque más allá de la segmentación básica.")
*   **Explicación del Concepto (basada en el análisis previo):** (Ej: "Como se comentó, la segmentación granular de PD implica dividir la cartera en subgrupos más pequeños y homogéneos. Para la cartera minorista, esto podría involucrar combinar el score de originación con variables de comportamiento (uso de tarjeta de crédito, saldos en cuenta, frecuencia de transacciones), datos demográficos refinados, o incluso el canal de adquisición del cliente, para capturar matices de riesgo que un modelo de PD más agregado podría pasar por alto.")
*   **Beneficio Potencial para '{nombre_nuestro_banco_prompt}':** (Ej: "Mayor poder predictivo de los modelos de PD para la cartera minorista de {nombre_nuestro_banco_prompt}, permitiendo una mejor toma de decisiones en originación, una fijación de precios más ajustada al riesgo, y una gestión de portafolio más proactiva.")
*   **Consideraciones para Implementación en '{nombre_nuestro_banco_prompt}':** (Ej: "Disponibilidad y calidad de datos adicionales para segmentación. Capacidad de modelización para desarrollar y validar estos modelos más granulares. Potencial incremento en la complejidad del modelo y su monitoreo.")
*   **Sugerencia de Próximo Paso para '{nombre_nuestro_banco_prompt}':** (Ej: "**El equipo de Modelización de {nombre_nuestro_banco_prompt} podría iniciar un proyecto de investigación para identificar y probar nuevas variables de segmentación para la PD de la cartera minorista**, utilizando técnicas de feature engineering y evaluando el uplift predictivo en backtesting contra los modelos actuales.")

### 4. Recomendaciones Estratégicas Específicas para '{nombre_nuestro_banco_prompt}' (Basadas en el análisis de '{nombre_banco_externo_prompt}')
*(Top 1-2 recomendaciones más accionables y específicas para '{nombre_nuestro_banco_prompt}'. **Enfócate en acciones de exploración, análisis interno o pilotos.**)*

#### Recomendación #1 (Inspirada por {nombre_banco_externo_prompt}): [Acción Concreta y Específica, ej: "Priorizar Investigación Interna sobre Aplicabilidad de FCD para LGD en Segmentos Clave de {nombre_nuestro_banco_prompt}"]
*   **Descripción:** (Ej: "Dado que {nombre_banco_externo_prompt} (y potencialmente otros bancos avanzados) utiliza Flujos de Caja Descontados (FCD) para LGD en carteras específicas (ej. corporativa, hipotecaria), **se recomienda que {nombre_nuestro_banco_prompt} priorice una investigación interna y un posible piloto para evaluar la aplicabilidad y beneficios de FCD para la estimación de LGD en segmentos clave de su propia cartera**, como [mencionar 1-2 carteras de {nombre_nuestro_banco_prompt} donde podría tener más impacto, ej. 'grandes empresas con garantías complejas' o 'créditos hipotecarios en recuperación avanzada'].")
*   **Justificación del Valor para '{nombre_nuestro_banco_prompt}':** (Ej: "La adopción de FCD podría mejorar significativamente la precisión de las estimaciones de LGD en {nombre_nuestro_banco_prompt} para esas carteras, llevando a provisiones más realistas, una mejor comprensión del valor recuperable de los colaterales y, potencialmente, una optimización de los activos ponderados por riesgo.")
*   **Próximos Pasos Sugeridos para '{nombre_nuestro_banco_prompt}':** (Ej: "1. **El equipo de Metodologías de Riesgo de {nombre_nuestro_banco_prompt} define el alcance del piloto**: seleccionar una subcartera y los tipos de datos de recuperación a recolectar/proyectar. 2. **Desarrollar un modelo FCD simplificado para el piloto y compararlo con las estimaciones LGD actuales** para esa subcartera. 3. **Evaluar la complejidad de datos y sistemas** requerida para una implementación más amplia.")

#### Recomendación #2 (Inspirada por {nombre_banco_externo_prompt}): [Acción Concreta y Específica, ej: "Realizar Estudio de Segmentación de CCF para Productos Rotativos de {nombre_nuestro_banco_prompt}"]
*   **Descripción:** (Ej: "Inspirado por la práctica de {nombre_banco_externo_prompt} de segmentar Factores de Conversión Crediticia (CCF) para su EAD en productos rotativos, **se recomienda que {nombre_nuestro_banco_prompt} realice un estudio interno para analizar la viabilidad y el impacto de una segmentación más granular de sus propios CCFs** para productos clave como tarjetas de crédito y líneas de crédito comerciales/PYME.")
*   **Justificación del Valor para '{nombre_nuestro_banco_prompt}':** (Ej: "Una segmentación de CCF basada en datos empíricos de {nombre_nuestro_banco_prompt} (ej. por tipo de producto, segmento de cliente, nivel de uso histórico) puede resultar en estimaciones de EAD más precisas, lo que afinaría el cálculo de ECL y los APR, y podría revelar diferencias de riesgo no capturadas por CCFs genéricos.")
*   **Próximos Pasos Sugeridos para '{nombre_nuestro_banco_prompt}':** (Ej: "1. **El equipo de Datos y Modelización de {nombre_nuestro_banco_prompt} recopila y analiza datos históricos de disposición** de líneas de crédito no utilizadas. 2. **Probar diferentes criterios de segmentación para CCFs y estimar CCFs empíricos** para dichos segmentos. 3. **Evaluar el impacto cuantitativo en la EAD y ECL agregada** de aplicar estos CCFs segmentados versus el enfoque actual.")
---
"""

TEMPLATE_CONCLUSION_GLOBAL_BENCHMARK = """
Actúa como el Director de Estrategia de Riesgos de '{nombre_nuestro_banco_prompt}', presentando las conclusiones finales y recomendaciones estratégicas consolidadas de un ejercicio de benchmarking exhaustivo.
Se han analizado múltiples bancos competidores ({lista_nombres_competidores}) en comparación con las prácticas de '{nombre_nuestro_banco_prompt}'.
Los análisis individuales de cada competidor (que has procesado previamente, incluyendo explicaciones de conceptos clave) son la base para esta conclusión global.

TU TAREA:
Con base en TODOS los informes de benchmarking individuales que has generado, elabora una sección final para el informe consolidado. Esta sección debe ser **clara, accionable y explicar conceptos clave cuando sea necesario para la comprensión de las recomendaciones por parte del Comité de Riesgos de '{nombre_nuestro_banco_prompt}'. Evita recomendaciones genéricas. Enfócate en análisis internos, pilotos o desarrollo de capacidades que '{nombre_nuestro_banco_prompt}' pueda emprender.**

1.  **CONCLUSIÓN GENERAL DEL EJERCICIO DE BENCHMARKING:**
    *   Resume los **aprendizajes más significativos y transversales** obtenidos del análisis de todos los competidores, siempre desde la perspectiva de '{nombre_nuestro_banco_prompt}'.
    *   Identifica **temas o tendencias recurrentes** observadas en las prácticas de los competidores que sean relevantes para '{nombre_nuestro_banco_prompt}' (ej. "mayor granularidad en la segmentación de LGD y EAD con técnicas como FCD o CCFs segmentados", "uso de múltiples escenarios y variables macroeconómicas específicas en FLI", "exploración incipiente o consolidada de datos alternativos para mejorar PD", "sistemas de monitoreo y validación de modelos más dinámicos y automatizados").
    *   Evalúa brevemente la **posición competitiva general de '{nombre_nuestro_banco_prompt}'** en términos de metodologías de riesgo IFRS 9 a la luz de este benchmark (ej. áreas de fortaleza, áreas con oportunidad de alineación con mejores prácticas observadas).

2.  **RECOMENDACIONES ESTRATÉGICAS AGREGADAS Y PRIORIZADAS PARA '{nombre_nuestro_banco_prompt}':**
    *   Identifica las **3 a 4 recomendaciones estratégicas MÁS IMPACTANTES Y PRIORITARIAS** para '{nombre_nuestro_banco_prompt}' que surgen del conjunto de todos los análisis individuales.
    *   Para cada recomendación priorizada:
        *   **Recomendación Consolidada #[Número]:** (Descripción clara y concisa de la acción específica para '{nombre_nuestro_banco_prompt}').
        *   **Justificación del Valor Estratégico para '{nombre_nuestro_banco_prompt}':** (¿Por qué esta recomendación es crucial? Impacto esperado en precisión, eficiencia, capital, cumplimiento, etc.).
        *   **Prácticas Inspiradoras de Competidores (Ejemplos Clave y Explicación del Concepto si es necesario):** 
            *   (Menciona qué competidores, ej. 'Según lo visto en Competidor A (que usa FCD para LGD) y Competidor C (que segmenta CCFs para EAD)', inspiran o respaldan esta recomendación).
            *   **Si la práctica del competidor fue mencionada pero no detallada en sus PDFs (y por ende en los informes individuales), RECUERDA la explicación general que ya se debió haber dado sobre dicha práctica en la industria y por qué es valiosa.**
            *   (Ejemplo para "sistemas de monitoreo proactivos" si no hubo detalle en PDFs: "Varios competidores como Banco X e Y mencionaron 'sistemas de monitoreo proactivos'. Aunque los detalles específicos de su implementación no fueron provistos, en la industria esto típicamente implica el seguimiento continuo del desempeño de los modelos (ej. PD, LGD) contra los resultados reales, el uso de dashboards automatizados, la definición de umbrales de alerta temprana para desviaciones significativas, y procesos de re-calibración o re-validación periódica más frecuentes que el mínimo regulatorio. El valor para {nombre_nuestro_banco_prompt} reside en asegurar la relevancia y precisión de nuestros modelos en un entorno cambiante y detectar deterioros predictivos tempranamente.")
            *   (Ejemplo para "datos alternativos" si no hubo detalle en PDFs: "Competidores como Banco Z parecen explorar 'datos alternativos' para sus modelos de riesgo. En la industria bancaria, 'datos alternativos' se refiere a información no tradicionalmente usada en el scoring crediticio, como por ejemplo: datos transaccionales muy detallados de cuentas corrientes o tarjetas (con consentimiento), información de comportamiento digital en plataformas del banco, datos de uso de servicios de telecomunicaciones, o incluso información de redes sociales o e-commerce (siempre con un marco ético y de privacidad robusto). Su valor potencial para {nombre_nuestro_banco_prompt} radica en mejorar el poder predictivo de los modelos de riesgo, especialmente para segmentos con historial crediticio escaso ('thin file') o para capturar cambios recientes en el comportamiento del cliente no reflejados en datos tradicionales de bureaus.")
        *   **Próximos Pasos Sugeridos de Alto Nivel para '{nombre_nuestro_banco_prompt}' (Accionables y Realistas):** (Ej: "Realizar un análisis de viabilidad interna en {nombre_nuestro_banco_prompt} para la aplicación de [técnica específica] en la cartera [cartera específica de Credicorp].", "Iniciar un proyecto piloto (PoC) en {nombre_nuestro_banco_prompt} enfocado en [aspecto X, ej. segmentación de CCF para tarjetas de crédito] para evaluar su aplicabilidad y beneficios.", "Capacitar al equipo de modelización de {nombre_nuestro_banco_prompt} en [nueva técnica o enfoque, ej. modelado de FCD].", "El equipo de Datos de {nombre_nuestro_banco_prompt} debe investigar fuentes de datos internas (ej. datos transaccionales detallados) que podrían ser explotadas para enriquecer los modelos de PD para [segmento específico].")

3.  **VISIÓN A FUTURO PARA LA FUNCIÓN DE BENCHMARKING EN '{nombre_nuestro_banco_prompt}':**
    *   Breves reflexiones sobre cómo este ejercicio de benchmarking puede institucionalizarse (ej. frecuencia anual, enfoque en parámetros específicos rotativamente) o qué áreas podrían explorarse en futuros análisis para mantener a '{nombre_nuestro_banco_prompt}' a la vanguardia.

Utiliza formato MARKDOWN para una presentación clara y profesional, con encabezados (`##`, `###`) y viñetas.
El enfoque debe ser 100% en cómo '{nombre_nuestro_banco_prompt}' puede mejorar y fortalecerse de manera práctica.
"""

# --- FUNCIONES ---
def clean_camelot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df_cleaned = df.dropna(axis=0, how='all')
    if df_cleaned.empty: return df_cleaned
    df_cleaned = df_cleaned.dropna(axis=1, how='all')
    if df_cleaned.empty: return df_cleaned
    if not df_cleaned.empty:
        for col in df_cleaned.columns:
            try:
                if df_cleaned[col].dtype == 'object' or pd.api.types.is_string_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
            except Exception as e:
                logging.warning(f"No se pudo limpiar la columna '{col}'. Error: {e}")
                try: df_cleaned[col] = df_cleaned[col].astype(str)
                except Exception as e_conv: logging.error(f"Error fatal al convertir '{col}' a str: {e_conv}")
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def extract_and_format_tables_from_pdf(pdf_path: str, pages: str = 'all') -> list[Document]:
    table_docs = []
    short_pdf_name = os.path.basename(pdf_path)
    flavors_to_try = ['lattice', 'stream']
    tables_found_with_first_flavor = False
    # Suprimir warnings de Camelot durante la extracción
    camelot_logger = logging.getLogger('camelot')
    original_level = camelot_logger.level
    camelot_logger.setLevel(logging.CRITICAL) # O logging.ERROR para ver solo errores de Camelot

    for flavor in flavors_to_try:
        if flavor == 'stream' and tables_found_with_first_flavor:
            logging.info(f"Tablas encontradas con 'lattice' para {short_pdf_name}. Omitiendo 'stream'.")
            break
        logging.info(f"Extrayendo tablas de {short_pdf_name} (págs: {pages}, flavor: '{flavor}')...")
        try:
            camelot_kwargs = {'pages': pages, 'flavor': flavor}
            if flavor == 'lattice': camelot_kwargs['line_scale'] = 40
            elif flavor == 'stream': camelot_kwargs['edge_tol'] = 100; camelot_kwargs['row_tol'] = 5
            tables = camelot.read_pdf(pdf_path, **camelot_kwargs)
            if tables.n > 0:
                if flavor == 'lattice': tables_found_with_first_flavor = True
                logging.info(f"Camelot ('{flavor}') encontró {tables.n} tabla(s) en {short_pdf_name}.")
                for i, table_report in enumerate(tables):
                    try:
                        df = table_report.df
                        cleaned_df = clean_camelot_dataframe(df.copy())
                        if not cleaned_df.empty:
                            markdown_table = cleaned_df.to_markdown(index=False, tablefmt="pipe")
                            prefix = (f"Contexto: Tabla de '{short_pdf_name}', pág {table_report.page} (Camelot '{flavor}', precisión: {table_report.parsing_report.get('accuracy', 'N/A')}).")
                            content = f"{prefix}\n\n[INICIO DE TABLA Markdown]\n{markdown_table}\n[FIN DE TABLA Markdown]"
                            metadata = {"source": short_pdf_name, "page": table_report.page, "is_table": True,
                                        "table_id": f"{short_pdf_name}_t{i+1}_p{table_report.page}_{flavor}",
                                        "extraction_method": f"camelot_{flavor}"}
                            table_docs.append(Document(page_content=content, metadata=metadata))
                        else: logging.info(f"Tabla {i+1} (pág {table_report.page}) en {short_pdf_name} ('{flavor}') vacía tras limpiar; omitida.")
                    except Exception as e_table: logging.error(f"Error procesando tabla {i+1}, pág {table_report.page}, {short_pdf_name}: {e_table}")
            elif flavor == 'lattice': logging.info(f"Camelot 'lattice' no encontró tablas en {short_pdf_name}. Intentando 'stream'.")
            elif flavor == 'stream': logging.info(f"Camelot 'stream' tampoco encontró tablas en {short_pdf_name}.")
        except Exception as e:
            logging.error(f"Error crítico con Camelot para {short_pdf_name} ('{flavor}'): {e}")
            if "ghostscript" in str(e).lower(): logging.error("¡ERROR GHOSTSCRIPT DETECTADO! Asegúrate de que Ghostscript esté instalado y en el PATH del sistema."); break
    
    camelot_logger.setLevel(original_level) # Restaurar nivel de logging
    if not table_docs and not tables_found_with_first_flavor: logging.info(f"No se extrajeron tablas de {short_pdf_name} con Camelot.")
    return table_docs

def normalize(text: str) -> str:
    text = text.lower(); return " ".join(text.split())

def preprocess_documents(docs: list[Document]) -> list[Document]:
    processed_docs = []
    for doc_idx, doc in enumerate(docs):
        try:
            content = doc.page_content
            if not doc.metadata.get("is_table", False): content = normalize(content)
            if content: processed_docs.append(Document(page_content=content, metadata=doc.metadata))
            else: logging.warning(f"Contenido vacío en {doc.metadata.get('source','N/A')}, pág {doc.metadata.get('page','N/A')}. Omitido.")
        except Exception as e: logging.warning(f"Error procesando {doc.metadata.get('source','N/A')}, pág {doc.metadata.get('page','N/A')}: {e}")
    return processed_docs

def chunk_documents(documents: list[Document], chunk_size: int = 1800, chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""], length_function=len)
    logging.info(f"Dividiendo {len(documents)} docs en chunks (tamaño={chunk_size}, solapamiento={chunk_overlap})...")
    return splitter.split_documents(documents)

class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try: self.model = SentenceTransformer(model_name); logging.info(f"Embeddings '{model_name}' cargado.")
        except Exception as e: logging.error(f"Error cargando SentenceTransformer '{model_name}': {e}"); raise
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode([t.replace("\n", " ") for t in texts], convert_to_numpy=True).tolist()
    def embed_query(self, text: str) -> list[float]:
        encoded = self.model.encode(text.replace("\n", " "), convert_to_numpy=True)
        return encoded.flatten().tolist()

def build_or_load_faiss_index(documents: list[Document] = None, embeddings: Embeddings = None, persist_path: str = "faiss_index_local") -> FAISS:
    if os.path.exists(persist_path) and os.listdir(persist_path) and not REBUILD_FAISS_INDEX:
        logging.info(f"Cargando índice FAISS desde {persist_path}")
        if not embeddings: raise ValueError("Embeddings requeridos para cargar índice.")
        try: return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error(f"Error cargando índice FAISS ({e}). Se reconstruirá si se proporcionan documentos y REBUILD_FAISS_INDEX es True.")
            # No forzar REBUILD_FAISS_INDEX = True aquí, la lógica principal lo decidirá.
            # Simplemente devolvemos None para indicar que la carga falló.
            return None 
    
    if REBUILD_FAISS_INDEX and documents and embeddings:
        logging.info(f"Construyendo nuevo índice FAISS en {persist_path}")
        os.makedirs(persist_path, exist_ok=True)
        if os.path.exists(persist_path) and os.listdir(persist_path):
            logging.info(f"Limpiando directorio de índice existente: {persist_path}")
            for f_item in os.listdir(persist_path):
                item_path = os.path.join(persist_path, f_item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e_del:
                    logging.error(f'Failed to delete {item_path}. Reason: {e_del}')
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(persist_path)
        logging.info(f"Nuevo índice FAISS construido y guardado en {persist_path}.")
        return vectorstore
    
    # Si REBUILD_FAISS_INDEX es False y el índice no existe o no se pudo cargar
    if not (os.path.exists(persist_path) and os.listdir(persist_path)):
        if REBUILD_FAISS_INDEX: # Pero no se proporcionaron documentos/embeddings
             raise FileNotFoundError(f"REBUILD_FAISS_INDEX es True pero no se proporcionaron documentos o embeddings para construir el índice en {persist_path}.")
        else: # No existe y REBUILD_FAISS_INDEX es False
            raise FileNotFoundError(f"Índice FAISS no encontrado en {persist_path} y REBUILD_FAISS_INDEX es False.")

    # Si el índice existe, REBUILD_FAISS_INDEX es False, pero la carga inicial falló y no se proporcionaron documentos para reconstruir
    if not documents or not embeddings:
        raise RuntimeError(f"Índice FAISS en {persist_path} no pudo ser cargado y no se proporcionaron documentos/embeddings para reconstrucción (REBUILD_FAISS_INDEX={REBUILD_FAISS_INDEX}).")

    return None # Caso por defecto si no se entra en ninguna lógica de construcción/carga exitosa

def semantic_search_filtered(query: str, vectorstore: FAISS, k: int = 10, source_filename: str = None) -> list[Document]:
    logging.info(f"Búsqueda semántica: '{query}' (k={k}, filtro='{source_filename or 'Ninguno'}')")
    if not (vectorstore and hasattr(vectorstore, 'index') and vectorstore.index and vectorstore.index.ntotal > 0):
        logging.error(f"Índice FAISS no disponible o vacío para la query: '{query}'."); return []
    try:
        k_fetch = min(k * 2, vectorstore.index.ntotal) 
        if k_fetch == 0 and k > 0 : k_fetch = min(k, vectorstore.index.ntotal)
        if k_fetch == 0: logging.warning(f"k_fetch es 0 para la query '{query}'. No se buscará."); return []
        
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k_fetch)

        if not results_with_scores: logging.warning(f"similarity_search_with_relevance_scores no devolvió resultados para '{query}'."); return []
        
        final_results = []
        for doc, score in results_with_scores:
            if source_filename:
                if doc.metadata.get("source") == source_filename:
                    final_results.append(doc)
            else:
                final_results.append(doc)
            if len(final_results) >= k:
                break
        logging.info(f"Se obtuvieron {len(final_results)} resultados (de k={k} solicitados) para '{query}' con filtro '{source_filename or 'todos'}'.")
        return final_results
    except Exception as e: logging.error(f"Excepción en búsqueda semántica para '{query}': {e}", exc_info=True); return []

def initialize_llm(provider: str = "google", model_name: str = None, temperature: float = 0.15):
    if provider.lower() == "google":
        model_to_use = model_name if model_name else "gemini-1.5-flash-latest"
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.critical("CRÍTICO: GOOGLE_API_KEY no encontrada en .env o en el entorno.")
            raise ValueError("GOOGLE_API_KEY no encontrada en .env.")
        try:
            safety_settings_corrected = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            llm = ChatGoogleGenerativeAI(
                model=model_to_use,
                google_api_key=api_key,
                temperature=temperature,
                safety_settings=safety_settings_corrected,
                request_timeout=300 # Timeout aumentado a 5 minutos
            )
            logging.info(f"LLM Google Gemini ('{model_to_use}', temp={temperature}) inicializado.")
            return llm
        except Exception as e:
            logging.error(f"Error inicializando Google Gemini: {e}", exc_info=True); raise
    else:
        raise ValueError(f"Proveedor de LLM no soportado: {provider}.")

# Usando PromptTemplate de langchain.prompts para compatibilidad con la v11 que funcionó
PROMPT_DESCRIBIR_METODOLOGIA_NB = PromptTemplate(
    input_variables=["context", "nombre_parametro", "aspectos_relevantes_parametro"],
    template=TEMPLATE_DESCRIBIR_METODOLOGIA_NUESTRO_BANCO_POR_PARAMETRO
)
PROMPT_EXTRACCION_CAMBIOS_BE = PromptTemplate(
    input_variables=["context", "nombre_parametro", "aspectos_a_buscar_en_cambios", "nombre_banco_externo_prompt"],
    template=TEMPLATE_EXTRACCION_CAMBIOS_BANCO_EXTERNO_POR_PARAMETRO
)
PROMPT_INFORME_COMPETIDOR = PromptTemplate(
    input_variables=["contexto_completo_analisis", "pdf_nuestro_banco", "pdf_banco_externo", "nombre_nuestro_banco_prompt", "nombre_banco_externo_prompt"],
    template=TEMPLATE_INFORME_BENCHMARK_COMPETIDOR_V2
)
PROMPT_CONCLUSION_GLOBAL = PromptTemplate(
    input_variables=["textos_informes_individuales_concatenados_placeholder", "nombre_nuestro_banco_prompt", "lista_nombres_competidores"],
    template=TEMPLATE_CONCLUSION_GLOBAL_BENCHMARK
)

def run_llm_chain(llm, prompt_template: PromptTemplate, inputs: dict, task_description: str) -> str:
    context_keys = ["context", "contexto_completo_analisis", "textos_informes_individuales_concatenados_placeholder"]
    relevant_context_present = False
    for key in context_keys:
        if key in inputs and isinstance(inputs[key], str) and inputs[key].strip(): # Asegurarse que es string
            relevant_context_present = True
            break
    
    if task_description.startswith("analizar") and "context" in inputs and (not isinstance(inputs["context"], str) or not inputs["context"].strip()):
        logging.warning(f"Contexto vacío o no string para {task_description} (análisis de banco externo). El LLM será llamado para que indique 'no información'.")
        inputs["context"] = inputs.get("context", "") # Asegurar que "context" existe, aunque sea vacío
    elif not relevant_context_present and not task_description.startswith("generar conclusión global"):
        logging.warning(f"Contexto relevante vacío para {task_description}. No se llamará al LLM.")
        return f"No se proporcionó contexto válido para: {task_description}."

    # Verificar que todas las variables esperadas por el prompt están en inputs
    missing_vars = [var for var in prompt_template.input_variables if var not in inputs]
    if missing_vars:
        logging.error(f"Error para {task_description}: Faltan variables en input para PromptTemplate: {missing_vars}. Esperadas: {prompt_template.input_variables}. Recibidas: {list(inputs.keys())}")
        return f"Error de configuración de Prompt: Faltan variables {missing_vars} para la tarea '{task_description}'."

    chain = prompt_template | llm | StrOutputParser()
    approx_chars = 0
    for v_key, v_val in inputs.items():
        if isinstance(v_val, str):
            approx_chars += len(v_val)

    logging.info(f"Enviando al LLM para: {task_description} (aprox. {approx_chars} chars)...")
    try:
        response_content = chain.invoke(inputs)
        return response_content
    except KeyError as ke:
        logging.error(f"KeyError específico durante chain.invoke para {task_description}: {ke}", exc_info=True)
        key_arg = "No disponible"
        if ke.args: key_arg = ke.args[0]
        logging.error(f"Argumento del KeyError (clave problemática si está disponible): {key_arg}")
        # El mensaje de error de Langchain ya es bastante descriptivo
        return f"Error LLM (KeyError en invoke) para {task_description}. Detalle: {str(ke)}. Verifique las variables del prompt y las claves en `inputs`. Clave problemática podría ser: '{key_arg}'. Ver logs."
    except Exception as e:
        logging.error(f"Error en LLM para {task_description}. Tipo: {type(e).__name__}, Mensaje: {str(e)[:500]}", exc_info=True)
        error_str = str(e).lower()
        if "rate limit" in error_str or "quota" in error_str or "429" in error_str or "resource_exhausted" in error_str:
             return f"Error de API (Rate limit/Cuota) para {task_description}. Considera aumentar pausas o revisar cuotas de Gemini."
        if "context length" in error_str or "request payload" in error_str or "token" in error_str:
             return f"Error: Contexto demasiado largo para {task_description} (aprox. {approx_chars} chars). El modelo no puede procesar esta cantidad de información."
        if "candidate" in error_str and "blocked" in error_str:
             block_reason = "razón desconocida"
             if "safety" in error_str: block_reason = "SAFETY"
             if "recitation" in error_str: block_reason = "RECITATION"
             logging.warning(f"Respuesta bloqueada por Gemini ({block_reason}) para {task_description}. Contenido del error: {str(e)}")
             return f"Respuesta de Gemini bloqueada (Razón: {block_reason}) para {task_description}. Esto puede ocurrir si el contenido se parece a datos de entrenamiento o es percibido como no seguro. Revisa los logs."
        if "unknown field for part: thought" in error_str:
            return f"Error específico de Gemini (Unknown field for Part: thought) para {task_description}. Revisa versiones de librerías o la estructura del prompt/respuesta."
        return f"Error LLM para {task_description} (Tipo: {type(e).__name__}. Ver logs para detalle)."

# =========================================
# EJECUCIÓN PRINCIPAL
# =========================================
if __name__ == "__main__":
    if not os.path.isdir(FOLDER_INPUT_PDFS):
        logging.critical(f"La carpeta de PDFs de entrada '{FOLDER_INPUT_PDFS}' no existe. Saliendo.")
        exit(1)

    todos_los_pdfs_en_carpeta = [f for f in os.listdir(FOLDER_INPUT_PDFS) if f.lower().endswith('.pdf')]

    if PDF_NUESTRO_BANCO_FILENAME not in todos_los_pdfs_en_carpeta:
        logging.critical(f"El PDF de nuestro banco '{PDF_NUESTRO_BANCO_FILENAME}' no se encontró en '{FOLDER_INPUT_PDFS}'. Saliendo.")
        exit(1)

    LISTA_PDFS_COMPETIDORES = [pdf for pdf in todos_los_pdfs_en_carpeta if pdf != PDF_NUESTRO_BANCO_FILENAME]

    if not LISTA_PDFS_COMPETIDORES:
        logging.warning("No se encontraron PDFs de competidores en la carpeta. El análisis comparativo no se realizará.")
    else:
        logging.info(f"PDFs de competidores encontrados: {', '.join(LISTA_PDFS_COMPETIDORES)}")

    pdfs_a_indexar_y_validar = list(set([PDF_NUESTRO_BANCO_FILENAME] + LISTA_PDFS_COMPETIDORES))

    try: embedder = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception: logging.critical(f"Fallo inicializando embeddings. Saliendo."); exit(1)

    vector_store = None
    # Lógica de carga/reconstrucción del índice FAISS
    if not REBUILD_FAISS_INDEX and os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
        try:
            logging.info(f"Intentando cargar índice FAISS desde {FAISS_INDEX_PATH}...")
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True)
            if not (vector_store and hasattr(vector_store, 'index') and vector_store.index and vector_store.index.ntotal > 0):
                 logging.warning(f"Índice cargado desde {FAISS_INDEX_PATH} está vacío o inválido. Se marcará para reconstrucción.")
                 REBUILD_FAISS_INDEX = True 
                 vector_store = None 
            else:
                 logging.info(f"Índice FAISS cargado exitosamente desde {FAISS_INDEX_PATH} con {vector_store.index.ntotal} vectores.")
        except Exception as e:
            logging.warning(f"Error cargando índice FAISS existente desde {FAISS_INDEX_PATH} ({e}). Se marcará para reconstrucción.")
            REBUILD_FAISS_INDEX = True 
            vector_store = None 
    
    if REBUILD_FAISS_INDEX: # Si se necesita reconstruir (ya sea por la variable o porque la carga falló)
        logging.info(f"Iniciando (re)construcción del índice FAISS en {FAISS_INDEX_PATH}...")
        all_docs_for_processing = []
        if not pdfs_a_indexar_y_validar:
            logging.critical(f"No hay PDFs para indexar. No se puede construir el índice. Saliendo.")
            exit(1)

        for fname in pdfs_a_indexar_y_validar:
            full_path = os.path.join(FOLDER_INPUT_PDFS, fname)
            logging.info(f"\n--- Procesando para índice: {fname} ---")
            try:
                loader = PyPDFLoader(full_path)
                pages = loader.load()
                current_page_docs = [Document(page_content=p.page_content, metadata={"source": fname, "page": p.metadata.get("page", i) + 1, "is_table": False}) for i, p in enumerate(pages)]
                all_docs_for_processing.extend(current_page_docs)
                logging.info(f"Texto cargado de {fname} ({len(current_page_docs)} págs).")
                
                table_documents = extract_and_format_tables_from_pdf(full_path, pages="all")
                all_docs_for_processing.extend(table_documents)
                logging.info(f"{len(table_documents)} tablas procesadas de {fname}.")
            except Exception as e: logging.error(f"Error procesando PDF {full_path}: {e}", exc_info=True)

        if not all_docs_for_processing: logging.critical("No se cargaron documentos para el índice. Saliendo."); exit(1)
        
        clean_docs = preprocess_documents(all_docs_for_processing)
        if not clean_docs: logging.critical("No hay documentos después de la limpieza. Saliendo."); exit(1)
        
        chunked_docs = chunk_documents(clean_docs, chunk_size=1800, chunk_overlap=200) 
        if not chunked_docs: logging.critical("No se generaron chunks. Saliendo."); exit(1)
        
        try:
            # La función build_or_load_faiss_index ahora maneja la reconstrucción si REBUILD_FAISS_INDEX es True
            vector_store = build_or_load_faiss_index(documents=chunked_docs, embeddings=embedder, persist_path=FAISS_INDEX_PATH)
            if vector_store and hasattr(vector_store, 'index') and vector_store.index and vector_store.index.ntotal > 0:
                 logging.info(f"Índice FAISS (re)construido y disponible con {vector_store.index.ntotal} vectores.")
                 # No es necesario cambiar REBUILD_FAISS_INDEX aquí, se hará al inicio de la próxima ejecución si es False
            else:
                 logging.critical("Falló la construcción del índice FAISS o el índice está vacío post-construcción. Saliendo.")
                 exit(1)
        except Exception as e: logging.critical(f"Error crítico construyendo/guardando índice FAISS: {e}. Saliendo.", exc_info=True); exit(1)
    
    if not vector_store: # Si después de toda la lógica, vector_store sigue siendo None
        logging.critical(f"Vector Store no pudo ser cargado ni construido. Verifique la ruta del índice '{FAISS_INDEX_PATH}' y los PDFs. Saliendo.")
        exit(1)

    llm = None
    try:
        llm = initialize_llm(provider="google", model_name="gemini-1.5-flash-latest", temperature=0.15)
    except Exception as e:
        logging.error(f"Fallo CRÍTICO al inicializar LLM. Saliendo. Error: {e}")
        exit(1)

    lista_informes_individuales_md = []
    nombres_competidores_analizados_lista = [] 

    if llm and vector_store and hasattr(vector_store, 'index') and vector_store.index and vector_store.index.ntotal > 0:
      if not LISTA_PDFS_COMPETIDORES:
          logging.info("No hay PDFs de competidores para analizar. Finalizando el script.")
      else:
        for pdf_banco_externo_actual in LISTA_PDFS_COMPETIDORES:
            nombre_banco_externo_actual_prompt = os.path.splitext(pdf_banco_externo_actual)[0].replace("_", " ").replace("-", " ").title()
            nombres_competidores_analizados_lista.append(nombre_banco_externo_actual_prompt)

            print(f"\n\n=======================================================================")
            print(f"--- INICIANDO ANÁLISIS: {NOMBRE_NUESTRO_BANCO_PROMPT} vs. {nombre_banco_externo_actual_prompt} ---")
            print(f"--- (Archivo Externo: {pdf_banco_externo_actual}) ---")
            print("=======================================================================")

            resultados_por_parametro_lista_actual = []
            K_VALUE_SEARCH_NB = 7 
            K_VALUE_SEARCH_BE = 15

            for param_info in PARAMETROS_CLAVE:
                nombre_p = param_info["nombre_parametro"]
                # q_nb = param_info["query_rag_nuestro_banco_template"] # No se usa para generar el informe de nuestro banco por parámetro
                q_be = param_info["query_rag_banco_externo_template"].format(nombre_banco_externo=nombre_banco_externo_actual_prompt)
                aspectos_p = param_info["aspectos_parametro"]
                print(f"\n--- PROCESANDO PARÁMETRO: {nombre_p} (para {nombre_banco_externo_actual_prompt}) ---")

                met_nb_txt = f"### Descripción Contextual de la Metodología de '{NOMBRE_NUESTRO_BANCO_PROMPT}' para {nombre_p}\n*Nota: El análisis detallado de {NOMBRE_NUESTRO_BANCO_PROMPT} no se genera en este paso para enfocar en el competidor. Se asume conocimiento interno o se puede generar por separado.*\n"
                
                print(f"--- B.1. Analizando metodología y cambios en '{pdf_banco_externo_actual}' para '{nombre_p}' ---")
                chunks_be = semantic_search_filtered(q_be, vector_store, K_VALUE_SEARCH_BE, pdf_banco_externo_actual)
                ctx_be_llm = ""
                if chunks_be:
                    ctx_be_llm = "\n\n---\n\n".join([f"Fuente: {d.metadata.get('source')}, Página: {d.metadata.get('page')}\n{d.page_content}" for d in chunks_be])
                else:
                     logging.warning(f"No se recuperaron chunks para '{nombre_p}' de '{pdf_banco_externo_actual}'. El LLM lo indicará.")

                logging.info(f"Pausa LLM (Análisis BE - {nombre_p})..."); time.sleep(5) 
                
                # Inputs para PROMPT_EXTRACCION_CAMBIOS_BE
                inputs_extraccion_be = {
                    "context": ctx_be_llm, 
                    "nombre_parametro": nombre_p,
                    "aspectos_a_buscar_en_cambios": aspectos_p,
                    "nombre_banco_externo_prompt": nombre_banco_externo_actual_prompt
                }
                analisis_be_txt = run_llm_chain(llm, PROMPT_EXTRACCION_CAMBIOS_BE, 
                                              inputs_extraccion_be,
                                              f"analizar {pdf_banco_externo_actual} para {nombre_p}")
                
                resultados_por_parametro_lista_actual.append({
                    "parametro": nombre_p,
                    "metodologia_nuestro_banco_contexto": met_nb_txt,
                    "analisis_banco_externo_detallado": analisis_be_txt
                })
                print(f"--- FINALIZADO PARÁMETRO: {nombre_p} (para {nombre_banco_externo_actual_prompt}) ---")

            contexto_completo_analisis_para_informe = f"**Contexto de Referencia - Metodologías de {NOMBRE_NUESTRO_BANCO_PROMPT} (Archivo: {PDF_NUESTRO_BANCO_FILENAME}):**\n"
            contexto_completo_analisis_para_informe += "*Nota: La descripción detallada de la metodología de Credicorp por parámetro se ha omitido en esta pasada para enfocar el análisis en el competidor. El LLM debe centrarse en el análisis del banco externo.*\n"
            
            contexto_completo_analisis_para_informe += "\n\n**===============================================**\n"
            contexto_completo_analisis_para_informe += f"**ANÁLISIS PRINCIPAL - Metodologías y Cambios en {nombre_banco_externo_actual_prompt} (Archivo: {pdf_banco_externo_actual}) (Generado con explicaciones de conceptos):**\n"
            for res in resultados_por_parametro_lista_actual:
                contexto_completo_analisis_para_informe += f"\n{res['analisis_banco_externo_detallado']}\n" 
            contexto_completo_analisis_para_informe += "---\n"
            
            debug_informe_context_filename = f"contexto_informe_v11_FINAL_COMPETIDOR_{nombre_banco_externo_actual_prompt.replace(' ', '_')}_debug.txt"
            with open(debug_informe_context_filename, "w", encoding="utf-8") as f_debug:
                f_debug.write(contexto_completo_analisis_para_informe)
            logging.info(f"Contexto para informe individual ({nombre_banco_externo_actual_prompt}) guardado en: {debug_informe_context_filename}")

            print(f"\n--- D. GENERANDO INFORME DE BENCHMARKING INDIVIDUAL PARA {nombre_banco_externo_actual_prompt} ---")
            logging.info(f"Pausa antes de LLM (Informe Individual para {nombre_banco_externo_actual_prompt})..."); time.sleep(6) 

            max_chars_informe_individual_ctx = 300000 
            contexto_final_para_informe_llm = contexto_completo_analisis_para_informe
            
            if len(contexto_final_para_informe_llm) > max_chars_informe_individual_ctx:
                logging.warning(f"Contexto para informe individual ({len(contexto_final_para_informe_llm)} chars) excede {max_chars_informe_individual_ctx}. Se truncará.")
                contexto_final_para_informe_llm = contexto_final_para_informe_llm[:max_chars_informe_individual_ctx - 500] + "\n\n... (CONTEXTO COMPLETO TRUNCADO POR LONGITUD EXCESIVA)"
            
            informe_benchmark_individual_texto_error_base = (
                f"## Análisis Comparativo: {NOMBRE_NUESTRO_BANCO_PROMPT} vs. {nombre_banco_externo_actual_prompt}\n"
                f"*(Basado en el archivo del competidor: {pdf_banco_externo_actual})*\n\n"
            )

            context_has_errors = any("Error LLM" in res["analisis_banco_externo_detallado"] for res in resultados_por_parametro_lista_actual) or \
                                 any("Error de configuración de Prompt" in res["analisis_banco_externo_detallado"] for res in resultados_por_parametro_lista_actual) or \
                                 any("Respuesta de Gemini bloqueada" in res["analisis_banco_externo_detallado"] for res in resultados_por_parametro_lista_actual) or \
                                 len(contexto_final_para_informe_llm.strip()) < 200
            
            if context_has_errors:
                error_details_from_params = [res["analisis_banco_externo_detallado"] for res in resultados_por_parametro_lista_actual if "Error LLM" in res["analisis_banco_externo_detallado"] or "Respuesta de Gemini bloqueada" in res["analisis_banco_externo_detallado"] or "Error de configuración de Prompt" in res["analisis_banco_externo_detallado"]]
                informe_benchmark_individual_texto = informe_benchmark_individual_texto_error_base + \
                                                  (f"**Error Crítico:** No se pudo generar el informe detallado para {nombre_banco_externo_actual_prompt} porque el análisis por parámetros previo para este competidor resultó en uno o más errores, o no produjo contenido útil.\n"
                                                   f"Detalles del error en parámetros (primeros errores): {' | '.join(error_details_from_params[:2])}'\n" # Mostrar hasta 2 errores para brevedad
                                                   f"Por favor, revise los logs anteriores correspondientes a las llamadas 'PROMPT_EXTRACCION_CAMBIOS_BE' para cada parámetro de {nombre_banco_externo_actual_prompt} y el archivo de debug '{debug_informe_context_filename}'.\n")
            elif llm:
                inputs_informe_competidor = {
                    "contexto_completo_analisis": contexto_final_para_informe_llm,
                    "pdf_nuestro_banco": PDF_NUESTRO_BANCO_FILENAME,
                    "pdf_banco_externo": pdf_banco_externo_actual,
                    "nombre_nuestro_banco_prompt": NOMBRE_NUESTRO_BANCO_PROMPT,
                    "nombre_banco_externo_prompt": nombre_banco_externo_actual_prompt
                }
                informe_benchmark_individual_texto = run_llm_chain(
                    llm,
                    PROMPT_INFORME_COMPETIDOR,
                    inputs_informe_competidor,
                    f"generar informe benchmark individual para {nombre_banco_externo_actual_prompt}"
                )
                if "Error LLM" in informe_benchmark_individual_texto or "Respuesta de Gemini bloqueada" in informe_benchmark_individual_texto or "Error:" in informe_benchmark_individual_texto or "Error de configuración de Prompt" in informe_benchmark_individual_texto:
                    informe_benchmark_individual_texto = informe_benchmark_individual_texto_error_base + \
                                                  (f"**Error al generar el resumen del informe para {nombre_banco_externo_actual_prompt}:** {informe_benchmark_individual_texto}\n"
                                                   f"Esto ocurrió al intentar resumir y estructurar el análisis por parámetros del competidor.\n"
                                                   f"El contexto enviado al LLM para este resumen tenía aproximadamente {len(contexto_final_para_informe_llm)} caracteres. "
                                                   f"Se recomienda revisar el archivo de debug: '{debug_informe_context_filename}' y los logs del script para más detalles.\n")
            else:
                 informe_benchmark_individual_texto = informe_benchmark_individual_texto_error_base + \
                                                   f"**Error Crítico:** LLM no disponible para generar el informe de {nombre_banco_externo_actual_prompt}.\n"

            lista_informes_individuales_md.append(informe_benchmark_individual_texto)
            print(f"--- INFORME INDIVIDUAL GENERADO (O ERROR REGISTRADO) PARA {nombre_banco_externo_actual_prompt} ---")

            logging.info(f"===== ANÁLISIS COMPLETADO PARA {pdf_banco_externo_actual} =====")
            if len(LISTA_PDFS_COMPETIDORES) > 1 and pdf_banco_externo_actual != LISTA_PDFS_COMPETIDORES[-1]:
                logging.info("Pausa larga antes del siguiente banco externo..."); time.sleep(90)

        # --- Generación del Informe Consolidado Final ---
        if lista_informes_individuales_md:
            print(f"\n\n=======================================================================")
            print(f"--- GENERANDO INFORME CONSOLIDADO FINAL PARA {NOMBRE_NUESTRO_BANCO_PROMPT} ---")
            print("=======================================================================")
            
            placeholder_contexto_conclusion = "Basado en los análisis individuales previamente generados para cada competidor, donde se detallaron sus prácticas y se explicaron conceptos clave."

            logging.info("Pausa antes de LLM (Conclusión Global)..."); time.sleep(7)
            
            conclusion_global_texto = f"## CONCLUSIONES GLOBALES Y RECOMENDACIONES ESTRATÉGICAS AGREGADAS\n*Error: No se pudo generar la conclusión global y recomendaciones agregadas. Verificar logs.*"
            
            informes_validos_para_conclusion = [
                informe for informe in lista_informes_individuales_md 
                if not ("Error Crítico:" in informe or "Error al generar" in informe or "Error LLM" in informe or "Error de configuración de Prompt" in informe)
            ]

            if informes_validos_para_conclusion and llm:
                inputs_conclusion_global = {
                    "textos_informes_individuales_concatenados_placeholder": placeholder_contexto_conclusion,
                    "nombre_nuestro_banco_prompt": NOMBRE_NUESTRO_BANCO_PROMPT,
                    "lista_nombres_competidores": ", ".join(nombres_competidores_analizados_lista)
                }
                conclusion_global_texto = run_llm_chain(
                    llm,
                    PROMPT_CONCLUSION_GLOBAL, 
                    inputs_conclusion_global,
                    "generar conclusión global y recomendaciones agregadas"
                )
                if "Error LLM" in conclusion_global_texto or "Error:" in conclusion_global_texto or "Error de configuración de Prompt" in conclusion_global_texto:
                     conclusion_global_texto = (f"## CONCLUSIONES GLOBALES Y RECOMENDACIONES ESTRATÉGICAS AGREGADAS\n"
                                                f"*Error al generar la conclusión global y recomendaciones agregadas. Mensaje del LLM: {conclusion_global_texto}*\n"
                                                f"*Esto ocurrió a pesar de tener algunos informes individuales aparentemente válidos. Revise los logs.*\n")
            elif not informes_validos_para_conclusion:
                conclusion_global_texto = (f"## CONCLUSIONES GLOBALES Y RECOMENDACIONES ESTRATÉGICAS AGREGADAS\n"
                                           f"*No se pudo generar una conclusión global significativa ya que todos los análisis individuales de los competidores resultaron en errores críticos o no se pudieron procesar correctamente.*\n"
                                           f"*Por favor, revise los errores detallados en cada sección de análisis de competidor anterior y los logs del script.*\n")

            documento_final_md = f"# INFORME CONSOLIDADO DE BENCHMARKING METODOLÓGICO IFRS 9\n\n"
            documento_final_md += f"## Para: Comité de Riesgos de {NOMBRE_NUESTRO_BANCO_PROMPT}\n"
            documento_final_md += f"**Fecha de Generación:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            documento_final_md += "## Introducción\n"
            intro_competidores = ", ".join(nombres_competidores_analizados_lista) if nombres_competidores_analizados_lista else 'N/A (No se procesaron competidores)'
            documento_final_md += (
                f"Este informe presenta un análisis de benchmarking de las metodologías de riesgo crediticio bajo IFRS 9. "
                f"El objetivo principal es identificar prácticas destacadas en bancos competidores ({intro_competidores}) "
                f"que puedan inspirar oportunidades de mejora y refinamiento en las metodologías de **{NOMBRE_NUESTRO_BANCO_PROMPT}**. "
                f"Se busca fortalecer la gestión de riesgos y la robustez de los modelos en nuestra entidad.\n\n"
                "Cada competidor ha sido analizado individualmente. Los hallazgos detallados (o errores en su procesamiento) se presentan a continuación, seguidos de una conclusión global y recomendaciones estratégicas agregadas para Credicorp.\n\n"
            )
            documento_final_md += "---\n"

            for informe_ind in lista_informes_individuales_md:
                documento_final_md += informe_ind + "\n\n---\n\n"
            
            documento_final_md += conclusion_global_texto
            
            safe_nb_name = NOMBRE_NUESTRO_BANCO_PROMPT.replace(" ", "_").replace("/","_").replace("\\","_")
            output_filename_consolidado_md = f"Informe_Consolidado_Benchmark_{safe_nb_name}_{time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(output_filename_consolidado_md, "w", encoding="utf-8") as f_out_final:
                f_out_final.write(documento_final_md)
            logging.info(f"Informe consolidado final guardado en: {output_filename_consolidado_md}")
            print(f"\n--- INFORME CONSOLIDADO FINAL GUARDADO EN: {output_filename_consolidado_md} ---")
        else:
            logging.info("No se generaron informes individuales de competidores (probablemente debido a que la lista de competidores estaba vacía o todos los análisis fallaron), por lo tanto no se creará un informe consolidado.")

    elif not llm:
        logging.critical("La inicialización del LLM falló. El análisis no puede continuar.")
    elif not (vector_store and hasattr(vector_store, 'index') and vector_store.index and vector_store.index.ntotal > 0):
        logging.critical(f"El Vector Store no está disponible o está vacío. El análisis no puede continuar.")
    else:
        logging.warning("LLM o Vector Store no disponibles/vacíos por alguna otra razón no cubierta. Análisis no posible.")

    logging.info("--- SCRIPT FINALIZADO ---")