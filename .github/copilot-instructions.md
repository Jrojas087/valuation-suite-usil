# Copilot / AI Agent instructions — ValuationSuite USIL

Resumen rápido
- Entrada principal: [app.py](app.py#L1-L40) — app Streamlit que orquesta UI, cálculos y export.
- Exporter: [report_usil.py](report_usil.py#L1-L40) — `OnePager` dataclass y funciones TXT/PDF (ReportLab).
- Dependencias clave: [requirements.txt](requirements.txt#L1)

Arquitectura y propósito
- Este repo es una pequeña “single-page” app Streamlit para evaluación financiera (DCF + Monte Carlo).
- `app.py` contiene la UI (sidebar, métricas, gráficos) y la lógica de negocio en una sola capa; delega la generación de TXT/PDF a `report_usil.py`.
- Diseño intencional: evitar dependencias gráficas tradicionales (no usa matplotlib), PDF se genera con ReportLab (paquete `reportlab`).

Flujos críticos y comandos
- Ejecutar la app localmente:
  - `streamlit run app.py`
- Si falta export PDF, instalar ReportLab:
  - `pip install reportlab` o agregar a `requirements.txt` y reinstalar.
- No hay tests automatizados en el repo; validación manual: ejecutar la app y probar export de TXT/PDF.

Patrones y convenciones del proyecto
- UI-first: todos los inputs están en el sidebar de `app.py` (buscar `st.sidebar` para cambios de parámetros).
- Cálculos reproducibles: Monte Carlo usa `numpy.random.default_rng()` internamente y no expone semilla en UI — evita modificar RNG global en PRs sin motivo.
- Caching: funciones costosas usan `@st.cache_data` (ej. `run_monte_carlo`) — respeta la inmutabilidad de entradas para evitar resultados inesperados.
- Validaciones críticas: la consistencia del TV depende de `wacc > g_inf + MIN_SPREAD` (constante `MIN_SPREAD` en `app.py`) — mantener esta comprobación cuando refactors afecten valoración.
- Exporter API: `report_usil.OnePager` es el contrato para generación TXT/PDF; `app.py` crea la instancia y llama `build_onepager_text` / `generate_onepager_pdf`.

Integraciones y puntos de fricción
- ReportLab optional: `report_usil.py` define `REPORTLAB_OK` — la app condiciona botones de descarga PDF a esta bandera.
- Numpy-financial: la función `npf.irr` se usa en `safe_irr` — pruebas numéricas deben considerar casos NaN/inf.
- Plotly para gráficos interactivos; Streamlit `st.plotly_chart` está en `app.py`.

Sugerencias específicas para AI agents
- Prioriza cambios pequeños y comprobables: ajustar textos, añadir parámetros en sidebar, o mejorar tolerancias numéricas.
- No modifiques el RNG o elimines `st.cache_data` sin pruebas manuales en local: puede cambiar resultados y rendimiento.
- Para cambios en export PDF, inspecciona [report_usil.py](report_usil.py#L1-L200): la generación de PDF está centralizada y usa helpers locales (`rr`, `t`, `para`).
- Si agregas deps, actualiza [requirements.txt](requirements.txt#L1) y documenta por qué.

Ejemplos rápidos (extraídos del repo)
- Cómo se calcula FCF1 en `app.py`:
  - `fcf_y1 = nopat + dep_y1 - capex_y1 - delta_wc` (buscar en `app.py` "Puente contable a FCF1").
- Cómo se ejecuta Monte Carlo:
  - `run_monte_carlo(...)` está decorada con `@st.cache_data` y devuelve arrays `(npv_s, g_s, w_s, capex_s, mult_s, idx)`.

Qué evitar
- No expongas la semilla RNG a menos que el cambio sea explícitamente para reproducibilidad en pruebas.
- Evitar introducir matplotlib ni otras grandes dependencias de plotting; usar Plotly/Streamlit como patrón existente.

Dónde mirar primero para evaluar cambios
- [app.py](app.py#L1-L120) — UI, inputs y configuraciones.
- [app.py](app.py#L120-L320) — funciones utilitarias y Monte Carlo.
- [report_usil.py](report_usil.py#L1-L220) — modelo `OnePager` y generación TXT/PDF.
- [requirements.txt](requirements.txt#L1) — dependencias necesarias.

Notas finales
- El repo está pensado para uso académico (MBA). Las modificaciones que cambien la lógica financiera deben documentarse en el PR y preferiblemente acompañarse de una demostración local.

---
¿Quieres que incorpore ejemplos de PR templates o reglas de formato (Black, isort)? Puedo añadirlos si lo prefieres.