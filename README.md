# Proyecto MARA RL

# MARA - Asignación de Volumen a Proveedores usando Aprendizaje por Refuerzo (RL)

Este proyecto desarrolla un sistema de asignación inteligente de volumen de paquetes a proveedores utilizando técnicas de **Aprendizaje por Refuerzo** (*Reinforcement Learning*).  
El objetivo principal es mejorar la eficiencia logística optimizando la distribución de volumen entre distintos proveedores bajo criterios de costo, tiempo de entrega y cumplimiento.

---

## 📂 Estructura del Proyecto

- `data/` ➔ Datasets utilizados para entrenamiento y validación del agente (no incluidos en el repositorio por confidencialidad).
- `notebooks/` ➔ Notebooks de definición de entorno y entrenamiento del agente RL.
- `src/` ➔ Código fuente auxiliar (pendiente de incluir en futuras versiones).
- `models/` ➔ Modelos entrenados (no incluidos en esta versión pública).
- `docs/` ➔ Documentación técnica (en construcción).

---

## 🎯 Objetivos del Proyecto

- **Asignar volumen de paquetes** de forma optimizada entre varios proveedores.
- **Maximizar la eficiencia** considerando tiempo de entrega, costos y cumplimiento.
- **Adaptarse dinámicamente** a cambios en la demanda y restricciones operativas.
- **Simular escenarios** de alta estacionalidad o cambios bruscos en el mercado.

---

## 🛠️ Tecnologías y Herramientas Utilizadas

- **Python**
- **Stable-Baselines3** (algoritmos de RL como DQN y PPO)
- **OpenAI Gym** (para construcción de entornos personalizados)
- **Scikit-learn** (análisis de datos y preprocesamiento)
- **Matplotlib** y **Seaborn** (visualización de resultados)
- **Git/GitHub** (control de versiones)

---

## 🚧 Estado del Proyecto

El proyecto se encuentra **en desarrollo**.  
Se irán publicando progresivamente los notebooks de experimentación, scripts de entrenamiento, modelos finales y resultados de simulaciones.

---

## 📦 Notas sobre reproducibilidad

Este repositorio no incluye el archivo `rl_experiences.pkl` utilizado para entrenar el agente, ya que contiene datos generados internamente para simulaciones de entrenamiento.

Para reproducir el proceso de entrenamiento:
- Se debe generar un conjunto de experiencias (`experiences`) siguiendo la estructura esperada por el entorno `AsignacionEnv`.
- También es posible modificar los notebooks para crear experiencias sintéticas si se desea realizar pruebas independientes.

El modelo entrenado (`assignment_model.zip`) tampoco está incluido por razones de confidencialidad.

---


