import pickle
import numpy as np # type: ignore
from datetime import datetime
import pandas as pd # type: ignore
from stable_baselines3 import PPO # type: ignore
from stable_baselines3.common.env_checker import check_env # type: ignore
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback # type: ignore
from env import AsignacionEnv # type: ignore

# Cargar las experiencias
print("Cargando experiencias...")
with open('rl_experiences.pkl', 'rb') as f:
    experiences = pickle.load(f)

# Preparar datos  
date = pd.Timestamp('2025-01-01')
pre_date = [exp for exp in experiences if exp['semana'] < date]
post_date = [exp for exp in experiences if exp['semana'] >= pre_date]

train_experiences = pre_date[-5000:] + post_date  # Últimos 5000 antes de date + todos
validation_experiences = post_date[-500:]  # Últimos 500 datos para validación

# Crear entornos separados para entrenamiento y validación
train_env = AsignacionEnv(experiences=train_experiences, priority_supplier_enabled=True)
eval_env = AsignacionEnv(experiences=validation_experiences, priority_supplier_enabled=True)

# Verificar el entorno
check_env(train_env, warn=True)

# Callback para evaluación
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./logs/',
    log_path='./logs/',
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Inicializar el modelo con parámetros completos
model = PPO(
    "MlpPolicy", 
    train_env, 
    verbose=1, 
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

# Entrenamiento completo
print("Iniciando entrenamiento completo...")
model.learn(
    total_timesteps=50000,  
    callback=eval_callback,
    progress_bar=True
)

# Guardar el modelo entrenado
model.save("assignment_model")
print("Modelo guardado exitosamente")

# Evaluar en datos de marzo específicamente
print("\nEvaluando modelo en datos de marzo...")
test_env = AsignacionEnv(experiences=post_date, priority_supplier_enabled=True)
obs, _ = test_env.reset()
total_reward = 0
total_steps = 0

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    total_reward += reward
    total_steps += 1
    
    if total_steps % 50 == 0:
        print(f"Paso {total_steps}: Recompensa acumulada {total_reward:.4f}")

print(f"\nRecompensa total: {total_reward:.4f}")
print(f"Recompensa promedio por paso: {total_reward/total_steps:.4f}")