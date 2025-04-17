import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AsignacionEnv(gym.Env):
    def __init__(self, experiences=None, df=None, priority_supplier_enabled=True):
        super().__init__()
        print("Inicializando entorno de asignación...")

        # Guardar referencias a los datos
        self.experiences = experiences
        self.df = df.reset_index(drop=True) if df is not None else None

        # Determinar la fuente de datos
        self.data_source = self.experiences if self.experiences else self.df

        # Flag para la lógica de priorización
        self.priority_supplier_enabled = priority_supplier_enabled

        # Espacio de acciones discretas: 0%, 25%, 50%, 75%, 100%
        self.action_space = spaces.Discrete(5)
        self.action_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Detectar dimensión del espacio de estados
        if self.experiences:
            first_state = np.array(self.experiences[0]['state'], dtype=np.float32)
        else:
            first_state = np.array(self.df.loc[0, 'state_vector'], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(first_state),),
            dtype=np.float32
        )

        # Inicializar variables
        self.current_index = 0
        self.state = None
        self.num_experiences = len(self.experiences) if self.experiences else len(self.df)

        # Configuración para priorización por rango de peso (versión simplificada)
        self.priority_allocation = {
            0: 1.0, 1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 6: 0.5
        }

        # Límites por proveedor (nombres genéricos por confidencialidad)
        self.provider_limits = {
            "Proveedor_A": 0.5, "Proveedor_B": 0.5, "Proveedor_C": 0.25, "Proveedor_D": 0.25
        }

        print(f"Entorno cargado con {self.num_experiences} experiencias")

    def reset(self, *, seed=None, options=None):
        """Reiniciar el entorno para un nuevo episodio"""
        super().reset(seed=seed)
        self.current_index = 0

        # Obtener el estado inicial sin recursión
        if self.experiences:
            self.state = np.array(self.experiences[0]['state'], dtype=np.float32)
        else:
            self.state = np.array(self.df.loc[0, 'state_vector'], dtype=np.float32)

        return self.state, {}

    def step(self, action_index):
        """Ejecutar un paso en el entorno con la acción dada"""
        # Verificar límites para evitar errores
        if self.current_index >= self.num_experiences:
            # Estado terminal si estamos fuera de límites
            return np.zeros_like(self.state), 0.0, True, False, {}

        # Obtener datos del registro actual (sin recursión)
        if self.experiences:
            current_data = self.experiences[self.current_index]
            provider = current_data['provider_name']
            is_priority_supplier = current_data.get('es_priority_supplier', 0)
            weight_range = current_data.get('weight_range_ordinal', 0)
            original_reward = current_data['reward']
        else:
            current_data = self.df.iloc[self.current_index]
            provider = current_data['provider_name']
            is_priority_supplier = 1 if provider == 'Priority Supplier' else 0
            weight_range = int(current_data.get('weight_range_ordinal', 0))
            original_reward = current_data['reward_hibrido']

        # Aplicar lógica híbrida (sin recursión)
        action_value = self.action_values[action_index]

        # Modificar acción según reglas
        if self.priority_supplier_enabled:
            if is_priority_supplier:
                # Para el proveedor prioritario, usar el valor recomendado para su rango de peso
                recommended = self.priority_allocation.get(weight_range, 0.7)
                action_value = max(recommended, action_value)
            elif provider in self.provider_limits:
                # Aplicar límites para otros proveedores
                action_value = min(self.provider_limits[provider], action_value)

        # Calcular recompensa
        reward = original_reward * action_value

        # Avanzar al siguiente registro
        self.current_index += 1
        terminated = self.current_index >= self.num_experiences

        # Obtener siguiente estado
        if not terminated and self.current_index < self.num_experiences:
            if self.experiences:
                self.state = np.array(self.experiences[self.current_index]['state'], dtype=np.float32)
            else:
                self.state = np.array(self.df.iloc[self.current_index]['state_vector'], dtype=np.float32)

        # Información adicional
        info = {
            'provider': provider,
            'action_value': action_value,
            'original_reward': original_reward
        }

        return self.state, reward, terminated, False, info
