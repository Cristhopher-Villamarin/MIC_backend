import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics.pairwise import euclidean_distances

# Definición de la clase VAE corregida con métodos de serialización
class VAE(tf.keras.Model):
    def __init__(self, dim_entrada=42, dim_latente=256, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.dim_entrada = dim_entrada
        self.dim_latente = dim_latente
        
        # Construir las capas en __init__
        self.codificador = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(dim_entrada,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(dim_latente + dim_latente)  # Media y log varianza
        ])
        self.decodificador = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(dim_latente,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(dim_entrada)  # Activación lineal
        ])

    def codificar(self, x):
        media, logvar = tf.split(self.codificador(x), num_or_size_splits=2, axis=1)
        return media, logvar

    def reparametrizar(self, media, logvar):
        eps = tf.random.normal(shape=tf.shape(media))
        return media + tf.exp(0.5 * logvar) * eps

    def decodificar(self, z):
        return self.decodificador(z)

    def call(self, x):
        media, logvar = self.codificar(x)
        z = self.reparametrizar(media, logvar)
        x_reconstruida = self.decodificar(z)
        return x_reconstruida, media, logvar

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'dim_entrada': self.dim_entrada,
            'dim_latente': self.dim_latente
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Filtrar solo los argumentos válidos para el constructor
        valid_args = {'dim_entrada', 'dim_latente'}
        filtered_config = {k: v for k, v in config.items() if k in valid_args}
        return cls(**filtered_config)

def generar_datos_sinteticos_cargado(modelo, escalador, num_muestras, columnas_caracteristicas, columna_cluster):
    dim_latente = 256
    datos_sinteticos = []
    intentos = 0
    max_intentos = num_muestras * 3

    while len(datos_sinteticos) < num_muestras and intentos < max_intentos:
        z_muestra = tf.random.normal([1, dim_latente], mean=0.0, stddev=2.5).numpy()
        muestra = modelo.decodificador(tf.convert_to_tensor(z_muestra, dtype=tf.float32)).numpy()
        muestra = np.clip(muestra, 0, 1)  # Recortar a [0, 1] en espacio normalizado
        muestra = escalador.inverse_transform(muestra)

        if len(datos_sinteticos) == 0 or all(euclidean_distances(muestra, np.array(datos_sinteticos))[0] > 0.005):
            datos_sinteticos.append(muestra[0])

        intentos += 1

    if len(datos_sinteticos) < num_muestras:
        print(f"Advertencia: Solo se generaron {len(datos_sinteticos)} muestras únicas después de {max_intentos} intentos")

    datos_sinteticos = np.array(datos_sinteticos)

    # Post-procesamiento
    df_sintetico = pd.DataFrame(datos_sinteticos, columns=columnas_caracteristicas)
    columnas_cluster = [col for col in columnas_caracteristicas if col.startswith('cluster_')]
    probabilidades_cluster = df_sintetico[columnas_cluster].values
    clusters = np.argmax(probabilidades_cluster, axis=1)
    df_sintetico[columna_cluster] = clusters
    df_sintetico = df_sintetico.drop(columns=columnas_cluster)

    # Recortar para asegurar que emociones y otras características se mantengan en [0, 1]
    for col in df_sintetico.columns:
        if col != columna_cluster:
            if col == 'verified_account':
                df_sintetico[col] = np.round(df_sintetico[col]).astype(int)
            elif col in ['in_polarity', 'out_polarity']:
                df_sintetico[col] = np.clip(df_sintetico[col], -1.0, 1.0)
            else:
                df_sintetico[col] = np.clip(df_sintetico[col], 0, 1)

    return df_sintetico

def cargar_modelo_y_escalador(model_path='vae_model.keras', scaler_path='scaler.pkl', metadata_path='model_metadata.json'):
    try:
        # Cargar modelo con custom_objects
        modelo = tf.keras.models.load_model(model_path, custom_objects={'VAE': VAE})
        escalador = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        columnas_caracteristicas = metadata['feature_columns']
        columna_cluster = metadata['cluster_column']
        
        return modelo, escalador, columnas_caracteristicas, columna_cluster
        
    except Exception as e:
        print(f"Error detallado al cargar: {str(e)}")
        # Intentar cargar solo los metadatos y escalador para diagnóstico
        try:
            escalador = joblib.load(scaler_path)
            print("✓ Escalador cargado correctamente")
        except Exception as e_scaler:
            print(f"✗ Error al cargar escalador: {e_scaler}")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print("✓ Metadatos cargados correctamente")
            print(f"Dimensiones esperadas: entrada={len(metadata['feature_columns'])}")
        except Exception as e_meta:
            print(f"✗ Error al cargar metadatos: {e_meta}")
        
        raise Exception(f"Error al cargar modelo, escalador o metadatos: {str(e)}")

# Función adicional para recrear el modelo si es necesario
def recrear_modelo_vae(dim_entrada=42, dim_latente=256):
    """
    Crea una nueva instancia del modelo VAE con las dimensiones especificadas.
    Útil si el modelo guardado no se puede cargar debido a problemas de serialización.
    """
    try:
        modelo = VAE(dim_entrada=dim_entrada, dim_latente=dim_latente)
        
        # Construir el modelo con datos dummy
        dummy_input = tf.random.normal((1, dim_entrada))
        _ = modelo(dummy_input)
        
        print(f"✓ Modelo VAE recreado con dim_entrada={dim_entrada}, dim_latente={dim_latente}")
        return modelo
        
    except Exception as e:
        print(f"✗ Error al recrear modelo: {str(e)}")
        raise e

# Función para guardar el modelo correctamente
def guardar_modelo_correctamente(modelo, escalador, columnas_caracteristicas, columna_cluster, 
                                model_path='vae_model.keras', scaler_path='scaler.pkl', 
                                metadata_path='model_metadata.json'):
    """
    Guarda el modelo, escalador y metadatos correctamente.
    """
    try:
        # Guardar modelo
        modelo.save(model_path)
        print(f"✓ Modelo guardado en: {model_path}")
        
        # Guardar escalador
        joblib.dump(escalador, scaler_path)
        print(f"✓ Escalador guardado en: {scaler_path}")
        
        # Guardar metadatos
        metadata = {
            'feature_columns': columnas_caracteristicas,
            'cluster_column': columna_cluster,
            'model_info': {
                'dim_entrada': len(columnas_caracteristicas),
                'dim_latente': 256,
                'version': '1.0'
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadatos guardados en: {metadata_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error al guardar: {str(e)}")
        return False