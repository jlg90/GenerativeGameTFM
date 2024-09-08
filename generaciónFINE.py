import subprocess
import os

# Variables y rutas importantes
model_name = "CompVis/stable-diffusion-v1-4"  # Nombre del modelo base
instance_data_dir = "C:/Users/alifr/Desktop/PruebaGeneración/steampunk_images"  # Carpeta con tus imágenes de entrenamiento
output_dir = "C:/Users/alifr/Desktop/PruebaGeneración/output_model"  # Carpeta donde se guardará el modelo entrenado
instance_prompt = "a photo of a steampunk style object"  # Prompt asociado con tus imágenes
resolution = 512  # Resolución de las imágenes de entrada
train_batch_size = 1  # Tamaño del batch para el entrenamiento
learning_rate = 5e-6  # Tasa de aprendizaje
max_train_steps = 400  # Máximo número de pasos de entrenamiento
checkpointing_steps = 100  # Guardar checkpoints cada 100 pasos

# Ruta completa al script `train_dreambooth.py`
train_dreambooth_script = "C:/Users/alifr/documents/github/diffusers/examples/dreambooth/train_dreambooth.py"  # Cambia esta ruta según sea necesario

# Opcional: prior preservation para mejorar la diversidad en las imágenes generadas
use_prior_preservation = False
class_data_dir = "C:/Users/alifr/Desktop/PruebaGeneración/class_images"  # Carpeta con imágenes de preservación de clase (opcional)
class_prompt = "a photo of an object"  # Prompt para las imágenes de preservación de clase

# Comando para ejecutar el script oficial de DreamBooth
training_command = [
    "accelerate", "launch", "--mixed_precision=fp16", train_dreambooth_script,
    "--pretrained_model_name_or_path", model_name,
    "--instance_data_dir", instance_data_dir,
    "--output_dir", output_dir,
    "--instance_prompt", instance_prompt,
    "--resolution", str(resolution),
    "--train_batch_size", str(train_batch_size),
    "--learning_rate", str(learning_rate),
    "--max_train_steps", str(max_train_steps),
    "--checkpointing_steps", str(checkpointing_steps),
]

# Si usas preservación de clase, agrega los siguientes parámetros
if use_prior_preservation:
    training_command.extend([
        "--with_prior_preservation",
        "--class_data_dir", class_data_dir,
        "--class_prompt", class_prompt,
        "--prior_loss_weight", "1.0"
    ])

# Ejecutar el entrenamiento con el script oficial de DreamBooth
subprocess.run(training_command)
