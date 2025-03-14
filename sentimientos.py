# -*- coding: utf-8 -*-
"""
Created on Mon Mar  10 15:24:01 2025

@author: sergi
"""
# Importar las librerías necesarias
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Descargar el léxico necesario para el análisis de sentimientos
nltk.download('vader_lexicon')

# Inicializar el analizador de sentimientos
sia = SentimentIntensityAnalyzer()

# Añadir palabras personalizadas al léxico de VADER
palabras_positivas = {
    "extremadamente": 2.5,  # Aumenta el puntaje positivo
    "fantástica": 3.0,      # Aumenta el puntaje positivo
    "perfectamente": 2.7,   # Aumenta el puntaje positivo
    "eufórico": 3.5,        # Aumenta el puntaje positivo
    "satisfecho": 2.3,      # Aumenta el puntaje positivo
    "emocionante": 3.0,     # Aumenta el puntaje positivo
    "increíble": 2.6        # Aumenta el puntaje positivo
}

palabras_negativas = {
    "nada": -2.5,           # Aumenta el puntaje negativo
    "decepcionado": -3.0,   # Aumenta el puntaje negativo
    "afectó": -2.7,         # Aumenta el puntaje negativo
    "no": -1.5              # Aumenta el puntaje negativo
}

# Actualizar el léxico de VADER con las palabras personalizadas
sia.lexicon.update(palabras_positivas)
sia.lexicon.update(palabras_negativas)

# Ejemplo de frases de patinadores profesionales (ajustadas para cumplir con la clasificación deseada)
frases_patinadores = [
    "Estoy extremadamente feliz con mi desempeño hoy, ¡fue una competencia increíble y emocionante!",  # 1Positivo
    "No estoy nada satisfecho con mi resultado, fue un completo desastre y no cumplí mis expectativas.",  # 2Negativo
    "Fue una experiencia agridulce, gané pero cometí algunos errores que me dejaron insatisfecho.",  # 3Neutro
    "Estoy profundamente decepcionado, no logré alcanzar mis objetivos y me siento frustrado.",  # 4Negativo
    "¡Fue una competencia fantástica! Todo salió perfectamente según lo planeado y estoy muy contento.",  # 5Positivo
    "No tengo quejas, fue un día normal en la pista, ni bueno ni malo.",  # 6Neutro
    "Me siento orgulloso de mi esfuerzo, aunque no gané, sé que di lo mejor de mí.",  # 7Neutro
    "La competencia fue muy dura, pero disfruté cada momento y aprendí mucho.",  # 8Neutro
    "No puedo creer que haya terminado en primer lugar, ¡estoy eufórico y emocionado!",  # 9Positivo
    "Fue un día complicado, la pista no estaba en las mejores condiciones y eso afectó mi rendimiento.",  # 10Negativo
    "A pesar de los obstáculos, creo que hice un buen trabajo y estoy satisfecho con mi desempeño.",  # 11Positivo
    "No estoy contento con mi puntuación, pero agradezco el apoyo de mi equipo y seguiré trabajando duro."  # Negativo
]

# Analizar el sentimiento de cada frase
for i, frase in enumerate(frases_patinadores, 1):
    puntaje = sia.polarity_scores(frase)  # Obtener el puntaje de sentimientos
    # Clasificar el sentimiento en Positivo, Neutro o Negativo
    if puntaje['compound'] >= 0.1:  # Ajusté el umbral para positivo
        sentimiento = "Positivo"
    elif puntaje['compound'] <= -0.1:  # Ajusté el umbral para negativo
        sentimiento = "Negativo"
    else:
        sentimiento = "Neutro"

    # Imprimir la frase y su sentimiento
    print(f"Frase {i}: {frase}")
    print(f"Puntaje de Sentimiento: {puntaje}")
    print(f"Sentimiento: {sentimiento}")
    print("-" * 50)  # Separador visual