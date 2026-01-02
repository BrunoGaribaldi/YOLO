""" Cómo funciona:
Carga de datos: Lee detections.txt y extrae las coordenadas Y.
Métricas de movimiento:
Varianza: Variabilidad general de las posiciones Y
Desviación estándar: Dispersión de los valores
Rango: Diferencia entre máximo y mínimo (movimiento total)
Cambio promedio: Variación promedio entre frames consecutivos
Tendencia: Pendiente de la señal (movimiento direccional)
Detección ON/OFF:
Sistema de puntuación con 5 métricas ponderadas
Si 3 o más métricas indican movimiento → ON
Si menos de 3 métricas → OFF
Calcula una confianza (0-100%)
Retorna el estado (ON/OFF) y confianza.

Umbrales por defecto:
Rango mínimo: 10 píxeles
Desviación estándar mínima: 5 píxeles
Cambio promedio mínimo: 1 píxel por frame
Varianza mínima: 25

Nota: Se puede ajustar los umbrales de las métricas para mejorar la precisión. """

import os
import numpy as np
import re
from dotenv import load_dotenv

def load_points(path):
    """Carga los puntos (frame, y) desde detections.txt"""
    frames, ys = [], []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"\((\d+),\s*([0-9.]+)\)", line)
                if m:
                    frames.append(int(m.group(1)))
                    ys.append(float(m.group(2)))
    except FileNotFoundError:
        return np.array([]), np.array([])
    return np.array(frames), np.array(ys)

def calculate_movement_metrics(ys):
    """
    Calcula métricas de movimiento para determinar si el pump jack está activo.
    
    Returns:
        dict: Diccionario con métricas calculadas
    """
    if len(ys) < 2:
        return {
            'variance': 0.0,
            'std_dev': 0.0,
            'range': 0.0,
            'mean_change': 0.0,
            'max_change': 0.0,
            'trend': 0.0
        }
    
    # Varianza y desviación estándar
    variance = np.var(ys)
    std_dev = np.std(ys)
    
    # Rango de movimiento (max - min)
    y_range = np.max(ys) - np.min(ys)
    
    # Cambios entre frames consecutivos
    changes = np.abs(np.diff(ys))
    mean_change = np.mean(changes)
    max_change = np.max(changes)
    
    # Tendencia: pendiente promedio (usando regresión lineal simple)
    if len(ys) > 1:
        x = np.arange(len(ys))
        trend = np.polyfit(x, ys, 1)[0]  # Pendiente
    else:
        trend = 0.0
    
    return {
        'variance': variance,
        'std_dev': std_dev,
        'range': y_range,
        'mean_change': mean_change,
        'max_change': max_change,
        'trend': abs(trend)  # Valor absoluto de la tendencia
    }

def detect_on_off(detections_path, 
                  min_range=10.0,      # Rango mínimo de Y para considerar ON
                  min_std=5.0,          # Desviación estándar mínima
                  min_mean_change=1.0,  # Cambio promedio mínimo entre frames
                  min_variance=25.0):   # Varianza mínima
    """
    Detecta si el pump jack está encendido (ON) o apagado (OFF).
    
    Args:
        detections_path: Ruta al archivo detections.txt
        min_range: Rango mínimo de Y para considerar movimiento (default: 10 píxeles)
        min_std: Desviación estándar mínima (default: 5 píxeles)
        min_mean_change: Cambio promedio mínimo entre frames (default: 1 píxel)
        min_variance: Varianza mínima (default: 25)
    
    Returns:
        dict: {
            'status': 'ON' o 'OFF',
            'confidence': float (0-1),
            'metrics': dict con todas las métricas,
            'reason': str explicando la decisión
        }
    """
    # Cargar datos
    frames, ys = load_points(detections_path)
    
    # Validaciones
    if len(ys) == 0:
        return {
            'status': 'UNKNOWN',
            'confidence': 0.0,
            'metrics': {},
            'reason': 'No se encontraron datos en detections.txt'
        }
    
    if len(ys) < 5:
        return {
            'status': 'UNKNOWN',
            'confidence': 0.0,
            'metrics': {},
            'reason': f'Datos insuficientes: solo {len(ys)} puntos (mínimo 5)'
        }
    
    # Calcular métricas
    metrics = calculate_movement_metrics(ys)
    
    # Sistema de puntuación para determinar ON/OFF
    scores = {
        'range': 1.0 if metrics['range'] >= min_range else 0.0,
        'std': 1.0 if metrics['std_dev'] >= min_std else 0.0,
        'mean_change': 1.0 if metrics['mean_change'] >= min_mean_change else 0.0,
        'variance': 1.0 if metrics['variance'] >= min_variance else 0.0,
        'trend': 1.0 if abs(metrics['trend']) > 0.1 else 0.0  # Tendencia significativa
    }
    
    # Peso de cada métrica
    weights = {
        'range': 0.3,
        'std': 0.25,
        'mean_change': 0.25,
        'variance': 0.1,
        'trend': 0.1
    }
    
    # Calcular score total
    total_score = sum(scores[key] * weights[key] for key in scores)
    confidence = total_score
    
    # Determinar estado
    # Si al menos 3 de 5 métricas indican movimiento, está ON
    active_metrics = sum(1 for v in scores.values() if v > 0)
    
    if active_metrics >= 3 or total_score >= 0.6:
        status = 'ON'
        reason = f"Movimiento detectado: {active_metrics}/5 métricas activas, score={total_score:.2f}"
    else:
        status = 'OFF'
        reason = f"Sin movimiento significativo: {active_metrics}/5 métricas activas, score={total_score:.2f}"
    
    return {
        'status': status,
        'confidence': confidence,
        'metrics': metrics,
        'reason': reason,
        'n_points': len(ys)
    }

def check_pump_jack_status(OUTPUT_FILE):
    """
    Función principal que carga variables de entorno y detecta el estado.
    """
    
    
    if not OUTPUT_FILE:
        print("Error: OUTPUTFILE no está definido en .env")
        return None
    
    result = detect_on_off(OUTPUT_FILE)
    return result


