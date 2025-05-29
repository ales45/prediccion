package org.example.prediccion.weka; // Ajusta si tu paquete es diferente

import weka.core.Instances;

public interface WekaPredictor {
    /**
     * Construye un modelo a partir de los datos y devuelve una representación
     * del modelo o los resultados de la predicción.
     *
     * @param data Las instancias de Weka (ya deben tener el índice de clase asignado).
     * @return Un String con el resultado del modelo o la predicción.
     * @throws Exception Si ocurre un error durante la construcción o predicción.
     */
    String buildModelAndGetResults(Instances data) throws Exception;
}