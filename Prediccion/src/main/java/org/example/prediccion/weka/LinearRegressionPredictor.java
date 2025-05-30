package org.example.prediccion.weka; // Ajusta si tu paquete es diferente

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instance; // Importar weka.core.Instance
import weka.core.Instances;

public class LinearRegressionPredictor implements WekaPredictor {

    @Override
    public String buildModelAndGetResults(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalArgumentException(
                    "El atributo clase (target) no ha sido asignado en los datos de Weka (Instances) para Regresión Lineal."
            );
        }

        // Verificar que el atributo clase sea numérico para Regresión Lineal
        if (!data.classAttribute().isNumeric()) {
            throw new IllegalArgumentException(
                    "El atributo clase (target) debe ser NUMÉRICO para Regresión Lineal. " +
                            "El atributo '" + data.classAttribute().name() + "' es de tipo " +
                            Attribute.typeToString(data.classAttribute().type()) + "."
            );
        }

        if (data.numInstances() == 0) {
            return "No hay datos de entrenamiento para el modelo de Regresión Lineal.";
        }

        // 1. Crear y construir el clasificador de Regresión Lineal
        LinearRegression lr = new LinearRegression();
        // Opcional: Configurar parámetros de LinearRegression si es necesario
        // lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION)); // Ejemplo: no seleccionar atributos

        lr.buildClassifier(data); // Entrenar el modelo

        // 2. Preparar el String de resultados
        StringBuilder resultBuilder = new StringBuilder();

        // Añadir la información del modelo de regresión (ecuación, coeficientes, etc.)
        resultBuilder.append("=== Modelo de Regresión Lineal Construido ===\n");
        resultBuilder.append(lr.toString()).append("\n\n");

        // Añadir la sección para predicciones de instancias con valor objetivo original desconocido
        resultBuilder.append("=== Predicciones para Instancias con Valor Objetivo Originalmente Desconocido (?) ===\n");
        resultBuilder.append(String.format("%-10s | %-20s\n", "Instancia#", "Valor Predicho"));
        resultBuilder.append("---------------------------------------\n");

        boolean foundInstancesToPredict = false;

        // 3. Iterar sobre cada instancia para hacer una predicción
        //    SOLO para aquellas donde el valor objetivo original era desconocido/perdido.
        for (int i = 0; i < data.numInstances(); i++) {
            Instance currentInstance = data.instance(i);

            // Verificar si el valor de la clase original de esta instancia estaba marcado como perdido
            if (currentInstance.classIsMissing()) {
                foundInstancesToPredict = true;

                // Predecir el valor para la instancia actual usando el modelo entrenado
                double predictedValue = lr.classifyInstance(currentInstance); // Devuelve un valor numérico

                // Añadir la información de la instancia y su predicción al resultado
                // Formateamos el valor predicho a, por ejemplo, 4 decimales
                resultBuilder.append(String.format("%-10s | %-20.4f\n", (i + 1), predictedValue));
            }
        }

        // Si no se encontró ninguna instancia con valor objetivo desconocido, añadir un mensaje
        if (!foundInstancesToPredict) {
            resultBuilder.append("No se encontraron instancias con valor objetivo original desconocido (?) en este conjunto de datos.\n");
        }

        return resultBuilder.toString();
    }
}