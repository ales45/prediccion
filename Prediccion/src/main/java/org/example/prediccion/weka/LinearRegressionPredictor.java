package org.example.prediccion.weka; // Ajusta este paquete a tu estructura

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag; // <--- IMPORTANTE: Añadir este import

public class LinearRegressionPredictor implements WekaPredictor {

    @Override
    public String buildModelAndGetResults(Instances data) throws Exception {
        // Validación: ¿Está asignada la columna clase (objetivo)?
        if (data.classIndex() == -1) {
            throw new IllegalArgumentException(
                    "El atributo clase (target) no ha sido asignado en los datos de Weka (Instances) para Regresión Lineal."
            );
        }

        // Validación: ¿Es numérica la columna clase (objetivo)?
        if (!data.classAttribute().isNumeric()) {
            throw new IllegalArgumentException(
                    "El atributo clase (target) debe ser NUMÉRICO para Regresión Lineal. " +
                            "El atributo '" + data.classAttribute().name() + "' es de tipo " +
                            Attribute.typeToString(data.classAttribute().type()) + "."
            );
        }

        // Validación: ¿Hay datos para entrenar?
        if (data.numInstances() == 0) {
            return "No hay datos de entrenamiento para el modelo de Regresión Lineal.";
        }

        // 1. Crear una instancia del clasificador LinearRegression
        LinearRegression lr = new LinearRegression();

        // 2. Configurar opciones del clasificador (opcional)
        // Establecer el método de selección de atributos a NINGUNO para usar todos los atributos.
        // Esto podría dar una "fórmula más a lo largo" si Weka estaba descartando alguno por defecto.
        lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));

        // Otras opciones que podrías explorar (comentadas por ahora):
        // lr.setEliminateColinearAttributes(false); // Para no eliminar atributos colineales
        // lr.setRidge(1.0E-8); // Valor de Ridge por defecto

        // 3. Construir el clasificador (entrenar el modelo)
        lr.buildClassifier(data);

        // 4. Preparar el String de resultados
        StringBuilder resultBuilder = new StringBuilder();

        // Añadir la información del modelo de regresión (ecuación, coeficientes, etc.)
        resultBuilder.append("=== Modelo de Regresión Lineal Construido ===\n");
        resultBuilder.append(lr.toString()).append("\n\n"); // lr.toString() contiene la fórmula y estadísticas

        // Añadir la sección para predicciones de instancias con valor objetivo original desconocido
        resultBuilder.append("=== Predicciones para Instancias con Valor Objetivo Originalmente Desconocido (?) ===\n");
        resultBuilder.append(String.format("%-10s | %-20s\n", "Instancia#", "Valor Predicho"));
        resultBuilder.append("---------------------------------------\n");

        boolean foundInstancesToPredict = false;

        // 5. Iterar sobre cada instancia para hacer una predicción
        //    SOLO para aquellas donde el valor objetivo original era desconocido/perdido.
        for (int i = 0; i < data.numInstances(); i++) {
            Instance currentInstance = data.instance(i);

            // Verificar si el valor de la clase original de esta instancia estaba marcado como perdido
            if (currentInstance.classIsMissing()) {
                foundInstancesToPredict = true;

                // Predecir el valor para la instancia actual usando el modelo entrenado
                double predictedValue = lr.classifyInstance(currentInstance); // Devuelve un valor numérico

                // Añadir la información de la instancia y su predicción al resultado
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