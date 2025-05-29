package org.example.prediccion.weka; // Ajusta si tu paquete es diferente

import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.Attribute; // Para verificar el tipo de atributo

public class LinearRegressionPredictor implements WekaPredictor {

    @Override
    public String buildModelAndGetResults(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalArgumentException("El atributo clase (target) no ha sido asignado en los datos de Weka (Instances).");
        }

        // Verificar que el atributo clase sea numérico para Regresión Lineal
        if (!data.classAttribute().isNumeric()) {
            throw new IllegalArgumentException("El atributo clase (target) debe ser NUMÉRICO para Regresión Lineal. " +
                    "El atributo '" + data.classAttribute().name() + "' es de tipo " +
                    Attribute.typeToString(data.classAttribute().type()) + ".");
        }

        if (data.numInstances() == 0) {
            return "No hay datos de entrenamiento para el modelo de Regresión Lineal.";
        }

        // 1. Crear una instancia del clasificador LinearRegression
        LinearRegression lr = new LinearRegression();

        // 2. Opciones del clasificador (opcionales)
        // Por ejemplo, para seleccionar atributos usando M5' method:
        // lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_M5S, LinearRegression.TAGS_SELECTION));
        // Para no intentar seleccionar atributos:
        // lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));
        // Por defecto, Weka intentará alguna selección de atributos.

        // 3. Construir el clasificador (entrenar el modelo)
        lr.buildClassifier(data);

        // 4. Devolver la representación del modelo de regresión
        // El método toString() de LinearRegression muestra la ecuación de regresión,
        // los coeficientes y algunas estadísticas.
        return lr.toString();
    }
}