package org.example.prediccion.weka; // Ajusta este paquete a tu estructura

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays; // Necesario para trabajar con streams y arrays
import java.util.stream.Collectors; // Necesario para Collectors.joining

public class DecisionTreePredictor implements WekaPredictor {

    @Override
    public String buildModelAndGetResults(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalArgumentException(
                    "El atributo clase (target) no ha sido asignado en los datos de Weka (Instances). " +
                            "Asegúrate de llamar a data.setClassIndex() antes de pasar los datos al predictor."
            );
        }

        if (data.numInstances() == 0) {
            return "No hay datos de entrenamiento para el Árbol de Decisión.";
        }

        J48 tree = new J48();
        tree.buildClassifier(data); // Entrenar el modelo

        StringBuilder resultBuilder = new StringBuilder();
        resultBuilder.append("=== Árbol de Decisión J48 Construido ===\n");

        // Obtener el string del árbol y filtrar las líneas no deseadas
        String fullTreeString = tree.toString();
        String[] lines = fullTreeString.split("\n");
        String filteredTreeString = Arrays.stream(lines)
                .filter(line -> !line.trim().startsWith("Number of Leaves") &&
                        !line.trim().startsWith("Size of the tree"))
                .collect(Collectors.joining("\n"));

        resultBuilder.append(filteredTreeString).append("\n\n"); // Añadir el string del árbol filtrado

        // Sección para predicciones de instancias con clase original desconocida
        resultBuilder.append("=== Predicciones para Instancias con Clase Original Desconocida (?) ===\n");
        resultBuilder.append(String.format("%-10s | %-20s\n", "Instancia#", "Clase Predicha"));
        resultBuilder.append("---------------------------------------\n");

        boolean foundInstancesToPredict = false;
        for (int i = 0; i < data.numInstances(); i++) {
            Instance currentInstance = data.instance(i);
            if (currentInstance.classIsMissing()) { // Solo predecir si la clase original era '?'
                foundInstancesToPredict = true;
                double predictedClassIndex = tree.classifyInstance(currentInstance);
                String predictedClassValue = currentInstance.classAttribute().value((int) predictedClassIndex);
                resultBuilder.append(String.format("%-10s | %-20s\n", (i + 1), predictedClassValue));
            }
        }

        if (!foundInstancesToPredict) {
            resultBuilder.append("No se encontraron instancias con clase original desconocida (?) en este conjunto de datos.\n");
        }

        return resultBuilder.toString();
    }
}