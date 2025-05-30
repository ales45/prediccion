package org.example.prediccion.weka; // Ajusta este paquete a tu estructura

import weka.classifiers.trees.J48;    // El clasificador de árbol de decisión J48
import weka.core.Instance;           // Para referirse a una fila/instancia individual
import weka.core.Instances;          // Para el conjunto de datos de Weka

public class DecisionTreePredictor implements WekaPredictor { // Implementa tu interfaz

    @Override
    public String buildModelAndGetResults(Instances data) throws Exception {
        // Validación inicial: ¿Está asignada la columna clase (objetivo)?
        if (data.classIndex() == -1) {
            throw new IllegalArgumentException(
                    "El atributo clase (target) no ha sido asignado en los datos de Weka (Instances). " +
                            "Asegúrate de llamar a data.setClassIndex() antes de pasar los datos al predictor."
            );
        }

        // Validación inicial: ¿Hay datos para entrenar?
        if (data.numInstances() == 0) {
            return "No hay datos de entrenamiento para el Árbol de Decisión.";
        }

        // 1. Crear una instancia del clasificador J48
        J48 tree = new J48();

        // Opcional: Configurar parámetros del árbol J48 si lo deseas
        // Ejemplo: tree.setUnpruned(true); // Para no podar el árbol
        // Ejemplo: tree.setMinNumObj(5); // Número mínimo de objetos por hoja

        // 2. Construir el clasificador (entrenar el modelo) con los datos proporcionados
        // Weka J48 internamente maneja el entrenamiento usando solo las instancias
        // donde el atributo clase no está perdido (missing).
        tree.buildClassifier(data);

        // 3. Preparar el String de resultados
        StringBuilder resultBuilder = new StringBuilder();

        // Añadir la estructura del árbol de decisión al resultado
        resultBuilder.append("=== Árbol de Decisión J48 Construido ===\n");
        resultBuilder.append(tree.toString()).append("\n\n"); // El método toString() de J48 da la estructura del árbol

        // Añadir la sección para predicciones de instancias con clase original desconocida
        resultBuilder.append("=== Predicciones para Instancias con Clase Original Desconocida (?) ===\n");
        resultBuilder.append(String.format("%-10s | %-20s\n", "Instancia#", "Clase Predicha"));
        resultBuilder.append("---------------------------------------\n");

        boolean foundInstancesToPredict = false;

        // 4. Iterar sobre cada instancia en los datos para hacer una predicción
        //    SOLO para aquellas donde la clase original era desconocida/perdida.
        for (int i = 0; i < data.numInstances(); i++) {
            Instance currentInstance = data.instance(i);

            // Verificar si la clase original de esta instancia estaba marcada como perdida
            if (currentInstance.classIsMissing()) {
                foundInstancesToPredict = true;

                // Predecir la clase para la instancia actual usando el árbol entrenado
                double predictedClassIndex = tree.classifyInstance(currentInstance);
                // Convertir el índice de la clase predicha a su valor de etiqueta String
                String predictedClassValue = currentInstance.classAttribute().value((int) predictedClassIndex);

                // Añadir la información de la instancia y su predicción al resultado
                resultBuilder.append(String.format("%-10s | %-20s\n", (i + 1), predictedClassValue));
            }
        }

        // Si no se encontró ninguna instancia con clase desconocida, añadir un mensaje
        if (!foundInstancesToPredict) {
            resultBuilder.append("No se encontraron instancias con clase original desconocida (?) en este conjunto de datos para realizar predicciones específicas.\n");
        }

        return resultBuilder.toString();
    }
}