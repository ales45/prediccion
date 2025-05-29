package org.example.prediccion.weka; // Ajusta si tu paquete es diferente

import weka.classifiers.trees.J48; // Clasificador para Árboles de Decisión (C4.5)
import weka.core.Instances;

public class DecisionTreePredictor implements WekaPredictor { // Implementa la interfaz

    @Override
    public String buildModelAndGetResults(Instances data) throws Exception {
        // Verificar que el atributo clase (columna objetivo) esté asignado en los datos
        if (data.classIndex() == -1) {
            throw new IllegalArgumentException("El atributo clase (target) no ha sido asignado en los datos de Weka (Instances). " +
                    "Asegúrate de llamar a data.setClassIndex() antes de pasar los datos al predictor.");
        }

        // 1. Crear una instancia del clasificador J48
        // J48 es la implementación en Weka del algoritmo C4.5, que genera árboles de decisión.
        J48 tree = new J48();

        // 2. Opciones del clasificador (Opcional)
        // Puedes experimentar con diferentes opciones para el árbol J48 más adelante.
        // Por ejemplo, para evitar la poda del árbol (lo que podría hacerlo más grande y específico para los datos de entrenamiento):
        // tree.setUnpruned(true);
        // Para cambiar el factor de confianza usado en la poda (valores más pequeños tienden a podar más):
        // tree.setConfidenceFactor(0.25f); // El valor por defecto es 0.25f
        // Para establecer el número mínimo de instancias que debe haber en una hoja del árbol:
        // tree.setMinNumObj(2); // El valor por defecto es 2

        // 3. Construir el clasificador (es decir, entrenar el modelo de Árbol de Decisión)
        // Esto utiliza los datos (`data`) que se le pasan para aprender las reglas.
        tree.buildClassifier(data);

        // 4. Devolver la representación del árbol
        // El método `toString()` de un clasificador J48 entrenado devuelve una
        // representación textual del árbol de decisión, incluyendo las reglas y la estructura.
        return tree.toString();
    }
}