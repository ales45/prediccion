package org.example.prediccion.weka; // Ajusta si tu paquete es diferente

import weka.classifiers.lazy.IBk; // Clasificador para KNN
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch; // Ejemplo de un algoritmo de búsqueda de vecinos

public class KnnPredictor implements WekaPredictor {

    private int kNeighbors = 3; // Valor por defecto para K (número de vecinos)

    // Constructor por defecto que usa K=3
    public KnnPredictor() {
    }

    // Constructor opcional para especificar K
    public KnnPredictor(int k) {
        if (k > 0) {
            this.kNeighbors = k;
        } else {
            System.out.println("Advertencia: El valor de K para KNN debe ser mayor que 0. Usando K=" + this.kNeighbors + " por defecto.");
        }
    }

    @Override
    public String buildModelAndGetResults(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new IllegalArgumentException("El atributo clase (target) no ha sido asignado en los datos de Weka (Instances) para KNN.");
        }

        if (data.numInstances() == 0) {
            return "No hay datos de entrenamiento para el clasificador KNN.";
        }

        // Asegurarse de que K no sea mayor que el número de instancias de entrenamiento
        // (Weka IBk puede tener problemas o dar advertencias si K es muy grande comparado con los datos)
        int effectiveK = this.kNeighbors;
        if (data.numInstances() < this.kNeighbors) {
            System.out.println("Advertencia: K (" + this.kNeighbors + ") es mayor que el número de instancias de entrenamiento (" + data.numInstances() + "). Ajustando K a " + data.numInstances());
            effectiveK = data.numInstances(); // O podrías lanzar un error o usar k=1
            if (effectiveK == 0 && data.numInstances() > 0) effectiveK = 1; // Si hay datos, K debe ser al menos 1
        }
        if (effectiveK == 0 && data.numInstances() == 0) { // Si no hay datos, no se puede hacer nada.
            return "No hay suficientes datos para KNN (K=" + this.kNeighbors + ", Instancias=" + data.numInstances() + ")";
        }


        // 1. Crear una instancia del clasificador IBk (KNN)
        IBk knn = new IBk();

        // 2. Configurar el número de vecinos (K)
        knn.setKNN(effectiveK);

        // Opcional: Configurar el algoritmo de búsqueda de vecinos.
        // LinearNNSearch es simple y bueno para datasets pequeños.
        // Para datasets más grandes, podrías explorar KDTree, BallTree, etc.
        // LinearNNSearch lnn = new LinearNNSearch(data);
        // knn.setNearestNeighbourSearchAlgorithm(lnn);

        // 3. "Construir" el clasificador. Para KNN, esto principalmente implica
        // almacenar los datos de entrenamiento, ya que es un learner perezoso.
        knn.buildClassifier(data);

        // 4. Devolver información sobre el clasificador KNN configurado.
        // A diferencia de un árbol de decisión, KNN no tiene un "modelo" visible en forma de reglas.
        // El método toString() de IBk usualmente muestra las opciones con las que fue configurado.
        StringBuilder result = new StringBuilder();
        result.append("Clasificador KNN (IBk) configurado y 'entrenado' (datos almacenados).\n");
        result.append("Opciones del clasificador: ").append(String.join(" ", knn.getOptions())).append("\n");
        result.append("Número de vecinos (K) utilizado: ").append(knn.getKNN()).append("\n");
        result.append("Algoritmo de búsqueda de vecinos: ").append(knn.getNearestNeighbourSearchAlgorithm().getClass().getSimpleName()).append("\n");
        result.append("Número de instancias de entrenamiento: ").append(data.numInstances()).append("\n");
        result.append("\nNOTA: KNN es un 'lazy learner'. No construye un modelo explícito como un árbol.\n");
        result.append("La predicción para nuevas instancias se basará en la distancia a las ").append(knn.getKNN()).append(" instancias más cercanas en estos datos de entrenamiento.");

        // También podrías considerar hacer una evaluación rápida aquí, como predecir sobre el mismo
        // conjunto de entrenamiento para ver cómo se comporta, pero eso es más para evaluación.
        // Por ahora, la información de configuración es suficiente como "resultado del modelo".

        return result.toString();
        // Alternativamente, para una salida más técnica de Weka:
        // return knn.toString();
    }
}