package org.example.prediccion.weka; // Ajusta si tu paquete es diferente

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;

public class KMeansClusterer { // No implementa WekaPredictor directamente, tiene otra firma

    public String buildClustererAndGetAssignments(Instances data, int numClusters) throws Exception {
        if (data.numInstances() == 0) {
            return "No hay datos para realizar el clustering K-Means.";
        }
        if (numClusters <= 0) {
            throw new IllegalArgumentException("El número de clusters debe ser mayor que 0.");
        }

        // Crear una copia de los datos para no modificar el original si se usa en otro lado
        Instances dataForClustering = new Instances(data);

        // 1. K-Means es sensible al atributo clase. Si está seteado, algunos clusterers lo usan
        // o dan error. Para SimpleKMeans, es mejor no tenerlo o quitarlo.
        // También quitaremos atributos que no son útiles para clustering como IDs.
        // Asumiremos que "No." y "Novia" (si existen) son los primeros y podrían ser IDs.

        Remove removeFilter = new Remove();
        ArrayList<String> indicesToRemove = new ArrayList<>();

        if (dataForClustering.classIndex() >= 0) {
            // Añadir el índice de la clase para removerlo (+1 porque setAttributeIndices es base 1)
            indicesToRemove.add(Integer.toString(dataForClustering.classIndex() + 1));
            dataForClustering.setClassIndex(-1); // Quitar la asignación de clase
        }

        // Ejemplo: Si "No." es el primer atributo y "Novia" el segundo Y queremos quitarlos
        // (Ajusta esto según los nombres reales de tus columnas de ID)
        Attribute idAttributeNo = dataForClustering.attribute("No.");
        if (idAttributeNo != null) {
            indicesToRemove.add(Integer.toString(idAttributeNo.index() + 1));
        }
        Attribute idAttributeNovia = dataForClustering.attribute("Novia");
        if (idAttributeNovia != null) {
            indicesToRemove.add(Integer.toString(idAttributeNovia.index() + 1));
        }

        if (!indicesToRemove.isEmpty()) {
            removeFilter.setAttributeIndices(String.join(",", indicesToRemove));
            removeFilter.setInputFormat(dataForClustering); // Importante antes de usar el filtro
            dataForClustering = Filter.useFilter(dataForClustering, removeFilter);
        }

        // 2. K-Means funciona mejor con atributos numéricos.
        // Convertir atributos Nominales restantes a Binarios.
        NominalToBinary nominalToBinaryFilter = new NominalToBinary();
        // Necesitamos especificar qué atributos NO convertir (los numéricos).
        // Por ahora, vamos a asumir que todos los que queden se intentan convertir.
        // Una mejor aproximación sería iterar y construir el rango de los nominales.
        // nominalToBinaryFilter.setAttributeIndices("first-last"); // Opción por defecto es convertir todos los nominales

        // Verificar si quedan atributos después de la eliminación
        if (dataForClustering.numAttributes() == 0) {
            return "No quedaron atributos para el clustering K-Means después de los filtros iniciales.";
        }

        nominalToBinaryFilter.setInputFormat(dataForClustering);
        Instances numericData = Filter.useFilter(dataForClustering, nominalToBinaryFilter);

        if (numericData.numAttributes() == 0) {
            return "Después de convertir nominal a binario, no quedaron atributos para el clustering K-Means.";
        }

        // 3. Configurar y construir el clusterer SimpleKMeans
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(numClusters); // Establecer el número de clusters K
        kMeans.setPreserveInstancesOrder(true); // Para que las asignaciones coincidan con el orden original
        // kMeans.setDistanceFunction(new weka.core.EuclideanDistance()); // Por defecto es Euclidiana

        kMeans.buildClusterer(numericData); // Construir clusters sobre los datos preprocesados

        // 4. Obtener las asignaciones de cluster para cada instancia
        int[] assignments = kMeans.getAssignments();

        StringBuilder result = new StringBuilder();
        result.append("Resultados del Clustering K-Means (").append(numClusters).append(" clusters):\n\n");
        result.append("Asignación de Cluster por Instancia (fila del CSV original):\n");
        for (int i = 0; i < assignments.length; i++) {
            result.append("Instancia ").append(i + 1).append(" -> Cluster ").append(assignments[i]).append("\n");
        }

        // Opcional: Mostrar información sobre los centroides de los clusters
        // Instances centroids = kMeans.getClusterCentroids();
        // result.append("\nCentroides de los Clusters (en el espacio transformado):\n").append(centroids.toString());

        result.append("\nInformación adicional del clusterer SimpleKMeans:\n");
        result.append(kMeans.toString());


        return result.toString();
    }
}