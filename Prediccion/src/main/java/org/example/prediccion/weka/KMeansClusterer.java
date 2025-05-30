package org.example.prediccion.weka;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;
import java.util.ArrayList;
import java.util.List;

public class KMeansClusterer { // Si tienes WekaClusterer interfaz, impleméntala

    // Modificamos el método para aceptar el targetColumnName original de la UI
    public String buildClustererAndGetAssignments(Instances originalData, int numClusters, String uiTargetColumnName) throws Exception {
        if (originalData.numInstances() == 0) {
            return "No hay datos para realizar el clustering K-Means.";
        }
        if (numClusters <= 0) {
            throw new IllegalArgumentException("El número de clusters debe ser mayor que 0.");
        }

        // 1. Identificar instancias con '?' en la uiTargetColumnName ANTES de cualquier modificación de atributos
        List<Integer> indicesOfInstancesWithMissingUiTarget = new ArrayList<>();
        int uiTargetAttributeIndex = -1; // Índice de la uiTargetColumnName en originalData

        if (uiTargetColumnName != null && !uiTargetColumnName.trim().isEmpty()) {
            Attribute potentialUiTargetAttribute = originalData.attribute(uiTargetColumnName.trim());
            if (potentialUiTargetAttribute != null) {
                uiTargetAttributeIndex = potentialUiTargetAttribute.index();
                for (int i = 0; i < originalData.numInstances(); i++) {
                    if (originalData.instance(i).isMissing(uiTargetAttributeIndex)) {
                        indicesOfInstancesWithMissingUiTarget.add(i); // Guardamos el índice de la instancia original
                    }
                }
            } else {
                System.out.println("Advertencia: La 'columna objetivo/clase' especificada (" + uiTargetColumnName + ") no se encontró en los datos para el filtrado de salida de K-Medias. Se mostrarán todas las asignaciones.");
            }
        } else {
            System.out.println("INFO: No se especificó 'columna objetivo/clase' para el filtrado de salida de K-Medias. Se mostrarán todas las asignaciones.");
        }


        // Crear una copia para el preprocesamiento
        Instances dataForClustering = new Instances(originalData);

        // 2. Preprocesamiento: Quitar el atributo clase (si está formalmente asignado en Weka)
        // y otros atributos de ID antes de convertir a binario.
        Remove removeFilter = new Remove();
        ArrayList<String> attributesToRemoveIndices = new ArrayList<>();

        // Si la uiTargetColumnName fue la que se usó para setClassIndex en originalData
        // Y queremos quitarla explícitamente
        if (originalData.classIndex() != -1 && originalData.classIndex() == uiTargetAttributeIndex && uiTargetAttributeIndex != -1) {
            if(!attributesToRemoveIndices.contains(Integer.toString(originalData.classIndex() + 1))) {
                attributesToRemoveIndices.add(Integer.toString(originalData.classIndex() + 1));
            }
        }
        // También quitamos "No." y "Novia" si existen, como antes
        Attribute idAttributeNo = dataForClustering.attribute("No.");
        if (idAttributeNo != null && !attributesToRemoveIndices.contains(Integer.toString(idAttributeNo.index() + 1))) {
            attributesToRemoveIndices.add(Integer.toString(idAttributeNo.index() + 1));
        }
        Attribute idAttributeNovia = dataForClustering.attribute("Novia");
        if (idAttributeNovia != null && !attributesToRemoveIndices.contains(Integer.toString(idAttributeNovia.index() + 1))) {
            attributesToRemoveIndices.add(Integer.toString(idAttributeNovia.index() + 1));
        }

        if (!attributesToRemoveIndices.isEmpty()) {
            removeFilter.setAttributeIndices(String.join(",", attributesToRemoveIndices));
            removeFilter.setInvertSelection(false);
            removeFilter.setInputFormat(dataForClustering); // Importante
            dataForClustering = Filter.useFilter(dataForClustering, removeFilter);
        }
        // Después de quitar atributos, el classIndex se pierde o cambia, así que lo reseteamos.
        dataForClustering.setClassIndex(-1);


        // 3. Convertir atributos Nominales restantes a Binarios
        if (dataForClustering.numAttributes() == 0) {
            return "No quedaron atributos para el clustering K-Means después de los filtros iniciales.";
        }
        NominalToBinary nominalToBinaryFilter = new NominalToBinary();
        nominalToBinaryFilter.setInputFormat(dataForClustering);
        Instances numericData = Filter.useFilter(dataForClustering, nominalToBinaryFilter);

        if (numericData.numAttributes() == 0) {
            return "Después de convertir nominal a binario, no quedaron atributos para el clustering K-Means.";
        }

        // 4. Configurar y construir el clusterer SimpleKMeans
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(numClusters);
        kMeans.setPreserveInstancesOrder(true); // Importante para mapear asignaciones
        kMeans.buildClusterer(numericData);

        // 5. Obtener las asignaciones de cluster para cada instancia
        int[] assignments = kMeans.getAssignments(); // Corresponden al orden de numericData (y dataForClustering)

        // 6. Construir el String de resultado
        StringBuilder result = new StringBuilder();
        result.append("=== Modelo K-Means Construido ===\n");
        result.append(kMeans.toString()).append("\n\n"); // Información general del modelo (SSE, centroides, etc.)

        result.append("=== Asignaciones de Cluster para Instancias con '").append(uiTargetColumnName == null || uiTargetColumnName.trim().isEmpty() ? "Cualquier Valor" : uiTargetColumnName).append("' Originalmente Desconocido (?) ===\n");
        if (uiTargetColumnName != null && !uiTargetColumnName.trim().isEmpty() && uiTargetAttributeIndex != -1) {
            if (indicesOfInstancesWithMissingUiTarget.isEmpty()) {
                result.append("No se encontraron instancias con valor desconocido ('?') en la columna '").append(uiTargetColumnName).append("'.\n");
            } else {
                result.append(String.format("%-10s | %-15s\n", "Instancia#", "Cluster Asignado"));
                result.append("-------------------------------\n");
                for (int originalIndex : indicesOfInstancesWithMissingUiTarget) {
                    // Asumimos que el orden de las instancias se ha preservado a través de los filtros
                    // que no eliminan filas (Remove de atributos y NominalToBinary no eliminan filas)
                    // por lo que assignments[originalIndex] es la asignación correcta.
                    result.append(String.format("%-10s | %-15s\n", (originalIndex + 1), assignments[originalIndex]));
                }
            }
        } else {
            result.append("No se especificó una 'Columna Objetivo/Clase' para filtrar las asignaciones, o no se encontró.\n");
            result.append("Mostrando todas las asignaciones como referencia:\n");
            result.append(String.format("%-10s | %-15s\n", "Instancia#", "Cluster Asignado"));
            result.append("-------------------------------\n");
            for (int i = 0; i < assignments.length; i++) {
                result.append(String.format("%-10s | %-15s\n", (i + 1), assignments[i]));
            }
        }
        return result.toString();
    }
}