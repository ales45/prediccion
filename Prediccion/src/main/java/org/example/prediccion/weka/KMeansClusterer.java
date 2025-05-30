package org.example.prediccion.weka; // Ajusta si tu paquete es diferente

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;
import java.text.DecimalFormat; // Aunque no lo usemos mucho ahora, es bueno tenerlo
import java.util.ArrayList;
import java.util.List;

public class KMeansClusterer {

    public String buildClustererAndGetAssignments(Instances originalData, int numClusters, String uiTargetColumnName) throws Exception {
        if (originalData.numInstances() == 0) {
            return "No hay datos para realizar el clustering K-Medias.";
        }
        if (numClusters <= 0) {
            throw new IllegalArgumentException("El número de clusters (K) debe ser mayor que 0.");
        }

        // 1. Identificar instancias con '?' en la uiTargetColumnName ANTES de cualquier modificación
        List<Integer> indicesOfInstancesWithMissingUiTarget = new ArrayList<>();
        int uiTargetAttributeIndex = -1;
        boolean filterOutputByUiTarget = false; // Determina si filtramos la salida de asignaciones

        if (uiTargetColumnName != null && !uiTargetColumnName.trim().isEmpty()) {
            Attribute potentialUiTargetAttribute = originalData.attribute(uiTargetColumnName.trim());
            if (potentialUiTargetAttribute != null) {
                uiTargetAttributeIndex = potentialUiTargetAttribute.index();
                filterOutputByUiTarget = true;
                for (int i = 0; i < originalData.numInstances(); i++) {
                    if (originalData.instance(i).isMissing(uiTargetAttributeIndex)) {
                        indicesOfInstancesWithMissingUiTarget.add(i);
                    }
                }
            } else {
                System.out.println("Advertencia KMC: La 'columna objetivo/clase' (" + uiTargetColumnName + ") para filtrar salida no se encontró. Se mostrarán todas las asignaciones.");
                filterOutputByUiTarget = false; // No se puede filtrar si la columna no existe
            }
        } else {
            System.out.println("INFO KMC: No se especificó 'columna objetivo/clase' para filtrar la salida de asignaciones. Se mostrarán todas las asignaciones.");
            filterOutputByUiTarget = false;
        }

        // Crear una copia para el preprocesamiento
        Instances dataForClustering = new Instances(originalData);

        // Lógica de preprocesamiento (remover IDs, remover atributo clase si estaba seteado, NominalToBinary)
        // Esta parte es importante y debe ser la misma que te funcionó antes para K-Medias
        Remove removeFilter = new Remove();
        ArrayList<String> attributesToRemoveIndicesStr = new ArrayList<>();
        int classIdxCurrent = dataForClustering.classIndex();
        if (uiTargetAttributeIndex != -1 && classIdxCurrent == uiTargetAttributeIndex) { // Si la columna UI era la clase
            if (!attributesToRemoveIndicesStr.contains(Integer.toString(uiTargetAttributeIndex + 1))) {
                attributesToRemoveIndicesStr.add(Integer.toString(uiTargetAttributeIndex + 1));
            }
        } else if (classIdxCurrent != -1) { // Si no hubo uiTarget para filtrar, pero había una clase, quitarla
            if (!attributesToRemoveIndicesStr.contains(Integer.toString(classIdxCurrent + 1))) {
                attributesToRemoveIndicesStr.add(Integer.toString(classIdxCurrent + 1));
            }
        }

        Attribute idAttributeNo = dataForClustering.attribute("No.");
        if (idAttributeNo != null) {
            String idxStr = Integer.toString(idAttributeNo.index() + 1);
            if (!attributesToRemoveIndicesStr.contains(idxStr)) attributesToRemoveIndicesStr.add(idxStr);
        }
        Attribute idAttributeNovia = dataForClustering.attribute("Novia");
        if (idAttributeNovia != null) {
            String idxStr = Integer.toString(idAttributeNovia.index() + 1);
            if(!attributesToRemoveIndicesStr.contains(idxStr)) attributesToRemoveIndicesStr.add(idxStr);
        }

        List<String> distinctAttributesToRemove = new ArrayList<>();
        for (String idxStr : attributesToRemoveIndicesStr) {
            if (!distinctAttributesToRemove.contains(idxStr)) distinctAttributesToRemove.add(idxStr);
        }

        if (!distinctAttributesToRemove.isEmpty()) {
            removeFilter.setAttributeIndices(String.join(",", distinctAttributesToRemove));
            removeFilter.setInvertSelection(false);
            removeFilter.setInputFormat(dataForClustering);
            dataForClustering = Filter.useFilter(dataForClustering, removeFilter);
        }
        dataForClustering.setClassIndex(-1); // Asegurar que no haya clase para clustering

        if (dataForClustering.numAttributes() == 0) {
            return "No quedaron atributos para el clustering K-Means después de los filtros iniciales (remoción de IDs/Clase).";
        }
        NominalToBinary nominalToBinaryFilter = new NominalToBinary();
        nominalToBinaryFilter.setInputFormat(dataForClustering);
        Instances numericData = Filter.useFilter(dataForClustering, nominalToBinaryFilter);

        if (numericData.numAttributes() == 0) {
            return "Después de convertir nominal a binario, no quedaron atributos para el clustering K-Means.";
        }
        // Fin del Preprocesamiento

        // Configurar y construir el clusterer SimpleKMeans
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(numClusters);
        kMeans.setPreserveInstancesOrder(true); // Importante para mapear asignaciones al orden original

        System.out.println("INFO KMC: Iniciando K-Medias con K=" + numClusters);
        kMeans.buildClusterer(numericData);

        int[] assignments = kMeans.getAssignments(); // Corresponden al orden de numericData

        // Construir el String de resultado enfocado en las asignaciones
        StringBuilder result = new StringBuilder();
        result.append("=== Resultados del Clustering K-Medias ===\n");
        result.append("Número de Clusters (K) especificado: ").append(numClusters).append("\n\n");

        if (filterOutputByUiTarget && uiTargetAttributeIndex != -1) {
            result.append("--- Asignación de Cluster para Instancias donde '").append(uiTargetColumnName).append("' era '?' ---\n");
            if (indicesOfInstancesWithMissingUiTarget.isEmpty()) {
                result.append("No se encontraron instancias con valor desconocido ('?') en la columna '").append(uiTargetColumnName).append("'.\n");
            } else {
                result.append(String.format("%-10s | %-15s\n", "Instancia#", "Cluster Asignado"));
                result.append("-------------------------------\n");
                // 'assignments' está en el orden de las instancias DESPUÉS de los filtros de atributos
                // pero como esos filtros no eliminan INSTANCIAS, el orden se mantiene.
                for (int originalIndex : indicesOfInstancesWithMissingUiTarget) {
                    if (originalIndex < assignments.length) { // Chequeo de seguridad
                        result.append(String.format("%-10s | %-15s\n", (originalIndex + 1), assignments[originalIndex]));
                    } else {
                        System.err.println("KMC: Índice original " + originalIndex + " fuera de rango para assignments de tamaño " + assignments.length);
                    }
                }
            }
        } else { // Mostrar todas las asignaciones si no se filtra
            result.append("--- Asignación de Cluster por Instancia ---\n");
            result.append(String.format("%-10s | %-15s\n", "Instancia#", "Cluster Asignado"));
            result.append("-------------------------------\n");
            for (int i = 0; i < assignments.length; i++) {
                // (i + 1) se refiere al número de instancia en el conjunto de datos procesado.
                // Si hubo filtros que eliminaron instancias (no es el caso aquí), esto necesitaría un mapeo más complejo.
                result.append(String.format("%-10s | %-15s\n", (i + 1), assignments[i]));
            }
        }

        // Si quieres añadir alguna otra información mínima, puedes hacerlo aquí.
        // Por ejemplo, el SSE podría ser útil:
        // DecimalFormat df = new DecimalFormat("#.####");
        // result.append("\nSuma de errores cuadráticos intra-cluster (SSE): ").append(df.format(kMeans.getSquaredError())).append("\n");

        return result.toString();
    }
}