package org.example.prediccion.service; // Ajusta si tu paquete es diferente

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.example.prediccion.weka.KMeansClusterer;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

// Imports de Weka
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.WekaException;

// Imports para las nuevas clases de predicción (asegúrate que el paquete sea correcto)
import org.example.prediccion.weka.DecisionTreePredictor;
import org.example.prediccion.weka.WekaPredictor;
import org.example.prediccion.weka.LinearRegressionPredictor; // Ajusta el paquete si es necesario

// Cuando crees los otros predictores, los importarás aquí también:
// import org.example.prediccion.weka.KnnPredictor;
// import org.example.prediccion.weka.LinearRegressionPredictor;

import java.io.BufferedReader; // Para el log de diagnóstico si lo mantienes
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors; // Para el log de atributos

@Service
public class PredictionService {

// Este método va DENTRO de tu clase PredictionService.java

    private Instances convertCsvToWekaInstances(List<CSVRecord> csvRecords, List<String> rawHeaders, String targetColumnName) throws Exception {
        if (csvRecords.isEmpty() && (rawHeaders == null || rawHeaders.isEmpty())) {
            throw new IllegalArgumentException("No hay encabezados ni registros de datos en el CSV para procesar.");
        }
        if (csvRecords.isEmpty() && !rawHeaders.isEmpty()) {
            System.out.println("Advertencia en convertCsvToWekaInstances: No hay registros de datos en el CSV, solo encabezados. Weka necesita datos para la mayoría de los modelos.");
            // Podríamos lanzar una excepción aquí si se requieren datos, por ejemplo:
            // throw new IllegalArgumentException("Se requieren registros de datos para construir el modelo de Weka.");
        }

        // 1. Definir los atributos (columnas) para Weka, omitiendo vacíos y duplicados
        ArrayList<Attribute> wekaAttributes = new ArrayList<>();
        List<String> finalWekaHeaderNames = new ArrayList<>(); // Nombres de los atributos que realmente se añaden a Weka
        List<Integer> originalCsvColumnIndices = new ArrayList<>(); // Índice original en el CSV de cada atributo Weka

        for (int i = 0; i < rawHeaders.size(); i++) {
            String headerName = rawHeaders.get(i).trim(); // Quitar espacios

            if (headerName.isEmpty()) {
                System.out.println("INFO: Omitiendo encabezado vacío encontrado en la posición CSV original " + i);
                continue; // Saltar este encabezado vacío, no se creará atributo en Weka
            }

            // Verificar si este nombre de encabezado (después de trim) ya fue añadido a Weka
            if (finalWekaHeaderNames.contains(headerName)) {
                System.out.println("ADVERTENCIA: Nombre de encabezado duplicado '" + headerName + "' encontrado en la posición CSV original " + i + ". Se omitirá este duplicado para Weka.");
                continue; // Saltar este encabezado duplicado
            }

            // Si el encabezado no está vacío y no es duplicado, lo procesamos
            finalWekaHeaderNames.add(headerName);
            originalCsvColumnIndices.add(i); // Guardar el índice original del CSV para este atributo Weka

            // Lógica simple de inferencia de tipo (igual que antes)
            if (headerName.equalsIgnoreCase("No.")) {
                wekaAttributes.add(new Attribute(headerName));
            } else {
                ArrayList<String> uniqueValues = new ArrayList<>();
                // Para obtener los valores únicos, iteramos sobre los registros usando el índice original 'i'
                for (CSVRecord record : csvRecords) {
                    if (i < record.size()) { // Asegurar que el registro tenga esta columna (por el índice original)
                        String value = record.get(i).trim();
                        if (!value.isEmpty() && !value.equals("?") && !uniqueValues.contains(value)) {
                            uniqueValues.add(value);
                        }
                    }
                }
                if (uniqueValues.isEmpty()) {
                    uniqueValues.add("(valor_desconocido_o_vacio)"); // Placeholder si no hay valores
                    System.out.println("Advertencia: El atributo nominal '" + headerName + "' no tiene valores únicos o solo valores perdidos. Se añadió un valor placeholder.");
                }
                wekaAttributes.add(new Attribute(headerName, uniqueValues));
            }
        }

        if (wekaAttributes.isEmpty()) {
            throw new IllegalArgumentException("No se pudieron definir atributos válidos para Weka a partir de los encabezados del CSV.");
        }

        // 2. Crear el objeto Instances con los atributos filtrados
        Instances data = new Instances("DatasetCSV", wekaAttributes, csvRecords.size());

        // 3. Establecer el atributo clase (target column) usando los nombres de atributos finales de Weka
        int targetWekaAttributeIndex = -1;
        for (int i = 0; i < finalWekaHeaderNames.size(); i++) {
            if (finalWekaHeaderNames.get(i).equalsIgnoreCase(targetColumnName)) {
                targetWekaAttributeIndex = i;
                break;
            }
        }

        if (targetWekaAttributeIndex == -1) {
            throw new IllegalArgumentException("La columna objetivo '" + targetColumnName + "' no se encontró entre los encabezados válidos y no vacíos procesados para Weka. Encabezados Weka: " + finalWekaHeaderNames);
        }
        data.setClassIndex(targetWekaAttributeIndex); // Asignar usando el índice dentro de la lista de atributos de Weka

        // 4. Llenar las instancias con los datos
        for (CSVRecord csvRecord : csvRecords) {
            DenseInstance instance = new DenseInstance(wekaAttributes.size()); // Usar el número de atributos de Weka
            instance.setDataset(data);

            // Iterar sobre los atributos que SÍ se añadieron a Weka
            for (int wekaAttrIdx = 0; wekaAttrIdx < wekaAttributes.size(); wekaAttrIdx++) {
                Attribute attribute = data.attribute(wekaAttrIdx); // Atributo actual de Weka
                int originalCsvColIdx = originalCsvColumnIndices.get(wekaAttrIdx); // Obtener su índice original en el CSV

                String rawValue = "";
                if (originalCsvColIdx < csvRecord.size()) { // Leer del CSV usando el índice original
                    rawValue = csvRecord.get(originalCsvColIdx).trim();
                }

                if (rawValue.isEmpty() || rawValue.equals("?")) {
                    instance.setMissing(attribute);
                } else {
                    try {
                        if (attribute.isNumeric()) {
                            instance.setValue(attribute, Double.parseDouble(rawValue));
                        } else if (attribute.isNominal()) {
                            instance.setValue(attribute, rawValue);
                        } else { // String attributes, Date, etc. (simplificado)
                            instance.setValue(attribute, rawValue);
                        }
                    } catch (NumberFormatException e_num) {
                        System.err.println("Advertencia: No se pudo parsear '" + rawValue + "' como número para '" + attribute.name() + "'. Perdido. " + e_num.getMessage());
                        instance.setMissing(attribute);
                    } catch (IllegalArgumentException e_nominal) {
                        System.err.println("Advertencia: Valor '" + rawValue + "' no permitido para nominal '" + attribute.name() + "'. Perdido. " + e_nominal.getMessage());
                        instance.setMissing(attribute);
                    }
                }
            }
            data.add(instance);
        }
        return data;
    }


    public String processCsvAndPredict(MultipartFile file, String targetColumnName, String algorithm,Integer numClusters) throws Exception {
        System.out.println("Servicio: Procesando archivo: " + file.getOriginalFilename());
        System.out.println("Servicio: Columna objetivo: " + targetColumnName);
        System.out.println("Servicio: Algoritmo seleccionado: " + algorithm);

        // Código de diagnóstico temporal (opcional)
        try (BufferedReader initialReader = new BufferedReader(new InputStreamReader(file.getInputStream(), StandardCharsets.UTF_8))) {
            String firstLine = initialReader.readLine();
            System.out.println("LOG DIAGNÓSTICO: Primera línea del CSV leída directamente: \"" + firstLine + "\"");
        } catch (Exception e) {
            System.err.println("LOG DIAGNÓSTICO: Error al leer la primera línea para diagnóstico: " + e.getMessage());
        }
        // Fin código de diagnóstico

        List<CSVRecord> records;
        List<String> headers;
        long recordCount = 0;

        try (Reader reader = new InputStreamReader(file.getInputStream(), StandardCharsets.UTF_8); // Vuelve a abrir el stream
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.builder()
                     .setHeader()                 // <--- Usando la configuración que te funcionó
                     .setSkipHeaderRecord(true)   // <--- Usando la configuración que te funcionó
                     .setTrim(true)
                     .setIgnoreEmptyLines(true)
                     .setAllowMissingColumnNames(true)
                     .build())) {

            headers = csvParser.getHeaderNames();
            System.out.println("Encabezados del CSV (Apache Commons CSV): " + headers);

            if (headers == null || headers.isEmpty()) {
                throw new IllegalArgumentException("El CSV no contiene encabezados o está vacío.");
            }
            if (!headers.contains(targetColumnName)) {
                throw new IllegalArgumentException("La columna objetivo especificada '" + targetColumnName + "' no se encuentra en los encabezados del CSV leídos por Commons CSV: " + headers);
            }

            records = csvParser.getRecords();
            recordCount = records.size();

            System.out.println("Número total de registros de datos (después del encabezado): " + recordCount);
            if (recordCount == 0) {
                throw new IllegalArgumentException("El archivo CSV no contiene registros de datos (filas) para entrenar el modelo, aunque los encabezados fueron leídos.");
            }

            Instances wekaData = convertCsvToWekaInstances(records, headers, targetColumnName);
            System.out.println("Datos convertidos a formato Weka. Número de instancias: " + wekaData.numInstances() + ", Número de atributos: " + wekaData.numAttributes());
            if (wekaData.classIndex() != -1) { // Solo imprimir si el índice de clase es válido
                System.out.println("Atributo clase asignado en Weka: " + wekaData.classAttribute().name() + " (Índice: " + wekaData.classIndex() + ")");
            } else {
                System.err.println("Error: El atributo clase no fue asignado correctamente en Weka Instances.");
                throw new IllegalStateException("El atributo clase no fue asignado correctamente en Weka Instances después de la conversión.");
            }


            WekaPredictor predictor; // Para los algoritmos de predicción supervisada
            String resultOutput;    // Variable para el resultado del algoritmo

            switch (algorithm.toLowerCase()) {
                case "decisiontree":
                    // Asegurarse de que el índice de clase esté asignado para algoritmos supervisados
                    if (wekaData.classIndex() == -1) {
                        throw new IllegalStateException("El atributo clase (target) no ha sido asignado para el Árbol de Decisión. " +
                                "Verifica que la targetColumn '" + targetColumnName + "' sea válida y se haya procesado correctamente.");
                    }
                    predictor = new DecisionTreePredictor();
                    resultOutput = predictor.buildModelAndGetResults(wekaData);
                    break;

                case "knn":
                    // TODO: Implementar KnnPredictor y descomentar/modificar las siguientes líneas
                    // if (wekaData.classIndex() == -1) {
                    //     throw new IllegalStateException("El atributo clase (target) no ha sido asignado para KNN.");
                    // }
                    // predictor = new KnnPredictor();
                    // resultOutput = predictor.buildModelAndGetResults(wekaData);
                    resultOutput = "Predicción con KNN (PENDIENTE DE IMPLEMENTAR EN CLASE SEPARADA)";
                    break;

                case "linearregression":
                    if (wekaData.classIndex() == -1) {
                        throw new IllegalStateException("El atributo clase (target) no ha sido asignado para Regresión Lineal.");
                    }
                    // La verificación de si la clase es numérica se hace ahora dentro de LinearRegressionPredictor,
                    // pero también es bueno tenerla aquí como una guarda temprana si prefieres.
                    // if (!wekaData.classAttribute().isNumeric()) {
                    //     throw new IllegalStateException("El atributo clase (target) debe ser numérico para Regresión Lineal. Atributo actual: '" + wekaData.classAttribute().name() + "' es de tipo " + Attribute.typeToString(wekaData.classAttribute().type()));
                    // }
                    predictor = new LinearRegressionPredictor();
                    resultOutput = predictor.buildModelAndGetResults(wekaData);
                    break;


                case "kmedias":
                    if (numClusters == null || numClusters <= 0) {
                        throw new IllegalArgumentException("Para K-Medias, se requiere un número de clusters (numClusters) válido y mayor que 0.");
                    }
                    KMeansClusterer kMeansImplementation = new KMeansClusterer();
                    resultOutput = kMeansImplementation.buildClustererAndGetAssignments(wekaData, numClusters);
                    break;

                case "kmodas":
                    if (numClusters == null || numClusters <= 0) {
                        throw new IllegalArgumentException("Para K-Modas, se requiere un número de clusters (numClusters) válido y mayor que 0.");
                    }
                    // TODO: Implementar KModesClusterer (si se decide) o investigar alternativas.
                    // int numClustersForKModes = 3; // Ejemplo
                    resultOutput = "Clustering con K-Modas para " + numClusters + " clusters (PENDIENTE DE IMPLEMENTAR, complejo con Weka Core)";
                    break;

                default:
                    throw new IllegalArgumentException("Algoritmo '" + algorithm + "' no soportado.");
            }

            return "CSV '" + file.getOriginalFilename() + "' leído. " + recordCount + " registros procesados.\n--- Resultado del Algoritmo (" + algorithm + ") ---\n" + resultOutput;

        } catch (IllegalArgumentException e) {
            System.err.println("Error de Argumento/Datos en PredictionService: " + e.getMessage());
            throw e;
        } catch (WekaException e) {
            System.err.println("Error de Weka en PredictionService: " + e.getMessage());
            e.printStackTrace();
            throw new Exception("Error durante el procesamiento con Weka: " + e.getMessage(), e);
        } catch (Exception e) {
            System.err.println("Error general en PredictionService: " + e.getMessage());
            e.printStackTrace();
            throw new Exception("Error inesperado al procesar el archivo o ejecutar el algoritmo. Detalles: " + e.getMessage(), e);
        }
    }
}