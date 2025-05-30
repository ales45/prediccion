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
import org.example.prediccion.weka.KnnPredictor; // Ajusta el paquete si es necesario


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

    // Este método va DENTRO de tu clase PredictionService.java

    private Instances convertCsvToWekaInstances(List<CSVRecord> csvRecords, List<String> rawHeaders, String targetColumnName) throws Exception {
        if (csvRecords.isEmpty() && (rawHeaders == null || rawHeaders.isEmpty())) {
            throw new IllegalArgumentException("No hay encabezados ni registros de datos en el CSV para procesar.");
        }
        if (csvRecords.isEmpty() && !rawHeaders.isEmpty()) {
            System.out.println("Advertencia en convertCsvToWekaInstances: No hay registros de datos en el CSV, solo encabezados.");
            // Considerar si lanzar una excepción si se requieren datos para la mayoría de los modelos.
        }

        ArrayList<Attribute> wekaAttributes = new ArrayList<>();
        List<String> finalWekaHeaderNames = new ArrayList<>();
        List<Integer> originalCsvColumnIndices = new ArrayList<>();

        for (int i = 0; i < rawHeaders.size(); i++) {
            String headerName = rawHeaders.get(i).trim();

            if (headerName.isEmpty()) {
                System.out.println("INFO: Omitiendo encabezado vacío encontrado en la posición CSV original " + i);
                continue;
            }
            if (finalWekaHeaderNames.contains(headerName)) {
                System.out.println("ADVERTENCIA: Nombre de encabezado duplicado '" + headerName + "' encontrado en la posición CSV original " + i + ". Se omitirá este duplicado para Weka.");
                continue;
            }

            finalWekaHeaderNames.add(headerName);
            originalCsvColumnIndices.add(i);

            // Lógica de Detección de Tipo Mejorada
            boolean isNumericCandidate = true;
            ArrayList<String> uniqueNominalValues = new ArrayList<>();
            int validValueCount = 0;

            if (!csvRecords.isEmpty()) {
                for (CSVRecord record : csvRecords) {
                    if (i < record.size()) {
                        String value = record.get(i).trim();
                        if (value.isEmpty() || value.equals("?")) {
                            continue; // Ignorar valores perdidos/vacíos para la detección de tipo
                        }
                        validValueCount++;
                        // Intentar convertir a número
                        try {
                            Double.parseDouble(value);
                            // Si tiene éxito, sigue siendo un candidato numérico
                        } catch (NumberFormatException e) {
                            isNumericCandidate = false; // Si uno falla, no es consistentemente numérico
                        }
                        // Siempre recolectar valores únicos para el caso nominal
                        if (!uniqueNominalValues.contains(value)) {
                            uniqueNominalValues.add(value);
                        }
                    }
                }
            } else { // No hay registros de datos, no podemos inferir mucho
                isNumericCandidate = false; // Por defecto a nominal si no hay datos para verificar
            }

            // Si no hay valores válidos en la columna, por defecto a Nominal con un placeholder
            if (validValueCount == 0) {
                isNumericCandidate = false;
                if (uniqueNominalValues.isEmpty()) {
                    uniqueNominalValues.add("(columna_vacia_o_todo_missing)");
                }
                System.out.println("INFO: Atributo '" + headerName + "' no tiene valores válidos o está vacío. Se tratará como NOMINAL.");
            }


            if (isNumericCandidate) {
                // Podríamos añadir más heurísticas aquí (ej. si hay muy pocos valores únicos numéricos, tratarlo como nominal)
                // Por ejemplo: if (uniqueNominalValues.size() < 5 && uniqueNominalValues.size() < validValueCount * 0.1) isNumericCandidate = false;
                System.out.println("INFO: Atributo '" + headerName + "' detectado como NUMÉRICO.");
                wekaAttributes.add(new Attribute(headerName)); // Atributo numérico
            } else {
                System.out.println("INFO: Atributo '" + headerName + "' detectado como NOMINAL. Valores únicos: " + uniqueNominalValues.size());
                if (uniqueNominalValues.isEmpty()) { // Asegurar que haya al menos un valor para Weka Nominal
                    uniqueNominalValues.add("(valor_desconocido_o_vacio)");
                    System.out.println("Advertencia: El atributo nominal '" + headerName + "' no tenía valores únicos observables. Se añadió un placeholder.");
                }
                wekaAttributes.add(new Attribute(headerName, uniqueNominalValues));
            }
        }

        if (wekaAttributes.isEmpty()) {
            throw new IllegalArgumentException("No se pudieron definir atributos válidos para Weka a partir de los encabezados del CSV.");
        }

        Instances data = new Instances("DatasetCSV", wekaAttributes, csvRecords.size());
        int targetWekaAttributeIndex = -1;
        for (int i = 0; i < finalWekaHeaderNames.size(); i++) {
            if (finalWekaHeaderNames.get(i).equalsIgnoreCase(targetColumnName)) {
                targetWekaAttributeIndex = i;
                break;
            }
        }

        if (targetWekaAttributeIndex == -1) {
            // Verifica si la targetColumnName original estaba entre los rawHeaders pero fue filtrada (ej. por ser vacía o duplicada)
            boolean originalTargetExists = false;
            for(String rawH : rawHeaders) {
                if (rawH.trim().equalsIgnoreCase(targetColumnName)) {
                    originalTargetExists = true;
                    break;
                }
            }
            if (originalTargetExists && !finalWekaHeaderNames.contains(targetColumnName.trim())) { // targetColumnName.trim() para ser consistentes
                throw new IllegalArgumentException("La columna objetivo especificada '" + targetColumnName + "' existe en el CSV pero fue filtrada (ej. por ser vacía o duplicada) antes de crear atributos Weka. Encabezados Weka finales: " + finalWekaHeaderNames);
            } else {
                throw new IllegalArgumentException("La columna objetivo especificada '" + targetColumnName + "' no se encontró entre los encabezados válidos y no vacíos procesados para Weka. Encabezados Weka finales: " + finalWekaHeaderNames);
            }
        }
        data.setClassIndex(targetWekaAttributeIndex);

        // Llenar las instancias (esta parte del código es similar a la anterior, usando originalCsvColumnIndices)
        for (CSVRecord csvRecord : csvRecords) {
            DenseInstance instance = new DenseInstance(wekaAttributes.size());
            instance.setDataset(data);
            for (int wekaAttrIdx = 0; wekaAttrIdx < wekaAttributes.size(); wekaAttrIdx++) {
                Attribute attribute = data.attribute(wekaAttrIdx);
                int originalCsvColIdx = originalCsvColumnIndices.get(wekaAttrIdx); // Usar el índice original mapeado
                String rawValue = "";
                if (originalCsvColIdx < csvRecord.size()) {
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
                        } else {
                            instance.setValue(attribute, rawValue); // Para tipo String de Weka, si se usara
                        }
                    } catch (NumberFormatException e_num) {
                        System.err.println("Advertencia: No se pudo parsear '" + rawValue + "' como número para '" + attribute.name() + "'. Se establece como perdido.");
                        instance.setMissing(attribute);
                    } catch (IllegalArgumentException e_nominal) {
                        // Esto ocurre si el rawValue no es uno de los definidos para el atributo nominal
                        System.err.println("Advertencia: Valor '" + rawValue + "' no es un valor nominal permitido para '" + attribute.name() + "'. Se establece como perdido. Valores permitidos: " + java.util.Collections.list(attribute.enumerateValues()));
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
                    if (wekaData.classIndex() == -1) {
                        throw new IllegalStateException("El atributo clase (target) no ha sido asignado para KNN. Verifica que targetColumn '" + targetColumnName + "' sea válido.");
                    }
                    // Para KNN, podrías permitir configurar K a través de un parámetro en el futuro.
                    // Por ahora, usa el K por defecto (3) definido en KnnPredictor o uno específico:
                    // predictor = new KnnPredictor(5); // Ejemplo para K=5
                    predictor = new KnnPredictor(); // Usará K=3 por defecto
                    resultOutput = predictor.buildModelAndGetResults(wekaData);
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
                    // Pasamos wekaData, numClusters, y el targetColumnName original de la UI
                    resultOutput = kMeansImplementation.buildClustererAndGetAssignments(wekaData, numClusters, targetColumnName);
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