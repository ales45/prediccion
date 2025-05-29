package org.example.prediccion.controller;

import org.example.prediccion.service.PredictionService; // Importa el servicio del paquete correcto
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.bind.annotation.RequestParam;


@RestController
@RequestMapping("/api/predict") // Ruta base para las predicciones
public class PredictionController {

    private final PredictionService predictionService;

    @Autowired
    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping("/upload-csv")
    public ResponseEntity<String> uploadCsvFile(
            @RequestParam("file") MultipartFile file,
            @RequestParam("targetColumn") String targetColumnName,
            @RequestParam(value = "algorithm", defaultValue = "linearRegression") String algorithm,
            @RequestParam(value = "numClusters", required = false) Integer numClusters){

        if (file.isEmpty()) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Por favor, selecciona un archivo CSV para subir.");
        }

        String algorithmLower = algorithm.toLowerCase();
        if ((algorithmLower.equals("kmedias") || algorithmLower.equals("kmodas")) && (numClusters == null || numClusters <= 0)) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body("Para los algoritmos de clustering (kmedias, kmodas), por favor especifica un valor válido para 'numClusters' (mayor que 0).");
        }

        try {
            // Llamamos al servicio para procesar el archivo
            String processingResult = predictionService.processCsvAndPredict(file, targetColumnName, algorithm,numClusters);

            // Por ahora, devolvemos un mensaje simple. Más adelante será el resultado de la predicción.
            return ResponseEntity.ok("Archivo '" + file.getOriginalFilename() + "' procesado con algoritmo '" + algorithm + "'. Resultado: " + processingResult);

        } catch (IllegalArgumentException e) {
            // Captura específica para errores de validación, como columna no encontrada
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        } catch (Exception e) {
            e.printStackTrace(); // Importante para depuración en el servidor
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error interno al procesar el archivo: " + e.getMessage());
        }
    }
}
