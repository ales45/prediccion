package org.example.prediccion.view;

import org.example.prediccion.service.PredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class ViewController {

    @Autowired
    private PredictionService predictionService;

    // 1. Página de Inicio
    @GetMapping("/")
    public String mostrarPaginaInicio() {
        return "home_page"; // servirá home_page.html
    }

    // 2. Página de Selección (después de "Empezar")
    @GetMapping("/menu")
    public String mostrarMenuSeleccion() {
        return "menu_seleccion_page"; // servirá menu_seleccion_page.html
    }

    // 3. Página de Explicación de Algoritmos
    @GetMapping("/explicaciones")
    public String mostrarPaginaExplicaciones() {
        // Aquí podrías pasar al modelo una lista de algoritmos si las "cajitas" se generan dinámicamente
        return "explicaciones_page"; // servirá explicaciones_page.html
    }

    // 4. Página de Ejecución de Algoritmos (la que ya teníamos)
    @GetMapping("/ejecutar") // Cambié la ruta para que sea diferente de la raíz
    public String mostrarFormularioEjecucion(Model model) {
        // Asegurar que el modelo esté limpio o con valores por defecto al cargar la página
        model.addAttribute("predictionParameters", new PredictionParameters()); // Objeto para el formulario
        return "prediction_page"; // servirá prediction_page.html
    }

    @PostMapping("/ejecutar/procesar") // Ruta específica para el procesamiento del formulario de ejecución
    public String procesarAlgoritmo(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "targetColumn", required = false) String targetColumn,
            @RequestParam("algorithm") String algorithm,
            @RequestParam(value = "numClusters", required = false) Integer numClusters,
            Model model) {

        model.addAttribute("predictionParameters", new PredictionParameters(targetColumn, algorithm, numClusters)); // Para rellenar el form

        if (file.isEmpty()) {
            model.addAttribute("errorMessage", "Por favor, selecciona un archivo CSV para subir.");
            return "prediction_page";
        }

        String algorithmLower = algorithm.toLowerCase();
        boolean isClustering = algorithmLower.equals("kmedias") || algorithmLower.equals("kmodas");

        if (isClustering && (numClusters == null || numClusters <= 0)) {
            model.addAttribute("errorMessage", "Para '" + algorithm + "', el parámetro 'Número de Clusters' debe ser un número mayor que 0.");
            return "prediction_page";
        }
        if (!isClustering && (targetColumn == null || targetColumn.trim().isEmpty())) {
            model.addAttribute("errorMessage", "Para '" + algorithm + "', el parámetro 'Nombre de la Columna Objetivo/Clase' es requerido.");
            return "prediction_page";
        }

        try {
            String result = predictionService.processCsvAndPredict(file, targetColumn, algorithm, numClusters);
            model.addAttribute("formResult", result); // Cambiado de "predictionResult" a "formResult" para evitar confusión
        } catch (IllegalArgumentException e) {
            model.addAttribute("errorMessage", "Error en la solicitud: " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            model.addAttribute("errorMessage", "Error inesperado durante el procesamiento: " + e.getMessage());
        }
        return "prediction_page"; // Vuelve a mostrar la misma página con el resultado o error
    }

    // Clase interna simple para ayudar a Thymeleaf a recordar los valores del formulario (opcional pero útil)
    public static class PredictionParameters {
        private String targetColumn;
        private String algorithm;
        private Integer numClusters;

        public PredictionParameters() {}
        public PredictionParameters(String tc, String algo, Integer nc) {
            this.targetColumn = tc;
            this.algorithm = algo;
            this.numClusters = nc;
        }
        // Getters y Setters
        public String getTargetColumn() { return targetColumn; }
        public void setTargetColumn(String targetColumn) { this.targetColumn = targetColumn; }
        public String getAlgorithm() { return algorithm; }
        public void setAlgorithm(String algorithm) { this.algorithm = algorithm; }
        public Integer getNumClusters() { return numClusters; }
        public void setNumClusters(Integer numClusters) { this.numClusters = numClusters; }
    }
}