# Usar una imagen base de Java (OpenJDK es común y ligero).
# Asegúrate de que la versión de Java (ej. 17) coincida con la de tu proyecto.
FROM openjdk:17-jdk-slim

# Argumento para el nombre del archivo JAR.
# Maven lo construye en target/nombre-del-jar.jar
ARG JAR_FILE=target/*.jar

# Crear un directorio para la aplicación dentro de la imagen Docker
WORKDIR /app

# Copiar el archivo JAR empaquetado desde la carpeta target al directorio /app en la imagen
# y renombrarlo a app.jar para simplificar el comando de inicio.
COPY ${JAR_FILE} app.jar

# Exponer el puerto en el que tu aplicación Spring Boot corre internamente (usualmente 8080)
# Render mapeará su puerto externo a este puerto interno.
EXPOSE 8080

# Comando para ejecutar la aplicación cuando el contenedor Docker inicie
ENTRYPOINT ["java","-jar","/app/app.jar"]