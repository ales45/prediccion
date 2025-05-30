# ---- Etapa de Construcción (Build Stage) ----
# Usamos una imagen de Maven que ya incluye una JDK compatible (ej. Temurin 17)
FROM maven:3.9.6-eclipse-temurin-17 AS build

# Establecemos el directorio de trabajo dentro de la imagen de construcción
WORKDIR /workspace/app

# Copiamos primero el pom.xml para aprovechar el caché de capas de Docker
# si las dependencias no han cambiado.
COPY pom.xml .

# Descargamos todas las dependencias de Maven
# Si tu proyecto tiene un wrapper (mvnw y .mvn), puedes copiarlo y usarlo:
# COPY mvnw .
# COPY .mvn .mvn
# RUN chmod +x mvnw
# RUN ./mvnw dependency:go-offline -B
# Si no, usa mvn directamente:
RUN mvn dependency:go-offline -B

# Copiamos el resto del código fuente de tu proyecto
COPY src ./src

# Construimos la aplicación y generamos el JAR ejecutable
# Usamos -o para modo offline si las dependencias ya se bajaron, o quítalo.
# Si tienes mvnw: RUN ./mvnw package -DskipTests
RUN mvn package -DskipTests

# ---- Etapa de Ejecución (Runtime Stage) ----
# Usamos una imagen JRE ligera, ya que solo necesitamos ejecutar el JAR
FROM openjdk:17-jdk-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos el JAR que se construyó en la etapa 'build'
# Maven lo crea en /workspace/app/target/tu-artefacto-version.jar
# El * se encarga de tomar el JAR correcto sin saber la versión exacta si solo hay uno.
COPY --from=build /workspace/app/target/*.jar app.jar

# Exponemos el puerto en el que corre la aplicación Spring Boot (usualmente 8080)
EXPOSE 8080

# Comando para ejecutar la aplicación cuando el contenedor Docker inicie
ENTRYPOINT ["java","-jar","/app/app.jar"]