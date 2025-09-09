# Use OpenJDK 17 for runtime
FROM openjdk:17-jdk-slim

# Set working directory
WORKDIR /app

# Copy the pre-built jar file
COPY target/Auto_BE-0.0.1-SNAPSHOT.jar app.jar

# Expose port
EXPOSE 8080

# Run the application
ENTRYPOINT ["java", "-jar", "app.jar"]