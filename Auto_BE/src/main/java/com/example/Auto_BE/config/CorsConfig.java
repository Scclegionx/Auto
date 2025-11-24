package com.example.Auto_BE.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;

import java.util.Arrays;

@Configuration
public class CorsConfig {

    @Bean
    public CorsFilter corsFilter() {
        CorsConfiguration config = new CorsConfiguration();
        
        // Cho phép tất cả origins (development only)
        config.setAllowCredentials(true);
        config.addAllowedOriginPattern("*");
        
        // Cho phép tất cả headers
        config.addAllowedHeader("*");
        
        // Cho phép tất cả methods
        config.addAllowedMethod("*");
        
        // Expose headers cho client
        config.setExposedHeaders(Arrays.asList("Authorization"));
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        
        return new CorsFilter(source);
    }
}
