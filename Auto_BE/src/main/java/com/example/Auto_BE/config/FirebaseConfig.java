package com.example.Auto_BE.config;

import com.example.Auto_BE.service.FcmService;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct;

@Configuration
public class FirebaseConfig {

    @PostConstruct
    public void initializeFirebase() {
        FcmService.initialize();
    }
}