package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.service.NotificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/medication")
public class MedicationController {

    private final NotificationService notificationService;

    public MedicationController(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    @PostMapping("/confirm/{notificationId}")
    public ResponseEntity<BaseResponse<String>> confirmTaken(@PathVariable Long notificationId) {
        BaseResponse<String> response = notificationService.confirmTaken(notificationId);
        return ResponseEntity.ok(response);
    }
}