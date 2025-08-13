package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.repository.NotificationRepository;
import org.springframework.stereotype.Service;

import javax.management.Notification;

@Service
public class NotificationService {
    private final NotificationRepository notificationRepository;

    public NotificationService(NotificationRepository notificationRepository) {
        this.notificationRepository = notificationRepository;
    }


    public Notifications findById(Long id) {
        return notificationRepository.findById(id).orElse(null);
    }

    public Notifications save(Notifications log) {
        return notificationRepository.save(log);
    }
}
