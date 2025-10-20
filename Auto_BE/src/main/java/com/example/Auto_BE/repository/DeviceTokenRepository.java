package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.DeviceToken;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface DeviceTokenRepository extends JpaRepository<DeviceToken, Long> {

    // Tìm device tokens của user theo user ID
    List<DeviceToken> findByUserId(Long userId);
    
    // Tìm device tokens active của user
    List<DeviceToken> findByUserIdAndIsActiveTrue(Long userId);
    
    // Tìm theo FCM token
    DeviceToken findByFcmToken(String fcmToken);
}