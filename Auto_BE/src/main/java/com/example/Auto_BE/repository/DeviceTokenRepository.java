package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.DeviceToken;
import com.example.Auto_BE.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface DeviceTokenRepository extends JpaRepository<DeviceToken, Long> {

    // Tìm device tokens của user theo user ID
    List<DeviceToken> findByUserId(Long userId);
    
    // Tìm device tokens active của user
    List<DeviceToken> findByUserIdAndIsActiveTrue(Long userId);
    
    // Tìm device tokens active của user (bằng User object)
    List<DeviceToken> findByUserAndIsActive(User user, Boolean isActive);
    
    // Tìm theo FCM token
    DeviceToken findByFcmToken(String fcmToken);
    
    // Kiểm tra token đã tồn tại cho user chưa
    boolean existsByUserIdAndFcmToken(Long userId, String fcmToken);
}