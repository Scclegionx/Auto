package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.Optional;

public interface NotificationRepository extends JpaRepository<Notifications, Long> {

    // Tìm kiếm thông báo theo ID
    Optional<Notifications> findById(Long id);

    // Xóa thông báo theo ID
    void deleteById(Long id);

    // Kiểm tra xem có thông báo nào với trạng thái PENDING không
    boolean existsByStatus(ENotificationStatus status);

}