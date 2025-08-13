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
    
    // Kiểm tra notification đã tồn tại cho medication reminder và thời gian cụ thể
    @Query("SELECT COUNT(n) > 0 FROM Notifications n WHERE n.medicationReminder.id = :reminderId AND n.reminderTime = :reminderTime AND n.status IN ('PENDING', 'SENT')")
    boolean existsByMedicationReminderAndTime(@Param("reminderId") Long reminderId, @Param("reminderTime") LocalDateTime reminderTime);
    
    // Lấy notification đã tồn tại
    @Query("SELECT n FROM Notifications n WHERE n.medicationReminder.id = :reminderId AND n.reminderTime = :reminderTime AND n.status IN ('PENDING', 'SENT')")
    Optional<Notifications> findByMedicationReminderAndTime(@Param("reminderId") Long reminderId, @Param("reminderTime") LocalDateTime reminderTime);
}
