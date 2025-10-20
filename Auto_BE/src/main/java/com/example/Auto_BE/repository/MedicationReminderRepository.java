package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.MedicationReminder;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface MedicationReminderRepository extends JpaRepository<MedicationReminder, Long> {
    List<MedicationReminder> findByIsActiveTrue();

    // Query để lấy reminders với validation
    @Query("SELECT m FROM MedicationReminder m WHERE m.isActive = true AND m.type IS NOT NULL AND m.daysOfWeek IS NOT NULL AND m.reminderTime IS NOT NULL")
    List<MedicationReminder> findValidActiveReminders();

    // Query để lấy reminder với eager loading user và deviceTokens cho cron job
    @Query("SELECT m FROM MedicationReminder m " +
            "JOIN FETCH m.user u " +
            "LEFT JOIN FETCH u.deviceTokens " +
            "WHERE m.id = :id")
    java.util.Optional<MedicationReminder> findByIdWithUserAndDeviceTokens(@org.springframework.data.repository.query.Param("id") Long id);

    // Query để lấy tất cả thuốc cùng thời gian của user với eager loading
    @Query("SELECT DISTINCT m FROM MedicationReminder m " +
            "LEFT JOIN FETCH m.user u " +
            "WHERE u.id = :userId " +
            "AND m.reminderTime = :reminderTime " +
            "AND m.isActive = true")
    List<MedicationReminder> findByUserIdAndReminderTime(@org.springframework.data.repository.query.Param("userId") Long userId, 
                                                         @org.springframework.data.repository.query.Param("reminderTime") String reminderTime);
    
    // Query đơn giản không eager loading (backup)                                                     
    @Query("SELECT m FROM MedicationReminder m " +
            "WHERE m.user.id = :userId " +
            "AND m.reminderTime = :reminderTime " +
            "AND m.isActive = true")
    List<MedicationReminder> findByUserIdAndReminderTimeSimple(@org.springframework.data.repository.query.Param("userId") Long userId, 
                                                              @org.springframework.data.repository.query.Param("reminderTime") String reminderTime);
}