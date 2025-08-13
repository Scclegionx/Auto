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
}
