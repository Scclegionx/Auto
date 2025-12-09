package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.MedicationLog;
import com.example.Auto_BE.entity.enums.EMedicationLogStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface MedicationLogRepository extends JpaRepository<MedicationLog, Long> {

    /**
     * Tìm medication log theo ID
     */
    Optional<MedicationLog> findById(Long id);

    /**
     * Lấy tất cả medication logs của Elder
     */
    List<MedicationLog> findByElderUserOrderByReminderTimeDesc(ElderUser elderUser);

    /**
     * Lấy medication logs của Elder trong khoảng thời gian
     */
    @Query("SELECT ml FROM MedicationLog ml " +
           "WHERE ml.elderUser = :elderUser " +
           "AND ml.reminderTime BETWEEN :startTime AND :endTime " +
           "ORDER BY ml.reminderTime DESC")
    List<MedicationLog> findByElderUserAndReminderTimeBetween(
            @Param("elderUser") ElderUser elderUser,
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime
    );

    /**
     * Lấy medication logs của Elder theo status
     */
    @Query("SELECT ml FROM MedicationLog ml " +
           "WHERE ml.elderUser = :elderUser " +
           "AND ml.status = :status " +
           "ORDER BY ml.reminderTime DESC")
    List<MedicationLog> findByElderUserAndStatus(
            @Param("elderUser") ElderUser elderUser,
            @Param("status") EMedicationLogStatus status
    );

    /**
     * Đếm số lượng medication logs theo status trong khoảng thời gian
     */
    @Query("SELECT COUNT(ml) FROM MedicationLog ml " +
           "WHERE ml.elderUser = :elderUser " +
           "AND ml.status = :status " +
           "AND ml.reminderTime BETWEEN :startTime AND :endTime")
    Long countByElderUserAndStatusAndReminderTimeBetween(
            @Param("elderUser") ElderUser elderUser,
            @Param("status") EMedicationLogStatus status,
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime
    );

    /**
     * Tìm medication logs PENDING đã quá giờ (để detect MISSED)
     */
    @Query("SELECT ml FROM MedicationLog ml " +
           "WHERE ml.status = :status " +
           "AND ml.reminderTime < :currentTime " +
           "ORDER BY ml.reminderTime ASC")
    List<MedicationLog> findOverdueLogs(
            @Param("status") EMedicationLogStatus status,
            @Param("currentTime") LocalDateTime currentTime
    );

    /**
     * Lấy medication logs của Elder theo elder ID
     */
    @Query("SELECT ml FROM MedicationLog ml " +
           "WHERE ml.elderUser.id = :elderId " +
           "ORDER BY ml.reminderTime DESC")
    List<MedicationLog> findByElderUserId(@Param("elderId") Long elderId);

    /**
     * Lấy medication logs của Elder theo elder ID và khoảng thời gian
     */
    @Query("SELECT ml FROM MedicationLog ml " +
           "WHERE ml.elderUser.id = :elderId " +
           "AND ml.reminderTime BETWEEN :startTime AND :endTime " +
           "ORDER BY ml.reminderTime DESC")
    List<MedicationLog> findByElderUserIdAndReminderTimeBetween(
            @Param("elderId") Long elderId,
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime
    );

    /**
     * Thống kê adherence rate của Elder
     */
    @Query("SELECT " +
           "COUNT(ml) as total, " +
           "SUM(CASE WHEN ml.status = 'TAKEN' THEN 1 ELSE 0 END) as taken, " +
           "SUM(CASE WHEN ml.status = 'MISSED' THEN 1 ELSE 0 END) as missed, " +
           "SUM(CASE WHEN ml.status = 'TAKEN' AND ABS(ml.minutesLate) <= 15 THEN 1 ELSE 0 END) as onTime " +
           "FROM MedicationLog ml " +
           "WHERE ml.elderUser.id = :elderId " +
           "AND ml.reminderTime BETWEEN :startTime AND :endTime")
    Object[] getAdherenceStats(
            @Param("elderId") Long elderId,
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime
    );
}
