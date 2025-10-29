package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface NotificationRepository extends JpaRepository<Notifications, Long> {

    // Tìm kiếm thông báo theo ID
    Optional<Notifications> findById(Long id);

    // Xóa thông báo theo ID
    void deleteById(Long id);
    
    // ============= QUERIES MỚI CHO LỊCH SỬ THÔNG BÁO =============
    
    /**
     * Lấy tất cả lịch sử thông báo của user
     */
    @Query("SELECT n FROM Notifications n " +
           "WHERE n.user.id = :userId " +
           "ORDER BY n.reminderTime DESC")
    List<Notifications> findByUserId(@Param("userId") Long userId);
    
    /**
     * Lấy thông báo theo user và trạng thái đã đọc
     */
    @Query("SELECT n FROM Notifications n " +
           "WHERE n.user.id = :userId " +
           "AND n.isRead = :isRead " +
           "ORDER BY n.reminderTime DESC")
    List<Notifications> findByUserIdAndIsRead(
            @Param("userId") Long userId,
            @Param("isRead") Boolean isRead
    );
    
    /**
     * Đếm số thông báo chưa đọc
     */
    @Query("SELECT COUNT(n) FROM Notifications n " +
           "WHERE n.user.id = :userId " +
           "AND n.isRead = false")
    Long countUnreadByUserId(@Param("userId") Long userId);
    
    /**
     * Lọc thông báo với nhiều điều kiện
     */
    @Query("SELECT n FROM Notifications n " +
           "WHERE n.user.id = :userId " +
           "AND (:startDate IS NULL OR n.reminderTime >= :startDate) " +
           "AND (:endDate IS NULL OR n.reminderTime <= :endDate) " +
           "AND (:status IS NULL OR n.status = :status) " +
           "ORDER BY n.reminderTime DESC")
    List<Notifications> findByFilters(
            @Param("userId") Long userId,
            @Param("startDate") LocalDateTime startDate,
            @Param("endDate") LocalDateTime endDate,
            @Param("status") ENotificationStatus status
    );

}