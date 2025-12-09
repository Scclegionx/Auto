package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Notifications;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface NotificationRepository extends JpaRepository<Notifications, Long> {

    // Tìm kiếm thông báo theo ID
    Optional<Notifications> findById(Long id);

    // Xóa thông báo theo ID
    void deleteById(Long id);
    
    /**
     * Lấy tất cả notifications của user (sort by createdAt)
     */
    @Query("SELECT n FROM Notifications n " +
           "WHERE n.user.id = :userId " +
           "ORDER BY n.createdAt DESC")
    List<Notifications> findByUserId(@Param("userId") Long userId);
    
    /**
     * Lấy notifications theo user và trạng thái đã đọc
     */
    @Query("SELECT n FROM Notifications n " +
           "WHERE n.user.id = :userId " +
           "AND n.isRead = :isRead " +
           "ORDER BY n.createdAt DESC")
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

}