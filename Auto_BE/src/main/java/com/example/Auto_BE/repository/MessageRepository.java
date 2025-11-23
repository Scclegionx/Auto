package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Message;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface MessageRepository extends JpaRepository<Message, Long> {
    
    /**
     * Tìm tất cả message trong chat, sắp xếp theo thời gian (có phân trang)
     */
    Page<Message> findByChatIdOrderByCreatedAtDesc(Long chatId, Pageable pageable);
    
    /**
     * Đếm số message chưa đọc của user trong chat
     */
    @Query("SELECT COUNT(m) FROM Message m WHERE m.chat.id = :chatId " +
           "AND m.sender.id != :userId AND m.isRead = false")
    Long countUnreadMessages(@Param("chatId") Long chatId, 
                              @Param("userId") Long userId);
    
    /**
     * Đánh dấu tất cả message trong chat là đã đọc (của người nhận)
     */
    @Modifying
    @Query("UPDATE Message m SET m.isRead = true, m.readAt = CURRENT_TIMESTAMP " +
           "WHERE m.chat.id = :chatId AND m.sender.id != :userId AND m.isRead = false")
    void markAllAsRead(@Param("chatId") Long chatId, 
                        @Param("userId") Long userId);
}
