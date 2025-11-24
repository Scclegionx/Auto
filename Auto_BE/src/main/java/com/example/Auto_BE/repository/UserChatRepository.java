package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.UserChat;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserChatRepository extends JpaRepository<UserChat, Long> {
    
    /**
     * Tìm tất cả chat của 1 user (sắp xếp theo lastMessageAt)
     */
    @Query("SELECT uc FROM UserChat uc " +
           "JOIN FETCH uc.chat c " +
           "WHERE uc.user.id = :userId AND uc.isActive = true " +
           "ORDER BY c.lastMessageAt DESC NULLS LAST, c.createdAt DESC")
    List<UserChat> findAllByUserId(@Param("userId") Long userId);
    
    /**
     * Tìm UserChat của 1 user trong 1 chat cụ thể
     */
    @Query("SELECT uc FROM UserChat uc " +
           "WHERE uc.user.id = :userId AND uc.chat.id = :chatId")
    Optional<UserChat> findByUserIdAndChatId(@Param("userId") Long userId, 
                                               @Param("chatId") Long chatId);
    
    /**
     * Tìm tất cả users trong 1 chat
     */
    @Query("SELECT uc FROM UserChat uc " +
           "JOIN FETCH uc.user " +
           "WHERE uc.chat.id = :chatId AND uc.isActive = true")
    List<UserChat> findAllByChatId(@Param("chatId") Long chatId);
    
    /**
     * Đếm số users trong chat
     */
    @Query("SELECT COUNT(uc) FROM UserChat uc " +
           "WHERE uc.chat.id = :chatId AND uc.isActive = true")
    Long countUsersByChatId(@Param("chatId") Long chatId);
    
    /**
     * Reset unread count
     */
    @Modifying
    @Query("UPDATE UserChat uc SET uc.unreadCount = 0 " +
           "WHERE uc.user.id = :userId AND uc.chat.id = :chatId")
    void resetUnreadCount(@Param("userId") Long userId, 
                          @Param("chatId") Long chatId);
}
