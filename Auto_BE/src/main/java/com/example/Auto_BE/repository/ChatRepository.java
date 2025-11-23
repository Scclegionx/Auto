package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Chat;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface ChatRepository extends JpaRepository<Chat, Long> {
    
    /**
     * Tìm chat 1-1 giữa 2 users cụ thể
     */
    @Query("SELECT c FROM Chat c " +
           "WHERE c.chatType = 'DIRECT' " +
           "AND EXISTS (SELECT uc1 FROM UserChat uc1 WHERE uc1.chat.id = c.id AND uc1.user.id = :userId1) " +
           "AND EXISTS (SELECT uc2 FROM UserChat uc2 WHERE uc2.chat.id = c.id AND uc2.user.id = :userId2) " +
           "AND (SELECT COUNT(uc) FROM UserChat uc WHERE uc.chat.id = c.id AND uc.isActive = true) = 2")
    Optional<Chat> findDirectChatBetweenUsers(@Param("userId1") Long userId1, 
                                                @Param("userId2") Long userId2);
}
