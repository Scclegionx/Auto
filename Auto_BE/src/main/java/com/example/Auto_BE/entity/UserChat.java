package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * Bảng trung gian Many-to-Many giữa User và Chat
 */
@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Accessors(chain = true)
@Table(name = "user_chats", 
       uniqueConstraints = @UniqueConstraint(columnNames = {"user_id", "chat_id"}))
public class UserChat extends BaseEntity {
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "chat_id", nullable = false)
    private Chat chat;
    
    @Column(name = "unread_count", nullable = false)
    private Integer unreadCount = 0; // Số tin nhắn chưa đọc của user này
    
    @Column(name = "is_active", nullable = false)
    private Boolean isActive = true; // User còn trong chat không (cho tính năng rời chat sau này)
}
