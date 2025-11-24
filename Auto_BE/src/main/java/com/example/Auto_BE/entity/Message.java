package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * Message trong chat
 */
@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Accessors(chain = true)
@Table(name = "messages")
public class Message extends BaseEntity {
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "chat_id", nullable = false)
    private Chat chat; // Chat chứa message này
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "sender_id", nullable = false)
    private User sender; // Người gửi
    
    @Column(name = "content", nullable = false, length = 2000)
    private String content; // Nội dung tin nhắn
    
    @Column(name = "is_read", nullable = false)
    private Boolean isRead = false; // Đã đọc chưa
    
    @Column(name = "read_at")
    private java.time.Instant readAt; // Thời gian đọc
}
