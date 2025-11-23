package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.ArrayList;
import java.util.List;

/**
 * Chat conversation (1-1 hoặc group)
 * Dùng Many-to-Many với User qua bảng trung gian UserChat
 */
@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Accessors(chain = true)
@Table(name = "chats")
public class Chat extends BaseEntity {
    
    @Column(name = "chat_type", nullable = false, length = 20)
    private String chatType = "DIRECT"; // DIRECT (1-1) hoặc GROUP (nhóm)
    
    @Column(name = "chat_name", length = 100)
    private String chatName; // Tên chat (null cho chat 1-1, có giá trị cho group)
    
    @Column(name = "last_message_content", length = 500)
    private String lastMessageContent; // Preview message cuối
    
    @Column(name = "last_message_at")
    private java.time.Instant lastMessageAt; // Thời gian message cuối
    
    // Many-to-Many với User qua bảng UserChat
    @OneToMany(mappedBy = "chat", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<UserChat> userChats = new ArrayList<>();
    
    // One-to-Many với Message
    @OneToMany(mappedBy = "chat", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Message> messages = new ArrayList<>();
}
