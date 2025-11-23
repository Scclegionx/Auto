package com.example.Auto_BE.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class ChatResponse {
    private Long id;
    private Long user1Id;
    private String user1Name;
    private String user1Avatar;
    private Long user2Id;
    private String user2Name;
    private String user2Avatar;
    private String lastMessage;
    private Instant lastMessageTime;
    private Long unreadCount; // Số tin chưa đọc của user hiện tại
    private Instant createdAt;
    private Instant updatedAt;
}
