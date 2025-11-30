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
public class MessageResponse {
    private Long id;
    private Long chatId;
    private Long senderId;
    private String senderName;
    private String senderAvatar;
    private String content;
    private String messageType; // TEXT, IMAGE, FILE, AUDIO, VIDEO
    private String attachmentUrl; // Cloud storage URL
    private String attachmentName; // Original filename
    private String attachmentType; // MIME type
    private Long attachmentSize; // File size in bytes
    private Boolean isRead;
    private Instant readAt;
    private Instant createdAt;
}
