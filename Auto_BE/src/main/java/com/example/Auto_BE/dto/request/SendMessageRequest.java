package com.example.Auto_BE.dto.request;

import lombok.Data;

@Data
public class SendMessageRequest {
    private Long chatId; // ID của chat (nếu đã có)
    private Long receiverId; // ID người nhận (nếu tạo chat mới)
    private String content; // Nội dung tin nhắn
    private String messageType; // TEXT, IMAGE, FILE, AUDIO, VIDEO
    private String attachmentUrl; // Cloud storage URL
    private String attachmentName; // Original filename
    private String attachmentType; // MIME type
    private Long attachmentSize; // File size in bytes
}
