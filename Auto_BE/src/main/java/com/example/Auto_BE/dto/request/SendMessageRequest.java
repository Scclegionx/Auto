package com.example.Auto_BE.dto.request;

import lombok.Data;

@Data
public class SendMessageRequest {
    private Long chatId; // ID của chat (nếu đã có)
    private Long receiverId; // ID người nhận (nếu tạo chat mới)
    private String content; // Nội dung tin nhắn
}
