package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.request.SendMessageRequest;
import com.example.Auto_BE.dto.response.ChatResponse;
import com.example.Auto_BE.dto.response.MessageResponse;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.repository.UserRepository;
import com.example.Auto_BE.utils.JwtUtils;
import com.example.Auto_BE.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessageHeaderAccessor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * Controller cho chat 1-1
 */
@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
public class ChatController {
    
    private final ChatService chatService;
    private final JwtUtils jwtUtils;
    private final UserRepository userRepository;
    
    /**
     * REST API: Lấy tất cả chat của user
     */
    @GetMapping
    public ResponseEntity<List<ChatResponse>> getAllChats(
            @RequestHeader("Authorization") String authHeader) {
        
        Long userId = getUserIdFromToken(authHeader);
        List<ChatResponse> chats = chatService.getAllChats(userId);
        return ResponseEntity.ok(chats);
    }
    
    /**
     * REST API: Lấy chi tiết 1 chat
     */
    @GetMapping("/{chatId}")
    public ResponseEntity<ChatResponse> getChatById(
            @PathVariable Long chatId,
            @RequestHeader("Authorization") String authHeader) {
        
        Long userId = getUserIdFromToken(authHeader);
        ChatResponse chat = chatService.getChatById(chatId, userId);
        return ResponseEntity.ok(chat);
    }
    
    /**
     * REST API: Lấy danh sách message trong chat (có phân trang)
     */
    @GetMapping("/{chatId}/messages")
    public ResponseEntity<Page<MessageResponse>> getMessages(
            @PathVariable Long chatId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "50") int size,
            @RequestHeader("Authorization") String authHeader) {
        
        Long userId = getUserIdFromToken(authHeader);
        Pageable pageable = PageRequest.of(page, size);
        Page<MessageResponse> messages = chatService.getMessages(chatId, userId, pageable);
        return ResponseEntity.ok(messages);
    }
    
    /**
     * REST API: Gửi tin nhắn (alternative cho WebSocket)
     */
    @PostMapping("/send")
    public ResponseEntity<MessageResponse> sendMessage(
            @RequestBody SendMessageRequest request,
            @RequestHeader("Authorization") String authHeader) {
        
        Long userId = getUserIdFromToken(authHeader);
        MessageResponse message = chatService.sendMessage(request, userId);
        return ResponseEntity.ok(message);
    }
    
    /**
     * REST API: Đánh dấu tất cả message trong chat là đã đọc
     */
    @PutMapping("/{chatId}/read")
    public ResponseEntity<Void> markAsRead(
            @PathVariable Long chatId,
            @RequestHeader("Authorization") String authHeader) {
        
        Long userId = getUserIdFromToken(authHeader);
        chatService.markMessagesAsRead(chatId, userId);
        return ResponseEntity.ok().build();
    }
    
    /**
     * WebSocket: Gửi tin nhắn real-time
     * Client gửi message đến: /app/chat.send
     * Server gửi đến người nhận: /user/{receiverEmail}/queue/messages
     */
    @MessageMapping("/chat.send")
    public void sendMessageViaWebSocket(
            @Payload SendMessageRequest request,
            SimpMessageHeaderAccessor headerAccessor) {
        
        // Lấy username từ WebSocket session (đã set trong HandshakeInterceptor)
        String username = (String) headerAccessor.getSessionAttributes().get("username");
        
        if (username != null) {
            // Tìm userId từ username (email)
            Long userId = getUserIdFromUsername(username);
            
            // Gửi message (ChatService sẽ tự động broadcast qua WebSocket)
            chatService.sendMessage(request, userId);
        }
    }
    
    /**
     * Helper: Lấy userId từ JWT token
     */
    private Long getUserIdFromToken(String authHeader) {
        String token = authHeader.substring(7); // Bỏ "Bearer "
        return jwtUtils.getUserIdFromToken(token);
    }
    
    /**
     * Helper: Lấy userId từ username (email)
     */
    private Long getUserIdFromUsername(String username) {
        User user = userRepository.findByEmail(username).orElse(null);
        return user != null ? user.getId() : null;
    }
}
