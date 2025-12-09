package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.response.NotificationResponse;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.repository.UserRepository;
import com.example.Auto_BE.service.NotificationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;

@RestController
@RequestMapping("/api/notifications")
public class NotificationController {

    private final NotificationService notificationService;
    
    @Autowired
    private UserRepository userRepository;

    public NotificationController(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    /**
     * Đánh dấu thông báo đã đọc
     * PUT /api/notifications/{notificationId}/read
     */
    @PutMapping("/{notificationId}/read")
    public ResponseEntity<BaseResponse<String>> markAsRead(@PathVariable Long notificationId) {
        BaseResponse<String> response = notificationService.markAsRead(notificationId);
        return ResponseEntity.ok(response);
    }
    
    /**
     * Đánh dấu tất cả thông báo đã đọc
     * PUT /api/notifications/read-all
     */
    @PutMapping("/read-all")
    public ResponseEntity<BaseResponse<String>> markAllAsRead(Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        BaseResponse<String> response = notificationService.markAllAsRead(user.getId());
        return ResponseEntity.ok(response);
    }
    
    /**
     * Đếm số thông báo chưa đọc
     * GET /api/notifications/unread-count
     */
    @GetMapping("/unread-count")
    public ResponseEntity<BaseResponse<Long>> getUnreadCount(Authentication authentication) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        Long count = notificationService.getUnreadCount(user.getId());
        
        return ResponseEntity.ok(BaseResponse.<Long>builder()
                .status(SUCCESS)
                .message("Lấy số thông báo chưa đọc thành công")
                .data(count)
                .build());
    }

    /**
     * Lấy tất cả notifications của user
     * GET /api/notifications?page=0&size=20
     */
    @GetMapping
    public ResponseEntity<BaseResponse<List<NotificationResponse>>> getAllNotifications(
            Authentication authentication,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        BaseResponse<List<NotificationResponse>> response = 
                notificationService.getUserNotifications(user.getId(), null, page, size);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Lấy notifications chưa đọc
     * GET /api/notifications/unread?page=0&size=20
     */
    @GetMapping("/unread")
    public ResponseEntity<BaseResponse<List<NotificationResponse>>> getUnreadNotifications(
            Authentication authentication,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        BaseResponse<List<NotificationResponse>> response = 
                notificationService.getUserNotifications(user.getId(), false, page, size);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Lấy notifications đã đọc
     * GET /api/notifications/read?page=0&size=20
     */
    @GetMapping("/read")
    public ResponseEntity<BaseResponse<List<NotificationResponse>>> getReadNotifications(
            Authentication authentication,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        BaseResponse<List<NotificationResponse>> response = 
                notificationService.getUserNotifications(user.getId(), true, page, size);
        
        return ResponseEntity.ok(response);
    }
}