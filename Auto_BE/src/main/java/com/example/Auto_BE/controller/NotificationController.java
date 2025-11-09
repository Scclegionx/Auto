package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.response.NotificationResponse;
import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.repository.UserRepository;
import com.example.Auto_BE.service.NotificationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
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
     * Lấy lịch sử thông báo với filter và phân trang
     * GET /api/notifications/history?startDate=...&endDate=...&status=...&page=0&size=20
     * 
     * @param authentication - User hiện tại (từ JWT)
     * @param startDate - Ngày bắt đầu (optional, format: yyyy-MM-dd'T'HH:mm:ss)
     * @param endDate - Ngày kết thúc (optional)
     * @param status - Trạng thái: PENDING, SENT, FAILED (optional)
     * @param page - Số trang (default: 0)
     * @param size - Số item mỗi trang (default: 20)
     * @return Danh sách lịch sử thông báo
     */
    @GetMapping("/history")
    public ResponseEntity<BaseResponse<List<NotificationResponse>>> getHistory(
            Authentication authentication,
            @RequestParam(required = false) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam(required = false) 
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @RequestParam(required = false) ENotificationStatus status,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        // Lấy userId từ authentication
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        BaseResponse<List<NotificationResponse>> response = 
                notificationService.getHistory(user.getId(), startDate, endDate, status, page, size);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Lấy lịch sử theo ngày cụ thể với phân trang
     * GET /api/notifications/history/today?page=0&size=20
     */
    @GetMapping("/history/today")
    public ResponseEntity<BaseResponse<List<NotificationResponse>>> getTodayHistory(
            Authentication authentication,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        LocalDateTime startOfDay = LocalDateTime.now().withHour(0).withMinute(0).withSecond(0);
        LocalDateTime endOfDay = LocalDateTime.now().withHour(23).withMinute(59).withSecond(59);
        
        BaseResponse<List<NotificationResponse>> response = 
                notificationService.getHistory(user.getId(), startOfDay, endOfDay, null, page, size);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Lấy lịch sử 7 ngày gần nhất với phân trang
     * GET /api/notifications/history/week?page=0&size=20
     */
    @GetMapping("/history/week")
    public ResponseEntity<BaseResponse<List<NotificationResponse>>> getWeekHistory(
            Authentication authentication,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        LocalDateTime weekAgo = LocalDateTime.now().minusDays(7);
        LocalDateTime now = LocalDateTime.now();
        
        BaseResponse<List<NotificationResponse>> response = 
                notificationService.getHistory(user.getId(), weekAgo, now, null, page, size);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Lấy lịch sử theo trạng thái với phân trang
     * GET /api/notifications/history/status/{status}?page=0&size=20
     */
    @GetMapping("/history/status/{status}")
    public ResponseEntity<BaseResponse<List<NotificationResponse>>> getHistoryByStatus(
            Authentication authentication,
            @PathVariable ENotificationStatus status,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size
    ) {
        User user = userRepository.findByEmail(authentication.getName())
                .orElseThrow(() -> new RuntimeException("User không tồn tại"));
        
        BaseResponse<List<NotificationResponse>> response = 
                notificationService.getHistory(user.getId(), null, null, status, page, size);
        
        return ResponseEntity.ok(response);
    }

}