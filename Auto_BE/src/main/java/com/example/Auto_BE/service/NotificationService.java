package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.response.NotificationResponse;
import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.NotificationRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;

@Service
public class NotificationService {
    private final NotificationRepository notificationRepository;

    public NotificationService(NotificationRepository notificationRepository) {
        this.notificationRepository = notificationRepository;
    }

    public Notifications findById(Long id) {
        return notificationRepository.findById(id).orElse(null);
    }

    public Notifications save(Notifications log) {
        return notificationRepository.save(log);
    }
    
    /**
     * Đánh dấu thông báo đã đọc
     */
    public BaseResponse<String> markAsRead(Long notificationId) {
        try {
            Notifications notification = findById(notificationId);
            if (notification == null) {
                throw new BaseException.EntityNotFoundException("Thông báo không tồn tại");
            }
            
            notification.setIsRead(true);
            save(notification);
            
            return BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Đã đánh dấu đã đọc")
                    .data(null)
                    .build();
                    
        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            throw new BaseException.BadRequestException(e.getMessage());
        }
    }
    
    /**
     * Đánh dấu tất cả thông báo của user đã đọc
     */
    public BaseResponse<String> markAllAsRead(Long userId) {
        try {
            List<Notifications> notifications = notificationRepository.findByUserIdAndIsRead(userId, false);
            
            for (Notifications notification : notifications) {
                notification.setIsRead(true);
                save(notification);
            }
            
            return BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Đã đánh dấu tất cả đã đọc")
                    .data(null)
                    .build();
                    
        } catch (Exception e) {
            throw new BaseException.BadRequestException(e.getMessage());
        }
    }
    
    /**
     * Đếm số thông báo chưa đọc
     */
    public Long getUnreadCount(Long userId) {
        return notificationRepository.countUnreadByUserId(userId);
    }
    
    /**
     * Lấy danh sách thông báo của user (general notifications)
     * Note: Medication history đã chuyển sang MedicationLogController
     */
    public BaseResponse<List<NotificationResponse>> getUserNotifications(
            Long userId,
            Boolean isRead,
            int page,
            int size) {
        
        try {
            List<Notifications> notifications;
            
            if (isRead != null) {
                notifications = notificationRepository.findByUserIdAndIsRead(userId, isRead);
            } else {
                notifications = notificationRepository.findByUserId(userId);
            }
            
            // Sort by createdAt desc và phân trang
            notifications = notifications.stream()
                    .sorted((n1, n2) -> n2.getCreatedAt().compareTo(n1.getCreatedAt()))
                    .skip((long) page * size)
                    .limit(size)
                    .collect(Collectors.toList());
            
            // Convert to response DTO
            List<NotificationResponse> responses = notifications.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());
            
            return BaseResponse.<List<NotificationResponse>>builder()
                    .status(SUCCESS)
                    .message("Lấy thông báo thành công")
                    .data(responses)
                    .build();
                    
        } catch (Exception e) {
            throw new BaseException.BadRequestException("Lỗi khi lấy thông báo: " + e.getMessage());
        }
    }
    
    /**
     * Convert Notifications entity sang NotificationResponse DTO
     */
    private NotificationResponse convertToResponse(Notifications notification) {
        // Convert Instant to LocalDateTime
        LocalDateTime createdAt = notification.getCreatedAt() != null 
                ? LocalDateTime.ofInstant(notification.getCreatedAt(), java.time.ZoneId.systemDefault())
                : null;
        LocalDateTime updatedAt = notification.getUpdatedAt() != null
                ? LocalDateTime.ofInstant(notification.getUpdatedAt(), java.time.ZoneId.systemDefault())
                : null;
        
        return NotificationResponse.builder()
                .id(notification.getId())
                .notificationType(notification.getNotificationType())
                .title(notification.getTitle())
                .body(notification.getBody())
                .status(notification.getStatus())
                .isRead(notification.getIsRead())
                .actionUrl(notification.getActionUrl())
                .relatedElderId(notification.getRelatedElder() != null ? notification.getRelatedElder().getId() : null)
                .relatedMedicationLogId(notification.getRelatedMedicationLog() != null ? notification.getRelatedMedicationLog().getId() : null)
                .userId(notification.getUser().getId())
                .userEmail(notification.getUser().getEmail())
                .createdAt(createdAt)
                .updatedAt(updatedAt)
                .build();
    }
}