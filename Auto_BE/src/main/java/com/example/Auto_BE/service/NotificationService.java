package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.response.NotificationResponse;
import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
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
    
    // ============= LỊCH SỬ THÔNG BÁO =============
    
    /**
     * Lấy lịch sử thông báo với filter
     */
    public BaseResponse<List<NotificationResponse>> getHistory(
            Long userId,
            LocalDateTime startDate,
            LocalDateTime endDate,
            ENotificationStatus status,
            int page,
            int size) {
        
        try {
            List<Notifications> notifications;
            
            // Nếu có đầy đủ filters, dùng query tổng hợp
            if (startDate != null || endDate != null || status != null) {
                notifications = notificationRepository.findByFilters(
                        userId, startDate, endDate, status
                );
            } else {
                // Nếu không có filter nào, lấy tất cả của user
                notifications = notificationRepository.findByUserId(userId);
            }
            
            // ✅ Phân trang thủ công (sort by reminderTime desc)
            notifications = notifications.stream()
                    .sorted((n1, n2) -> n2.getReminderTime().compareTo(n1.getReminderTime()))
                    .skip((long) page * size)
                    .limit(size)
                    .collect(Collectors.toList());
            
            // Convert to response DTO
            List<NotificationResponse> responses = notifications.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());
            
            return BaseResponse.<List<NotificationResponse>>builder()
                    .status(SUCCESS)
                    .message("Lấy lịch sử thông báo thành công (trang " + page + ", " + responses.size() + " items)")
                    .data(responses)
                    .build();
                    
        } catch (Exception e) {
            throw new BaseException.BadRequestException("Lỗi khi lấy lịch sử: " + e.getMessage());
        }
    }
    
    /**
     * Convert Notifications entity sang NotificationResponse DTO
     */
    private NotificationResponse convertToResponse(Notifications notification) {
        return NotificationResponse.builder()
                .id(notification.getId())
                .reminderTime(notification.getReminderTime())
                .lastSentTime(notification.getLastSentTime())
                .status(notification.getStatus())
                .isRead(notification.getIsRead())
                .title(notification.getTitle())
                .body(notification.getBody())
                .medicationCount(notification.getMedicationCount())
                .medicationIds(notification.getMedicationIds())
                .medicationNames(notification.getMedicationNames())
                .userId(notification.getUser().getId())
                .userEmail(notification.getUser().getEmail())
                .build();
    }
}