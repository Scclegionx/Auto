package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.service.NotificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;

@RestController
@RequestMapping("/api/notifications")
public class NotificationController {
    
    private final NotificationService notificationService;

    public NotificationController(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    @PostMapping("/{notificationId}/confirm-taken")
    public ResponseEntity<BaseResponse<String>> confirmTaken(@PathVariable Long notificationId) {
        
        Notifications notification = notificationService.findById(notificationId);
        if (notification == null) {
            return ResponseEntity.notFound().build();
        }

        // Chỉ cho phép cập nhật nếu đang ở trạng thái PENDING
        if (notification.getStatus() != ENotificationStatus.PENDING) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Cannot confirm - notification already processed")
                    .data("Current status: " + notification.getStatus())
                    .build();
            return ResponseEntity.badRequest().body(response);
        }

        notification.setStatus(ENotificationStatus.TAKEN);
        notificationService.save(notification);

        BaseResponse<String> response = BaseResponse.<String>builder()
                .status(SUCCESS)
                .message("Medication taken confirmed successfully")
                .data("Notification ID: " + notificationId)
                .build();

        return ResponseEntity.ok(response);
    }
}
