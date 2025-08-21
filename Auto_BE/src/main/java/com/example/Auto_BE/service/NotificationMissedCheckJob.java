package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import org.quartz.Job;
import org.quartz.JobExecutionContext;
import org.quartz.JobExecutionException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * Job để kiểm tra và cập nhật trạng thái uống thuốc MISSED cho các notification 
 * mà user chưa xác nhận đã uống thuốc sau 15 phút
 */
@Component
public class NotificationMissedCheckJob implements Job {

    @Autowired
    private NotificationService notificationService;

    @Override
    public void execute(JobExecutionContext context) throws JobExecutionException {
        Long notificationId = context.getJobDetail().getJobDataMap().getLong("notificationId");
        
        System.out.println("=== Missed Check Job Executing ===");
        System.out.println("Time: " + LocalDateTime.now());
        System.out.println("NotificationId: " + notificationId);

        try {
            // Tìm notification
            Notifications notification = notificationService.findById(notificationId);
            
            if (notification == null) {
                System.err.println("Notification not found: " + notificationId);
                return;
            }

            // Kiểm tra nếu trạng thái uống thuốc vẫn là PENDING thì cập nhật thành MISSED
            if (notification.getStatus() == ENotificationStatus.PENDING) {
                notification.setStatus(ENotificationStatus.MISSED);
                notificationService.save(notification);
                
                System.out.println("Updated notification " + notificationId + 
                                 " to MISSED status (user didn't confirm taking medication)");
            } else {
                System.out.println("User already confirmed taking medication for notification " + notificationId + 
                                 " with status: " + notification.getStatus());
            }

        } catch (Exception e) {
            System.err.println("Error in NotificationMissedCheckJob: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("=== Missed Check Job Finished ===");
    }
}
