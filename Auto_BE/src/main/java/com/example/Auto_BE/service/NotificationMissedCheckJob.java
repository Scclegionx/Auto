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
        try {
            Notifications notification = notificationService.findById(notificationId);
            
            if (notification == null) {
                return;
            }

            if (notification.getStatus() == ENotificationStatus.PENDING) {
                notification.setStatus(ENotificationStatus.MISSED);
                notificationService.save(notification);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
        }
    }
}
