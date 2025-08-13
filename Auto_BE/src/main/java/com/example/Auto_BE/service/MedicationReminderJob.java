package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.google.firebase.messaging.BatchResponse;
import com.google.firebase.messaging.FirebaseMessagingException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.quartz.*;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;
import com.example.Auto_BE.entity.DeviceToken;

@Component
public class MedicationReminderJob implements Job {

    @Autowired
    private NotificationService notificationService;  

    @Autowired
    private FcmService fcmService;

    @Autowired
    private Scheduler scheduler;

    @Override
    public void execute(JobExecutionContext context) throws JobExecutionException {
        System.out.println("[Job] Bắt đầu execute job");
        JobDataMap dataMap = context.getJobDetail().getJobDataMap();
        Long notificationLogId = dataMap.getLong("notificationLogId");
        System.out.println("[Job] notificationLogId = " + notificationLogId);
        // Lấy thông tin notification_log từ DB
        Notifications log = notificationService.findById(notificationLogId);
        System.out.println("[Job] log từ DB = " + log);
        if (log == null || !log.getStatus().equals(ENotificationStatus.PENDING)) {
            System.out.println("[Job] Log null hoặc không ở trạng thái PENDING → hủy job");
            try {
                scheduler.deleteJob(context.getJobDetail().getKey());
            } catch (SchedulerException e) {
                e.printStackTrace();
            }
            return;
        }
        List<String> deviceTokenStrings = log.getUser().getDeviceTokens().stream()
                .map(DeviceToken::getFcmToken) // giả sử mỗi DeviceToken có phương thức getToken()
                .toList();
        System.out.println("[Job] Device tokens = " + deviceTokenStrings);
        boolean sent = false;
        try {
            System.out.println("[Job] Bắt đầu gửi FCM tới " + deviceTokenStrings.size() + " thiết bị...");
            // Gửi notification tới nhiều thiết bị
            BatchResponse response = fcmService.sendNotification(
                    deviceTokenStrings,
                    "Đến giờ uống thuốc",
                    "Bạn có thuốc cần uống lúc " + log.getReminderTime()
            );
            System.out.println("[Job] FCM gửi thành công: " + response.getSuccessCount() +
                    ", thất bại: " + response.getFailureCount());
            sent = response.getSuccessCount() > 0;
        } catch (FirebaseMessagingException e) {
            System.out.println("[Job] Lỗi khi gửi FCM:");
            e.printStackTrace();
        }
        if (sent) {
            // Cập nhật số lần gửi (retry_count) và thời gian gửi
            log.setRetryCount(log.getRetryCount() + 1);
            log.setLastSentTime(LocalDateTime.now());
            notificationService.save(log);
            System.out.println("[Job] Cập nhật retry_count = " + log.getRetryCount());

            if (log.getRetryCount() < 3) {
                // Nếu retry_count < 3, lên lịch lại Job nhắc lại sau 5 phút
                System.out.println("[Job] Lên lịch retry sau 5 phút...");
                try {
                    scheduleRetryJob(notificationLogId, Duration.ofMinutes(5));
                } catch (SchedulerException e) {
                    e.printStackTrace();
                }
            } else {
                // Nếu đã nhắc 3 lần mà user chưa xác nhận, đánh dấu missed
                log.setStatus(ENotificationStatus.MISSED);
                notificationService.save(log);
                System.out.println("[Job] Đánh dấu MISSED và hủy job");
                // Hủy job hiện tại
                try {
                    scheduler.deleteJob(context.getJobDetail().getKey());
                } catch (SchedulerException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void scheduleRetryJob(Long notificationLogId, Duration delay) throws SchedulerException {
        JobDetail jobDetail = JobBuilder.newJob(MedicationReminderJob.class)
                .withIdentity("reminderJobRetry-" + notificationLogId + "-" + System.currentTimeMillis())
                .usingJobData("notificationLogId", notificationLogId)
                .build();

        Trigger trigger = TriggerBuilder.newTrigger()
                .withIdentity("reminderTriggerRetry-" + notificationLogId + "-" + System.currentTimeMillis())
                .startAt(Date.from(Instant.now().plus(delay)))
                .build();

        scheduler.scheduleJob(jobDetail, trigger);
    }
}
