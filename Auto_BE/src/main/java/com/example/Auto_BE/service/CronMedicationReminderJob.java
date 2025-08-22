package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.DeviceToken;
import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import com.google.firebase.messaging.BatchResponse;
import com.google.firebase.messaging.FirebaseMessagingException;
import org.quartz.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.List;

@Component
public class CronMedicationReminderJob implements Job {

    @Autowired
    private MedicationReminderRepository medicationReminderRepository;

    @Autowired
    private NotificationService notificationService;

    @Autowired
    private FcmService fcmService;

    @Autowired
    private Scheduler scheduler;

    @Override
    public void execute(JobExecutionContext context) throws JobExecutionException {
        Long medicationReminderId = context.getJobDetail().getJobDataMap().getLong("medicationReminderId");
        try {
            MedicationReminder reminder = medicationReminderRepository.findByIdWithUserAndDeviceTokens(medicationReminderId)
                    .orElse(null);

            if (reminder == null) {
                return;
            }

            if (!reminder.getIsActive()) {
                return;
            }

            Notifications notificationLog = createNotificationLog(reminder);
            
            // Lấy FCM tokens của user
            List<String> deviceTokens = reminder.getUser().getDeviceTokens()
                    .stream()
                    .map(DeviceToken::getFcmToken)
                    .toList();

            if (deviceTokens.isEmpty()) {
                notificationService.save(notificationLog);
                scheduleMissedCheckJob(notificationLog.getId());
                return;
            }
            boolean sent = sendFcmNotification(reminder, deviceTokens, notificationLog);

            if (sent) {
                System.out.println("FCM sent successfully for reminder: " + reminder.getName());
            } else {
                System.err.println("Failed to send FCM for reminder: " + reminder.getName());
            }
            scheduleMissedCheckJob(notificationLog.getId());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Notifications createNotificationLog(MedicationReminder reminder) {
        Notifications log = new Notifications();
        log.setUser(reminder.getUser());
        log.setMedicationReminder(reminder);
        log.setReminderTime(LocalDateTime.now());
        log.setStatus(ENotificationStatus.PENDING);
        log.setRetryCount(1); // Cron job không retry, chỉ gửi 1 lần
        
        return notificationService.save(log);
    }

    private boolean sendFcmNotification(MedicationReminder reminder, List<String> deviceTokens, Notifications log) {
        try {
            String title = "Đến giờ uống thuốc";
            String body = String.format("Đã đến giờ uống %s lúc %s", 
                    reminder.getName(), 
                    reminder.getReminderTime());

            BatchResponse response = fcmService.sendNotification(deviceTokens, title, body);
            
            boolean success = response.getSuccessCount() > 0;
            log.setLastSentTime(LocalDateTime.now());
            notificationService.save(log);
            return success;
            
        } catch (FirebaseMessagingException e) {
            log.setLastSentTime(LocalDateTime.now());
            notificationService.save(log);
            return false;
        }
    }

    /**
     * Lên lịch job kiểm tra sau 15 phút để cập nhật trạng thái uống thuốc MISSED 
     * nếu user chưa xác nhận đã uống thuốc
     */
    private void scheduleMissedCheckJob(Long notificationId) {
        try {
            // Tạo job để check missed status sau 15 phút
            JobDetail jobDetail = JobBuilder.newJob(NotificationMissedCheckJob.class)
                    .withIdentity("missed-check-" + notificationId, "missed-check-group")
                    .usingJobData("notificationId", notificationId)
                    .build();

            // Trigger sau 15 phút
            Trigger trigger = TriggerBuilder.newTrigger()
                    .withIdentity("missed-trigger-" + notificationId, "missed-check-group")
                    .startAt(new java.util.Date(System.currentTimeMillis() + 15 * 60 * 1000)) // 15 minutes
                    .build();

            scheduler.scheduleJob(jobDetail, trigger);
        } catch (SchedulerException e) {
            e.printStackTrace();
        }
    }
}
