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
        
        System.out.println("=== Cron Job Executing ===");
        System.out.println("Time: " + LocalDateTime.now());
        System.out.println("MedicationReminderId: " + medicationReminderId);

        try {
            // 1. Load medication reminder với eager loading user và deviceTokens
            MedicationReminder reminder = medicationReminderRepository.findByIdWithUserAndDeviceTokens(medicationReminderId)
                    .orElse(null);

            if (reminder == null) {
                System.err.println("Medication reminder not found: " + medicationReminderId);
                return;
            }

            if (!reminder.getIsActive()) {
                System.out.println("Medication reminder is inactive, skipping: " + medicationReminderId);
                return;
            }

            System.out.println("Processing reminder: " + reminder.getName());

            // 2. Tạo notification log để tracking
            Notifications notificationLog = createNotificationLog(reminder);
            
            // 3. Lấy FCM tokens của user
            List<String> deviceTokens = reminder.getUser().getDeviceTokens()
                    .stream()
                    .map(DeviceToken::getFcmToken)
                    .toList();

            if (deviceTokens.isEmpty()) {
                System.out.println("No device tokens found for user: " + reminder.getUser().getEmail());
                // Không có device token thì vẫn tạo notification với status PENDING
                // Vì đây là trạng thái uống thuốc, không phải trạng thái gửi FCM
                notificationService.save(notificationLog);
                // Vẫn lên lịch kiểm tra sau 15 phút
                scheduleMissedCheckJob(notificationLog.getId());
                return;
            }

            System.out.println("Sending FCM to " + deviceTokens.size() + " devices");

            // 4. Gửi FCM notification (không ảnh hưởng đến trạng thái uống thuốc)
            boolean sent = sendFcmNotification(reminder, deviceTokens, notificationLog);

            if (sent) {
                System.out.println("FCM sent successfully for reminder: " + reminder.getName());
            } else {
                System.err.println("Failed to send FCM for reminder: " + reminder.getName());
            }
            
            // 5. Luôn lên lịch job kiểm tra sau 15 phút (bất kể FCM có gửi thành công hay không)
            // Vì trạng thái uống thuốc không phụ thuộc vào việc gửi FCM thành công
            scheduleMissedCheckJob(notificationLog.getId());

        } catch (Exception e) {
            System.err.println("Error in CronMedicationReminderJob: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("=== Cron Job Finished ===");
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
            
            // Chỉ cập nhật thời gian gửi FCM, không thay đổi status uống thuốc
            // Status của notification là trạng thái uống thuốc, không phải trạng thái gửi FCM
            log.setLastSentTime(LocalDateTime.now());
            notificationService.save(log);
            
            System.out.println("FCM Result - Success: " + response.getSuccessCount() + 
                             ", Failed: " + response.getFailureCount());
            
            return success;
            
        } catch (FirebaseMessagingException e) {
            System.err.println("FCM Error: " + e.getMessage());
            
            // Cập nhật thời gian gửi FCM (thất bại), nhưng không thay đổi status uống thuốc
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
            
            System.out.println("Scheduled missed check job for notification: " + notificationId + 
                             " to check medication taking status at " + LocalDateTime.now().plusMinutes(15));
            
        } catch (SchedulerException e) {
            System.err.println("Error scheduling missed check job: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
