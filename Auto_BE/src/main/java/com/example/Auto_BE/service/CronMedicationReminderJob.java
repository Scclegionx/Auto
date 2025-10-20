//package com.example.Auto_BE.service;
//
//import com.example.Auto_BE.entity.DeviceToken;
//import com.example.Auto_BE.entity.MedicationReminder;
//import com.example.Auto_BE.entity.Notifications;
//import com.example.Auto_BE.entity.enums.ENotificationStatus;
//import com.example.Auto_BE.repository.MedicationReminderRepository;
//import com.google.firebase.messaging.BatchResponse;
//import com.google.firebase.messaging.FirebaseMessagingException;
//import org.quartz.*;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.stereotype.Component;
//
//import java.time.LocalDateTime;
//import java.util.List;
//
//@Component
//public class CronMedicationReminderJob implements Job {
//
//    @Autowired
//    private MedicationReminderRepository medicationReminderRepository;
//
//    @Autowired
//    private NotificationService notificationService;
//
//    @Autowired
//    private FcmService fcmService;
//
//    @Autowired
//    private Scheduler scheduler;
//
//    @Override
//    public void execute(JobExecutionContext context) throws JobExecutionException {
//        Long medicationReminderId = context.getJobDetail().getJobDataMap().getLong("medicationReminderId");
//        System.out.println("=== Executing CronMedicationReminderJob for ID: " + medicationReminderId + " ===");
//
//        try {
//            MedicationReminder reminder = medicationReminderRepository.findByIdWithUserAndDeviceTokens(medicationReminderId)
//                    .orElse(null);
//
//            if (reminder == null) {
//                System.out.println("❌ MedicationReminder not found for id: " + medicationReminderId);
//                return;
//            }
//
//            if (!reminder.getIsActive()) {
//                System.out.println("⏸️ MedicationReminder is inactive for: " + reminder.getName());
//                return;
//            }
//
//            System.out.println("💊 Processing medication: " + reminder.getName() + " at " + reminder.getReminderTime());
//
//            Notifications notificationLog = createNotificationLog(reminder);
//
//            // Lấy FCM tokens của user
//            List<String> deviceTokens = reminder.getUser().getDeviceTokens()
//                    .stream()
//                    .map(DeviceToken::getFcmToken)
//                    .toList();
//
//            System.out.println("📱 Found " + deviceTokens.size() + " device tokens for user: " + reminder.getUser().getEmail());
//
//            if (deviceTokens.isEmpty()) {
//                System.out.println("⚠️ No device tokens found for user, saving notification without FCM");
//                notificationService.save(notificationLog);
//                scheduleMissedCheckJob(notificationLog.getId());
//                return;
//            }
//            boolean sent = sendFcmNotification(reminder, deviceTokens, notificationLog);
//
//            if (sent) {
//                System.out.println("✅ FCM sent successfully for reminder: " + reminder.getName());
//            } else {
//                System.err.println("❌ Failed to send FCM for reminder: " + reminder.getName());
//            }
//
//            System.out.println("⏰ Scheduling missed check job for notification ID: " + notificationLog.getId());
//            scheduleMissedCheckJob(notificationLog.getId());
//
//            System.out.println("=== Completed CronMedicationReminderJob for ID: " + medicationReminderId + " ===\n");
//
//        } catch (Exception e) {
//            System.err.println("💥 Error in CronMedicationReminderJob for ID: " + medicationReminderId);
//            e.printStackTrace();
//        }
//    }
//
//    private Notifications createNotificationLog(MedicationReminder reminder) {
//        Notifications log = new Notifications();
//        log.setUser(reminder.getUser());
//        log.setMedicationReminder(reminder);
//        log.setReminderTime(LocalDateTime.now());
//        log.setStatus(ENotificationStatus.PENDING);
//        log.setRetryCount(1); // Cron job không retry, chỉ gửi 1 lần
//
//        return notificationService.save(log);
//    }
//
//    private boolean sendFcmNotification(MedicationReminder reminder, List<String> deviceTokens, Notifications log) {
//        System.out.println("🔔 Starting FCM notification process for: " + reminder.getName());
//
//        try {
//            // Lấy tất cả thuốc cùng thời gian
//            List<MedicationReminder> sameTimeReminders = medicationReminderRepository
//                    .findByUserIdAndReminderTime(reminder.getUser().getId(), reminder.getReminderTime());
//
//            System.out.println("🔍 Found " + sameTimeReminders.size() + " medications at time: " + reminder.getReminderTime());
//
//            // Chỉ gửi FCM nếu đây là thuốc có ID nhỏ nhất (tránh duplicate)
//            Long smallestId = sameTimeReminders.stream()
//                    .mapToLong(MedicationReminder::getId)
//                    .min()
//                    .orElse(reminder.getId());
//
//            if (!reminder.getId().equals(smallestId)) {
//                System.out.println("🚫 Skipping FCM for " + reminder.getName() + " (ID: " + reminder.getId() +
//                                 ") - will be sent by medication ID: " + smallestId);
//                log.setLastSentTime(LocalDateTime.now());
//                notificationService.save(log);
//                return true; // Trả về true vì thông báo sẽ được gửi bởi job khác
//            }
//
//            System.out.println("✅ This is the primary job (ID: " + reminder.getId() + ") - proceeding with FCM");
//
//            String title = "⏰ Đến giờ uống thuốc";
//            String body = buildGroupedNotificationBody(sameTimeReminders);
//
//            System.out.println("📝 Notification content:");
//            System.out.println("Title: " + title);
//            System.out.println("Body: " + body.substring(0, Math.min(body.length(), 100)) + (body.length() > 100 ? "..." : ""));
//            System.out.println("🎯 Sending to " + deviceTokens.size() + " device(s)");
//
//            BatchResponse response = fcmService.sendNotification(deviceTokens, title, body);
//
//            System.out.println("📊 FCM Response - Success: " + response.getSuccessCount() + ", Failed: " + response.getFailureCount());
//
//            boolean success = response.getSuccessCount() > 0;
//            log.setLastSentTime(LocalDateTime.now());
//            notificationService.save(log);
//
//            System.out.println("💾 Notification log updated with lastSentTime");
//            return success;
//
//        } catch (FirebaseMessagingException e) {
//            System.err.println("🚨 FirebaseMessagingException in sendFcmNotification: " + e.getMessage());
//            log.setLastSentTime(LocalDateTime.now());
//            notificationService.save(log);
//            return false;
//        } catch (Exception e) {
//            System.err.println("💥 Unexpected error in sendFcmNotification: " + e.getMessage());
//            e.printStackTrace();
//            return false;
//        }
//    }
//
//    private String buildGroupedNotificationBody(List<MedicationReminder> reminders) {
//        System.out.println("📋 Building notification body for " + reminders.size() + " medication(s)");
//
//        if (reminders.size() == 1) {
//            MedicationReminder reminder = reminders.get(0);
//            System.out.println("📝 Single medication format for: " + reminder.getName());
//            return String.format("💊 %s lúc %s\n📝 %s",
//                    reminder.getName(),
//                    reminder.getReminderTime(),
//                    reminder.getDescription() != null ? reminder.getDescription() : "Uống theo chỉ dẫn");
//        }
//
//        System.out.println("📋 Multi-medication format - creating grouped notification");
//        StringBuilder body = new StringBuilder();
//        body.append(String.format("Bạn cần uống %d loại thuốc:\n\n", reminders.size()));
//
//        for (int i = 0; i < reminders.size(); i++) {
//            MedicationReminder reminder = reminders.get(i);
//            System.out.println("  " + (i + 1) + ". Adding medication: " + reminder.getName());
//            body.append(String.format("%d. 💊 %s\n", (i + 1), reminder.getName()));
//
//            if (reminder.getDescription() != null && !reminder.getDescription().trim().isEmpty()) {
//                System.out.println("     ↳ With description: " + reminder.getDescription());
//                body.append(String.format("   📝 %s\n", reminder.getDescription()));
//            } else {
//                System.out.println("     ↳ No description provided");
//            }
//
//            if (i < reminders.size() - 1) {
//                body.append("\n");
//            }
//        }
//
//        body.append("\n⚡ Nhớ uống đúng giờ để đảm bảo hiệu quả điều trị!");
//
//        String finalBody = body.toString();
//        System.out.println("✅ Notification body built successfully (" + finalBody.length() + " characters)");
//        return finalBody;
//    }
//
//    /**
//     * Lên lịch job kiểm tra sau 15 phút để cập nhật trạng thái uống thuốc MISSED
//     * nếu user chưa xác nhận đã uống thuốc
//     */
//    private void scheduleMissedCheckJob(Long notificationId) {
//        try {
//            // Tạo job để check missed status sau 15 phút
//            JobDetail jobDetail = JobBuilder.newJob(NotificationMissedCheckJob.class)
//                    .withIdentity("missed-check-" + notificationId, "missed-check-group")
//                    .usingJobData("notificationId", notificationId)
//                    .build();
//
//            // Trigger sau 15 phút
//            Trigger trigger = TriggerBuilder.newTrigger()
//                    .withIdentity("missed-trigger-" + notificationId, "missed-check-group")
//                    .startAt(new java.util.Date(System.currentTimeMillis() + 15 * 60 * 1000)) // 15 minutes
//                    .build();
//
//            scheduler.scheduleJob(jobDetail, trigger);
//        } catch (SchedulerException e) {
//            e.printStackTrace();
//        }
//    }
//}