package com.example.Auto_BE.utils;

import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.service.MedicationReminderJob;
import com.example.Auto_BE.service.NotificationService;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import org.quartz.*;
import org.quartz.impl.matchers.GroupMatcher;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.util.Date;
import java.util.List;

@Component
public class SchedulerReminder {

    @Autowired
    private MedicationReminderRepository medicationReminderRepository;

    @Autowired
    private NotificationService notificationService;

    @Autowired
    private Scheduler scheduler;

    // Chạy mỗi ngày lúc 00:01 để lên lịch cho tuần tiếp theo
    // @Scheduled(cron = "0 1 0 * * *")  // Production: mỗi ngày 00:01
    @Scheduled(cron = "0 * * * * *")     // Test: mỗi phút (để test)
    public void scheduleWeeklyReminders() {
        try {
            System.out.println("=== Starting to schedule weekly reminders ===");
            
            // Kiểm tra Scheduler status
            if (!scheduler.isStarted()) {
                System.err.println("ERROR: Quartz Scheduler is not started!");
                return;
            }
            System.out.println("Quartz Scheduler status: RUNNING");
            
            LocalDate today = LocalDate.now();
            LocalDate endOfWeek = today.plusDays(7);
            
            System.out.println("Scheduling period: " + today + " to " + endOfWeek);
            
            // Lấy tất cả medication reminders đang active và hợp lệ
            List<MedicationReminder> activeReminders;
            try {
                // Sử dụng method mới để tránh lỗi enum
                activeReminders = medicationReminderRepository.findValidActiveReminders();
                System.out.println("Found " + activeReminders.size() + " valid active reminders");
                
                // Fallback: nếu query custom lỗi, dùng method cũ với exception handling
                if (activeReminders.isEmpty()) {
                    System.out.println("Trying fallback method...");
                    activeReminders = medicationReminderRepository.findByIsActiveTrue();
                    System.out.println("Fallback found " + activeReminders.size() + " reminders");
                }
            } catch (Exception e) {
                System.err.println("ERROR loading reminders from database: " + e.getMessage());
                System.err.println("This might be due to invalid enum values in database.");
                System.err.println("Please check medication_reminders table for null or invalid 'type' values.");
                return;
            }
            
            if (activeReminders.isEmpty()) {
                System.out.println("No active reminders found. Skipping...");
                return;
            }
            
            int processedCount = 0;
            int errorCount = 0;
            int skippedCount = 0;
            int totalScheduled = 0;
            
            for (MedicationReminder reminder : activeReminders) {
                try {
                    // Validate reminder data before processing
                    if (reminder.getType() == null) {
                        System.err.println("WARNING: Reminder ID " + reminder.getId() + " has null type. Skipping...");
                        errorCount++;
                        continue;
                    }
                    if (reminder.getDaysOfWeek() == null || reminder.getDaysOfWeek().trim().isEmpty()) {
                        System.err.println("WARNING: Reminder ID " + reminder.getId() + " has null/empty daysOfWeek. Skipping...");
                        errorCount++;
                        continue;
                    }
                    if (reminder.getReminderTime() == null || reminder.getReminderTime().trim().isEmpty()) {
                        System.err.println("WARNING: Reminder ID " + reminder.getId() + " has null/empty reminderTime. Skipping...");
                        errorCount++;
                        continue;
                    }
                    
                    System.out.println("Processing reminder: " + reminder.getName() + " (ID: " + reminder.getId() + ")");
                    int scheduled = scheduleReminderForWeek(reminder, today, endOfWeek);
                    totalScheduled += scheduled;
                    processedCount++;
                    
                } catch (Exception e) {
                    System.err.println("ERROR processing reminder ID " + reminder.getId() + ": " + e.getMessage());
                    errorCount++;
                }
            }
            
            System.out.println("=== Summary ===");
            System.out.println("Total reminders: " + activeReminders.size());
            System.out.println("Successfully processed: " + processedCount);
            System.out.println("Errors/Skipped: " + errorCount);
            System.out.println("Total new notifications scheduled: " + totalScheduled);
            
            System.out.println("=== Finished scheduling weekly reminders ===");
            
        } catch (Exception e) {
            System.err.println("ERROR in scheduleWeeklyReminders: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private int scheduleReminderForWeek(MedicationReminder reminder, LocalDate startDate, LocalDate endDate) {
        try {
            String reminderTimeStr = reminder.getReminderTime(); // "HH:mm"
            LocalTime reminderTime = LocalTime.parse(reminderTimeStr);
            String daysOfWeek = reminder.getDaysOfWeek();
            
            System.out.println("  - Reminder: " + reminder.getName());
            System.out.println("  - Time: " + reminderTimeStr);
            System.out.println("  - Days pattern: " + daysOfWeek);
            
            int scheduledCount = 0;
            int skippedCount = 0;
            
            for (LocalDate currentDate = startDate; currentDate.isBefore(endDate); currentDate = currentDate.plusDays(1)) {
                boolean shouldSchedule = shouldScheduleForDate(reminder, currentDate);
                
                System.out.println("    Date " + currentDate + " (" + currentDate.getDayOfWeek() + "): " + 
                                 (shouldSchedule ? "SCHEDULE" : "SKIP"));
                
                if (shouldSchedule) {
                    LocalDateTime scheduledDateTime = currentDate.atTime(reminderTime);
                    
                    // Chỉ lên lịch cho tương lai
                    if (scheduledDateTime.isAfter(LocalDateTime.now())) {
                        boolean created = createNotificationAndScheduleJob(reminder, scheduledDateTime);
                        if (created) {
                            scheduledCount++;
                        } else {
                            skippedCount++;
                        }
                    } else {
                        System.out.println("      -> Skipped (past time): " + scheduledDateTime);
                        skippedCount++;
                    }
                }
            }
            
            System.out.println("  - New notifications scheduled: " + scheduledCount);
            System.out.println("  - Skipped (already exists): " + skippedCount);
            
            return scheduledCount;
            
        } catch (Exception e) {
            System.err.println("Error scheduling reminder for ID: " + reminder.getId() + " - " + e.getMessage());
            e.printStackTrace();
            return 0;
        }
    }

    private boolean shouldScheduleForDate(MedicationReminder reminder, LocalDate date) {
        String daysOfWeek = reminder.getDaysOfWeek(); // "1111111" hoặc "0111110"
        if (daysOfWeek == null || daysOfWeek.length() != 7) {
            return false;
        }
        
        // Java: MONDAY = 1, SUNDAY = 7
        // daysOfWeek: [0]=Monday, [1]=Tuesday, ..., [6]=Sunday
        int dayOfWeek = date.getDayOfWeek().getValue(); // 1-7
        int index = (dayOfWeek == 7) ? 6 : dayOfWeek - 1; // Convert to 0-6
        
        return daysOfWeek.charAt(index) == '1';
    }

    private boolean createNotificationAndScheduleJob(MedicationReminder reminder, LocalDateTime scheduledDateTime) {
        try {
            System.out.println("      -> Checking notification for: " + scheduledDateTime);
            
            // Kiểm tra xem notification đã tồn tại chưa
            boolean exists = notificationService.existsByMedicationReminderAndTime(reminder.getId(), scheduledDateTime);
            if (exists) {
                System.out.println("      -> Notification already exists, skipping...");
                return false; // Không tạo mới
            }
            
            System.out.println("      -> Creating new notification for: " + scheduledDateTime);
            
            // Tạo notification record trong database
            Notifications notification = new Notifications();
            notification.setUser(reminder.getUser());
            notification.setMedicationReminder(reminder);
            notification.setReminderTime(scheduledDateTime); // LocalDateTime
            notification.setStatus(ENotificationStatus.PENDING);
            notification.setRetryCount(0);
            
            // Lưu vào database
            Notifications savedNotification = notificationService.save(notification);
            System.out.println("      -> Notification saved with ID: " + savedNotification.getId());
            
            // Tạo Quartz job
            String jobId = "reminder-" + reminder.getId() + "-" + scheduledDateTime.toString().replace(":", "-");
            
            // Kiểm tra xem job đã tồn tại chưa
            JobKey jobKey = new JobKey(jobId);
            if (scheduler.checkExists(jobKey)) {
                System.out.println("      -> Job already exists, skipping job creation...");
                return true; // Notification đã tạo nhưng job đã có
            }
            
            JobDetail jobDetail = JobBuilder.newJob(MedicationReminderJob.class)
                    .withIdentity(jobId)
                    .usingJobData("notificationLogId", savedNotification.getId())
                    .build();

            Trigger trigger = TriggerBuilder.newTrigger()
                    .withIdentity("trigger-" + jobId)
                    .startAt(Date.from(scheduledDateTime.atZone(ZoneId.systemDefault()).toInstant()))
                    .build();

            scheduler.scheduleJob(jobDetail, trigger);
            
            System.out.println("      -> Quartz job scheduled: " + jobId);
            System.out.println("      -> Job will execute at: " + scheduledDateTime);
            
            return true; // Thành công tạo mới
            
        } catch (Exception e) {
            System.err.println("ERROR creating notification and job: " + e.getMessage());
            e.printStackTrace();
            return false; // Lỗi
        }
    }

    // Method để lên lịch cho một medication reminder cụ thể (có thể gọi từ API)
    public void scheduleReminderForSpecificWeek(Long medicationReminderId, LocalDate startDate) {
        MedicationReminder reminder = medicationReminderRepository.findById(medicationReminderId)
                .orElseThrow(() -> new RuntimeException("Medication reminder not found"));
        
        LocalDate endDate = startDate.plusDays(7);
        scheduleReminderForWeek(reminder, startDate, endDate);
    }

    // Method để hủy tất cả jobs của một medication reminder
    public void cancelAllJobsForReminder(Long medicationReminderId) {
        try {
            for (String groupName : scheduler.getJobGroupNames()) {
                for (JobKey jobKey : scheduler.getJobKeys(GroupMatcher.jobGroupEquals(groupName))) {
                    if (jobKey.getName().startsWith("reminder-" + medicationReminderId + "-")) {
                        scheduler.deleteJob(jobKey);
                        System.out.println("Cancelled job: " + jobKey.getName());
                    }
                }
            }
        } catch (SchedulerException e) {
            System.err.println("Error cancelling jobs: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Method để debug Scheduler status
    public void checkSchedulerStatus() {
        try {
            System.out.println("=== SCHEDULER STATUS ===");
            System.out.println("Scheduler name: " + scheduler.getSchedulerName());
            System.out.println("Scheduler instance ID: " + scheduler.getSchedulerInstanceId());
            System.out.println("Is started: " + scheduler.isStarted());
            System.out.println("Is in standby: " + scheduler.isInStandbyMode());
            System.out.println("Is shutdown: " + scheduler.isShutdown());
            System.out.println("Number of jobs: " + scheduler.getJobKeys(GroupMatcher.anyGroup()).size());
            System.out.println("=========================");
        } catch (SchedulerException e) {
            System.err.println("Error checking scheduler status: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
