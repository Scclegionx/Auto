package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import org.quartz.*;
import org.quartz.impl.matchers.GroupMatcher;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Set;

@Service
public class CronSchedulerService {

    @Autowired
    private MedicationReminderRepository medicationReminderRepository;

    @Autowired
    private Scheduler scheduler;

    /**
     * Tạo lịch nhắc bằng cron expression cho medication reminder
     */
    public void scheduleWithCron(Long medicationReminderId) {
        try {
            MedicationReminder reminder = medicationReminderRepository.findById(medicationReminderId)
                    .orElseThrow(() -> new RuntimeException("Medication reminder not found"));

            // Validate dữ liệu
            if (reminder.getReminderTime() == null || reminder.getReminderTime().trim().isEmpty() ||
                reminder.getDaysOfWeek() == null || reminder.getDaysOfWeek().trim().isEmpty()) {
                System.err.println("Invalid reminder data for ID: " + medicationReminderId);
                return;
            }

            // Parse thời gian nhắc
            LocalTime reminderTime = LocalTime.parse(reminder.getReminderTime(), DateTimeFormatter.ofPattern("HH:mm"));
            
            // Tạo cron expression từ daysOfWeek và reminderTime
            String cronExpression = buildCronExpression(reminder.getDaysOfWeek(), reminderTime);
            
            System.out.println("=== Creating cron job for reminder ===");
            System.out.println("Reminder: " + reminder.getName() + " (ID: " + reminder.getId() + ")");
            System.out.println("Cron Expression: " + cronExpression);

            // Tạo JobDetail
            String jobId = "cronReminder-" + reminder.getId();
            JobDetail jobDetail = JobBuilder.newJob(CronMedicationReminderJob.class)
                    .withIdentity(jobId, "CRON_REMINDERS")
                    .usingJobData("medicationReminderId", reminder.getId())
                    .storeDurably(true)
                    .build();

            // Tạo CronTrigger
            CronTrigger trigger = TriggerBuilder.newTrigger()
                    .withIdentity("cronTrigger-" + reminder.getId(), "CRON_REMINDERS")
                    .withSchedule(CronScheduleBuilder.cronSchedule(cronExpression))
                    .build();

            // Schedule job
            if (scheduler.checkExists(jobDetail.getKey())) {
                scheduler.deleteJob(jobDetail.getKey());
                System.out.println("Deleted existing job: " + jobId);
            }

            scheduler.scheduleJob(jobDetail, trigger);
            System.out.println("Successfully scheduled cron job: " + jobId);
            System.out.println("=== Finished creating cron job ===");

        } catch (Exception e) {
            System.err.println("ERROR in scheduleWithCron: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Tạo cron expression từ daysOfWeek và reminderTime
     * @param daysOfWeek "1111100" (T2-T6)
     * @param reminderTime 08:30
     * @return "0 30 8 ? * MON-FRI"
     */
    private String buildCronExpression(String daysOfWeek, LocalTime reminderTime) {
        // Cron format: second minute hour day-of-month month day-of-week
        
        int hour = reminderTime.getHour();
        int minute = reminderTime.getMinute();
        
        // Chuyển đổi daysOfWeek từ "1111100" sang "MON,TUE,WED,THU,FRI"
        String daysPart = convertDaysOfWeekToCron(daysOfWeek);
        
        // Tạo cron expression
        return String.format("0 %d %d ? * %s", minute, hour, daysPart);
    }

    /**
     * Chuyển đổi daysOfWeek pattern sang cron day format
     * @param daysOfWeek "1111100" 
     * @return "MON,TUE,WED,THU,FRI" hoặc "MON,WED,FRI" hoặc "*"
     */
    private String convertDaysOfWeekToCron(String daysOfWeek) {
        if (daysOfWeek.length() != 7) {
            throw new IllegalArgumentException("daysOfWeek must be 7 characters");
        }

        String[] cronDays = {"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"};
        StringBuilder result = new StringBuilder();

        // Thu thập các ngày được chọn
        for (int i = 0; i < 7; i++) {
            if (daysOfWeek.charAt(i) == '1') {
                if (result.length() > 0) {
                    result.append(",");
                }
                result.append(cronDays[i]);
            }
        }

        String daysList = result.toString();
        
        if (daysList.isEmpty()) {
            throw new IllegalArgumentException("At least one day must be selected");
        }
        
        // Chỉ có 1 trường hợp đặc biệt: tất cả 7 ngày
        if (daysList.equals("MON,TUE,WED,THU,FRI,SAT,SUN")) {
            return "*";
        }
        
        // Tất cả trường hợp khác đều dùng cách liệt kê
        return daysList;
    }

    /**
     * Hủy lịch nhắc cho medication reminder
     */
    public void cancelCronSchedule(Long medicationReminderId) {
        try {
            String jobId = "cronReminder-" + medicationReminderId;
            JobKey jobKey = new JobKey(jobId, "CRON_REMINDERS");
            
            if (scheduler.checkExists(jobKey)) {
                scheduler.deleteJob(jobKey);
                System.out.println("Cancelled cron job: " + jobId);
            } else {
                System.out.println("Cron job not found: " + jobId);
            }
        } catch (SchedulerException e) {
            System.err.println("Error cancelling cron job: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Tạo lịch cho tất cả medication reminders đang active
     */
    public void scheduleAllActiveReminders() {
        try {
            var activeReminders = medicationReminderRepository.findByIsActiveTrue();
            System.out.println("Scheduling cron jobs for " + activeReminders.size() + " active reminders");
            
            for (var reminder : activeReminders) {
                scheduleWithCron(reminder.getId());
            }
            
            System.out.println("Finished scheduling all active reminders");
        } catch (Exception e) {
            System.err.println("Error scheduling all reminders: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Debug: Liệt kê tất cả cron jobs
     */
    public void listAllCronJobs() {
        try {
            var jobKeys = scheduler.getJobKeys(GroupMatcher.jobGroupEquals("CRON_REMINDERS"));
            System.out.println("=== All Cron Jobs ===");
            
            for (JobKey jobKey : jobKeys) {
                var jobDetail = scheduler.getJobDetail(jobKey);
                var triggers = scheduler.getTriggersOfJob(jobKey);
                
                System.out.println("Job: " + jobKey.getName());
                System.out.println("  MedicationReminderId: " + 
                    jobDetail.getJobDataMap().getLong("medicationReminderId"));
                
                for (Trigger trigger : triggers) {
                    if (trigger instanceof CronTrigger) {
                        CronTrigger cronTrigger = (CronTrigger) trigger;
                        System.out.println("  Cron: " + cronTrigger.getCronExpression());
                        System.out.println("  Next Fire: " + trigger.getNextFireTime());
                    }
                }
                System.out.println();
            }
            
        } catch (SchedulerException e) {
            System.err.println("Error listing cron jobs: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
