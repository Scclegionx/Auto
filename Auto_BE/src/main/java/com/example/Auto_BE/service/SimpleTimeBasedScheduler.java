package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import org.quartz.*;
import org.quartz.impl.matchers.GroupMatcher;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class SimpleTimeBasedScheduler {

    @Autowired
    private Scheduler scheduler;

    @Autowired
    private MedicationReminderRepository medicationReminderRepository;

    private static final String JOB_GROUP = "simple-time-reminders";

    /**
     * Lên lịch thông báo đơn giản cho user - chỉ gửi FCM, không track confirm/missed
     */
    public void scheduleUserReminders(Long userId) {
        System.out.println("Scheduling reminders for user: " + userId);
        
        try {
            // Hủy tất cả jobs cũ của user
            cancelUserReminders(userId);

            // Lấy tất cả medications active của user
            List<MedicationReminder> activeMedications = medicationReminderRepository
                    .findAll()
                    .stream()
                    .filter(med -> med.getElderUser().getId().equals(userId) && med.getIsActive())
                    .collect(Collectors.toList());

            if (activeMedications.isEmpty()) {
                System.out.println("No active medications for user: " + userId);
                return;
            }

            // Group theo time slot
            Map<String, List<MedicationReminder>> timeSlots = activeMedications.stream()
                    .collect(Collectors.groupingBy(MedicationReminder::getReminderTime));

            System.out.println("Creating jobs for " + timeSlots.size() + " time slots");

            // Tạo job cho mỗi time slot
            for (Map.Entry<String, List<MedicationReminder>> entry : timeSlots.entrySet()) {
                String timeSlot = entry.getKey();
                List<MedicationReminder> medications = entry.getValue();
                createTimeSlotJob(userId, timeSlot, medications.size());
            }

            System.out.println("Successfully scheduled " + timeSlots.size() + " jobs for user: " + userId);

        } catch (Exception e) {
            System.err.println("Error scheduling user reminders: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void createTimeSlotJob(Long userId, String timeSlot, int medicationCount) {
        try {
            String jobId = "user-" + userId + "-" + timeSlot.replace(":", "");
            
            System.out.println("Creating job: " + jobId + " (" + medicationCount + " medications)");

            JobDetail jobDetail = JobBuilder.newJob(SimpleTimeBasedReminderJob.class)
                    .withIdentity(jobId, JOB_GROUP)
                    .usingJobData("userId", userId)
                    .usingJobData("timeSlot", timeSlot)
                    .build();

            // Tạo cron expression: "08:30" → "0 30 8 * * ?"
            String cronExpression = buildCronExpression(timeSlot);
            
            Trigger trigger = TriggerBuilder.newTrigger()
                    .withIdentity("trigger-" + jobId, JOB_GROUP)
                    .withSchedule(CronScheduleBuilder.cronSchedule(cronExpression))
                    .build();

            scheduler.scheduleJob(jobDetail, trigger);
            System.out.println("Job created with cron: " + cronExpression);

        } catch (SchedulerException e) {
            System.err.println("Failed to create job for " + timeSlot + ": " + e.getMessage());
        }
    }

    private String buildCronExpression(String timeSlot) {
        // timeSlot: "08:30" → gửi thông báo TRƯỚC 10 phút → "08:20"
        // cron: "0 20 8 * * ?"
        String[] parts = timeSlot.split(":");
        int hour = Integer.parseInt(parts[0]);
        int minute = Integer.parseInt(parts[1]);
        
        // Trừ 10 phút
        minute -= 10;
        if (minute < 0) {
            minute += 60;
            hour -= 1;
            if (hour < 0) {
                hour = 23; // Wrap around to previous day
            }
        }
        
        return String.format("0 %d %d * * ?", minute, hour);
    }

    /**
     * Hủy tất cả reminders của user
     */
    public void cancelUserReminders(Long userId) {
        try {
            Set<JobKey> allJobs = scheduler.getJobKeys(GroupMatcher.groupEquals(JOB_GROUP));
            String userPrefix = "user-" + userId + "-";
            
            List<JobKey> userJobs = allJobs.stream()
                    .filter(jobKey -> jobKey.getName().startsWith(userPrefix))
                    .collect(Collectors.toList());
            
            for (JobKey jobKey : userJobs) {
                scheduler.deleteJob(jobKey);
                System.out.println("Canceled: " + jobKey.getName());
            }
            
            if (!userJobs.isEmpty()) {
                System.out.println("Canceled " + userJobs.size() + " jobs for user: " + userId);
            }

        } catch (SchedulerException e) {
            System.err.println("Error canceling jobs for user: " + userId);
        }
    }

    /**
     * Xem jobs đang active của user
     */
    public List<String> getUserActiveJobs(Long userId) {
        try {
            Set<JobKey> allJobs = scheduler.getJobKeys(GroupMatcher.groupEquals(JOB_GROUP));
            String userPrefix = "user-" + userId + "-";
            
            return allJobs.stream()
                    .map(JobKey::getName)
                    .filter(name -> name.startsWith(userPrefix))
                    .collect(Collectors.toList());
                    
        } catch (SchedulerException e) {
            System.err.println("Error getting jobs for user: " + userId);
            return Collections.emptyList();
        }
    }

    /**
     * Thống kê jobs
     */
    public void printJobStats() {
        try {
            Set<JobKey> allJobs = scheduler.getJobKeys(GroupMatcher.groupEquals(JOB_GROUP));
            System.out.println("SIMPLE REMINDER STATS:");
            System.out.println("   Total active jobs: " + allJobs.size());
            
        } catch (SchedulerException e) {
            System.err.println("Error getting job stats");
        }
    }
}