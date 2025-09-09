package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import org.quartz.*;
import org.quartz.impl.matchers.GroupMatcher;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Set;

import static com.example.Auto_BE.constants.ErrorMessages.DAY_ERROR;
import static com.example.Auto_BE.constants.ErrorMessages.MEDICATION_NOT_FOUND;

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
                    .orElseThrow(() -> new BaseException.BadRequestException(MEDICATION_NOT_FOUND));

            // Validate dữ liệu
            if (reminder.getReminderTime() == null || reminder.getReminderTime().trim().isEmpty() ||
                    reminder.getDaysOfWeek() == null || reminder.getDaysOfWeek().trim().isEmpty()) {
                return;
            }

            // Parse thời gian nhắc
            LocalTime reminderTime = LocalTime.parse(reminder.getReminderTime(), DateTimeFormatter.ofPattern("HH:mm"));

            // Tạo cron expression từ daysOfWeek và reminderTime
            String cronExpression = buildCronExpression(reminder.getDaysOfWeek(), reminderTime);

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
                    .withSchedule(CronScheduleBuilder.cronSchedule(cronExpression)
                            .inTimeZone(java.util.TimeZone.getTimeZone("Asia/Ho_Chi_Minh")))
                    .build();

            // Schedule job
            if (scheduler.checkExists(jobDetail.getKey())) {
                scheduler.deleteJob(jobDetail.getKey());
            }

            scheduler.scheduleJob(jobDetail, trigger);
        } catch (Exception e) {
            System.err.println(e.getMessage());
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
            throw new IllegalArgumentException(DAY_ERROR);
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

        if (daysList.equals("MON,TUE,WED,THU,FRI,SAT,SUN")) {
            return "*";
        }
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
            }
        } catch (SchedulerException e) {
            System.err.println( e.getMessage());
            e.printStackTrace();
        }
    }

}