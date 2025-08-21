//package com.example.Auto_BE.service;
//
//import com.example.Auto_BE.entity.Notifications;
//import org.quartz.*;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.stereotype.Service;
//
//import java.time.ZoneId;
//import java.util.Date;
//
//@Service
//public class NotificationSchedulerService {
//
//    @Autowired
//    private Scheduler scheduler;
//
//    public void scheduleNotificationJob(Notifications log) throws SchedulerException {
//        JobDetail jobDetail = JobBuilder.newJob(MedicationReminderJob.class)
//                .withIdentity("reminderJob-" + log.getId())
//                .usingJobData("notificationLogId", log.getId())
//                .build();
//
//        Trigger trigger = TriggerBuilder.newTrigger()
//                .withIdentity("reminderTrigger-" + log.getId())
//                .startAt(Date.from(log.getReminderTime().atZone(ZoneId.systemDefault()).toInstant()))
//                .build();
//
//        scheduler.scheduleJob(jobDetail, trigger);
//    }
//}
