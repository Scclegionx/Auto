package com.example.Auto_BE.controller;

import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.service.NotificationService;
import org.quartz.JobDetail;
import org.quartz.JobKey;
import org.quartz.Scheduler;
import org.quartz.SchedulerException;
import org.quartz.impl.matchers.GroupMatcher;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/medication")
public class MedicationController {

    @Autowired
    private NotificationService notificationLogService;

    @Autowired
    private Scheduler scheduler;

    @PostMapping("/confirm/{notificationId}")
    public ResponseEntity<?> confirmTaken(@PathVariable Long notificationId) {
        Notifications log = notificationLogService.findById(notificationId);
        if (log == null) {
            return ResponseEntity.notFound().build();
        }

        log.setStatus(ENotificationStatus.TAKEN);
        notificationLogService.save(log);

        // Hủy các job nhắc lại còn tồn tại
        try {
            for (String groupName : scheduler.getJobGroupNames()) {
                for (JobKey jobKey : scheduler.getJobKeys(GroupMatcher.jobGroupEquals(groupName))) {
                    JobDetail jobDetail = scheduler.getJobDetail(jobKey);
                    if (jobDetail.getJobDataMap().getLong("notificationLogId") == notificationId) {
                        scheduler.deleteJob(jobKey);
                    }
                }
            }
        } catch (SchedulerException e) {
            e.printStackTrace();
        }

        return ResponseEntity.ok().build();
    }
}
