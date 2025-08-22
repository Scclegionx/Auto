package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.entity.Notifications;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.NotificationRepository;
import org.quartz.JobKey;
import org.quartz.Scheduler;
import org.quartz.SchedulerException;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Optional;

import static com.example.Auto_BE.constants.ErrorMessages.CONFIRM_ERROR;
import static com.example.Auto_BE.constants.ErrorMessages.NOTIFICATION_NOT_FOUND;
import static com.example.Auto_BE.constants.SuccessMessage.MEDICATION_CONFIRMED;
import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;

@Service
public class NotificationService {
    private final NotificationRepository notificationRepository;
    private final Scheduler scheduler;

    public NotificationService(NotificationRepository notificationRepository, Scheduler scheduler) {
        this.notificationRepository = notificationRepository;
        this.scheduler = scheduler;
    }

    public Notifications findById(Long id) {
        return notificationRepository.findById(id).orElse(null);
    }

    public Notifications save(Notifications log) {
        return notificationRepository.save(log);
    }

    public BaseResponse<String> confirmTaken(Long notificationId) {
        try {
            Notifications notification = findById(notificationId);
            if (notification == null) {
                throw new BaseException.EntityNotFoundException(NOTIFICATION_NOT_FOUND);
            }

            if (notification.getStatus() == ENotificationStatus.TAKEN) {
                return BaseResponse.<String>builder()
                        .status(SUCCESS)
                        .message(MEDICATION_CONFIRMED)
                        .data(null)
                        .build();
            }

            if (notification.getStatus() == ENotificationStatus.MISSED) {
                throw new BaseException.BadRequestException(CONFIRM_ERROR);
            }
            notification.setStatus(ENotificationStatus.TAKEN);
            save(notification);

            // Hủy job kiểm tra missed (nếu có)
            cancelMissedCheckJob(notificationId);

            return BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message(MEDICATION_CONFIRMED)
                    .data(null)
                    .build();

        } catch (BaseException e) {
            throw e;
        } catch (Exception e) {
            throw new BaseException.BadRequestException( e.getMessage());
        }
    }

    private void cancelMissedCheckJob(Long notificationId) {
        try {
            // Tìm và hủy job kiểm tra missed với pattern tên job
            String jobName = "missed-check-" + notificationId;
            String groupName = "missed-check-group";
            
            JobKey jobKey = new JobKey(jobName, groupName);
            if (scheduler.checkExists(jobKey)) {
                scheduler.deleteJob(jobKey);
            }
            
        } catch (SchedulerException e) {
            System.err.println(e.getMessage());
        }
    }
}
