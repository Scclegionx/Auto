package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.*;
import com.example.Auto_BE.entity.enums.EMedicationLogStatus;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.entity.enums.ENotificationType;
import com.example.Auto_BE.repository.ElderSupervisorRepository;
import com.example.Auto_BE.repository.MedicationLogRepository;
import com.example.Auto_BE.repository.NotificationRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class SupervisorAlertService {

    private final MedicationLogRepository medicationLogRepository;
    private final NotificationRepository notificationRepository;
    private final ElderSupervisorRepository elderSupervisorRepository;
    private final FcmService fcmService;

    /**
     * Kiểm tra và cảnh báo các medication bị bỏ lỡ
     * Chạy tự động mỗi 15 phút
     */
    @Scheduled(cron = "0 */15 * * * ?") // Mỗi 15 phút: 00, 15, 30, 45
    @Transactional
    public void checkMissedMedications() {
        log.info("Checking for missed medications...");
        
        try {
            // Tìm các logs PENDING đã quá 30 phút
            LocalDateTime thirtyMinutesAgo = LocalDateTime.now().minusMinutes(30);
            
            List<MedicationLog> overdueLogs = medicationLogRepository.findOverdueLogs(
                    EMedicationLogStatus.PENDING,
                    thirtyMinutesAgo
            );
            
            log.info(" Found {} overdue medication logs", overdueLogs.size());
            
            for (MedicationLog medicationLog : overdueLogs) {
                // Update status to MISSED
                medicationLog.setStatus(EMedicationLogStatus.MISSED);
                medicationLogRepository.save(medicationLog);
                
                log.info(" Medication MISSED: Elder={}, Medications={}, Time={}",
                        medicationLog.getElderUser().getEmail(),
                        medicationLog.getMedicationNames(),
                        medicationLog.getReminderTime());
                
                // Gửi cảnh báo cho Supervisors
                alertSupervisorsAboutMissedMedication(medicationLog);
            }
            
            log.info(" Finished checking missed medications");
            
        } catch (Exception e) {
            log.error(" Error checking missed medications: {}", e.getMessage(), e);
        }
    }

    /**
     * Cảnh báo Supervisors khi Elder bỏ lỡ uống thuốc
     */
    @Transactional
    public void alertSupervisorsAboutMissedMedication(MedicationLog medicationLog) {
        try {
            ElderUser elder = medicationLog.getElderUser();
            
            // Tìm tất cả Supervisors đang quản lý Elder này
            List<ElderSupervisor> supervisorRelations = elderSupervisorRepository
                    .findActiveByElderUserId(elder.getId());
            
            if (supervisorRelations.isEmpty()) {
                log.info("ℹ No active supervisors found for Elder: {}", elder.getEmail());
                return;
            }
            
            log.info("Alerting {} supervisors about missed medication", supervisorRelations.size());
            
            // Tạo notification cho từng Supervisor
            for (ElderSupervisor relation : supervisorRelations) {
                createSupervisorNotification(
                    relation.getSupervisorUser(),
                    elder,
                    medicationLog,
                    ENotificationType.ELDER_MISSED_MEDICATION
                );
            }
            
        } catch (Exception e) {
            log.error(" Error alerting supervisors: {}", e.getMessage(), e);
        }
    }

    /**
     * Cảnh báo khi Elder uống thuốc trễ (>30 phút)
     */
    @Transactional
    public void alertLateMedication(ElderUser elder, MedicationLog medicationLog) {
        try {
            log.info(" Elder took medication late: {} minutes", medicationLog.getMinutesLate());
            
            // Tìm Supervisors của Elder
            List<ElderSupervisor> supervisorRelations = elderSupervisorRepository
                    .findActiveByElderUserId(elder.getId());
            
            if (supervisorRelations.isEmpty()) {
                log.info("️ No active supervisors found for Elder: {}", elder.getEmail());
                return;
            }
            
            // Tạo notification cho từng Supervisor
            for (ElderSupervisor relation : supervisorRelations) {
                createSupervisorNotification(
                    relation.getSupervisorUser(),
                    elder,
                    medicationLog,
                    ENotificationType.ELDER_LATE_MEDICATION
                );
            }
            
        } catch (Exception e) {
            log.error(" Error alerting late medication: {}", e.getMessage(), e);
        }
    }

    /**
     * Tạo notification cho Supervisor về Elder
     */
    private void createSupervisorNotification(
            SupervisorUser supervisor,
            ElderUser elder,
            MedicationLog medicationLog,
            ENotificationType type
    ) {
        try {
            String title = buildNotificationTitle(type, elder);
            String body = buildNotificationBody(type, medicationLog);
            
            Notifications notification = new Notifications();
            notification.setNotificationType(type);
            notification.setUser(supervisor);
            notification.setRelatedElder(elder);
            notification.setRelatedMedicationLog(medicationLog);
            notification.setTitle(title);
            notification.setBody(body);
            notification.setActionUrl("/medication-logs/" + medicationLog.getId());
            notification.setStatus(ENotificationStatus.SENT);
            notification.setIsRead(false);
            
            notificationRepository.save(notification);
            
            // Gửi FCM notification
            sendFcmToSupervisor(supervisor, title, body);
            
            log.info(" Notification created for Supervisor: {}", supervisor.getEmail());
            
        } catch (Exception e) {
            log.error(" Error creating supervisor notification: {}", e.getMessage(), e);
        }
    }

    /**
     * Build notification title
     */
    private String buildNotificationTitle(ENotificationType type, ElderUser elder) {
        String elderName = elder.getFullName() != null ? elder.getFullName() : elder.getEmail();
        
        return switch (type) {
            case ELDER_MISSED_MEDICATION -> elderName + " bỏ lỡ uống thuốc";
            case ELDER_LATE_MEDICATION ->  elderName + " uống thuốc trễ";
            case ELDER_ADHERENCE_LOW ->  elderName + " - Tỷ lệ tuân thủ thấp";
            case ELDER_HEALTH_ALERT -> elderName + " - Cảnh báo sức khỏe";
            default -> "Thông báo về " + elderName;
        };
    }

    /**
     * Build notification body
     */
    private String buildNotificationBody(ENotificationType type, MedicationLog medicationLog) {
        String medications = medicationLog.getMedicationNames();
        String time = medicationLog.getReminderTime().toLocalTime().toString();
        
        return switch (type) {
            case ELDER_MISSED_MEDICATION -> 
                String.format("Chưa uống thuốc: %s vào lúc %s", medications, time);
            case ELDER_LATE_MEDICATION -> 
                String.format("Uống trễ %d phút: %s (dự kiến %s)", 
                    medicationLog.getMinutesLate(), medications, time);
            case ELDER_ADHERENCE_LOW -> 
                "Tỷ lệ tuân thủ uống thuốc thấp hơn 80% trong 7 ngày qua";
            default -> 
                "Vui lòng kiểm tra tình trạng người thân";
        };
    }

    /**
     * Gửi FCM notification cho Supervisor
     */
    private void sendFcmToSupervisor(SupervisorUser supervisor, String title, String body) {
        try {
            // Gửi FCM notification qua FcmService
            fcmService.sendNotificationToUser(supervisor, title, body);
            
            log.info(" FCM sent to Supervisor: {}", supervisor.getEmail());
            
        } catch (Exception e) {
            log.error("Error sending FCM: {}", e.getMessage(), e);
        }
    }

    /**
     * Kiểm tra tỷ lệ tuân thủ và cảnh báo nếu thấp
     */
    @Transactional
    public void checkAdherenceRate(ElderUser elder) {
        try {
            LocalDateTime weekAgo = LocalDateTime.now().minusDays(7);
            LocalDateTime now = LocalDateTime.now();
            
            List<MedicationLog> logs = medicationLogRepository.findByElderUserAndReminderTimeBetween(
                    elder, weekAgo, now
            );
            
            if (logs.isEmpty()) {
                return;
            }
            
            long totalCount = logs.size();
            long takenCount = logs.stream()
                    .filter(medicationLog -> medicationLog.getStatus() == EMedicationLogStatus.TAKEN)
                    .count();
            
            double adherenceRate = (takenCount * 100.0) / totalCount;
            
            log.info("Elder {} adherence rate: {}%", elder.getEmail(), adherenceRate);
            
            // Nếu tỷ lệ tuân thủ < 80%, cảnh báo Supervisors
            if (adherenceRate < 80.0) {
                log.warn(" Low adherence rate detected: {}%", adherenceRate);
                // TODO: Alert supervisors
            }
            
        } catch (Exception e) {
            log.error("Error checking adherence rate: {}", e.getMessage(), e);
        }
    }
}
