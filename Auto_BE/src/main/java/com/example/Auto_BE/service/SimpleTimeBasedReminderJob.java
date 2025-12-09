package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.*;
import com.example.Auto_BE.entity.enums.EMedicationLogStatus;
import com.example.Auto_BE.entity.enums.ENotificationStatus;
import com.example.Auto_BE.entity.enums.ENotificationType;
import com.example.Auto_BE.repository.MedicationLogRepository;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import com.example.Auto_BE.repository.UserRepository;
import com.example.Auto_BE.repository.DeviceTokenRepository;
import com.google.firebase.messaging.BatchResponse;
import com.google.firebase.messaging.FirebaseMessagingException;
import org.quartz.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Component
public class SimpleTimeBasedReminderJob implements Job {

    @Autowired
    private MedicationReminderRepository medicationReminderRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private DeviceTokenRepository deviceTokenRepository;

    @Autowired
    private FcmService fcmService;
    
    @Autowired
    private NotificationService notificationService;

    @Autowired
    private MedicationLogRepository medicationLogRepository;

    @Override
    @Transactional
    public void execute(JobExecutionContext context) throws JobExecutionException {
        Long userId = context.getJobDetail().getJobDataMap().getLong("userId");
        String timeSlot = context.getJobDetail().getJobDataMap().getString("timeSlot");
        
        System.out.println("=== Executing SimpleTimeBasedReminderJob ===");
        System.out.println("User ID: " + userId + " | ⏰ Time: " + timeSlot);
        
        try {
            System.out.println("Step 1: Searching medications for userId=" + userId + ", timeSlot=" + timeSlot);
            
            // FIXED: Tìm tất cả thuốc active của user tại thời điểm này (không lazy loading User)
            List<MedicationReminder> medications = medicationReminderRepository
                    .findByUserIdAndReminderTimeSimple(userId, timeSlot);

            if (medications.isEmpty()) {
                System.out.println("No medications found for user " + userId + " at " + timeSlot);
                return;
            }

            System.out.println("Found " + medications.size() + " medication(s):");
            medications.forEach(med -> {
                try {
                    System.out.println("  - " + med.getName() + " (ID: " + med.getId() + ")");
                } catch (Exception e) {
                    System.err.println("  - ERROR accessing medication name: " + e.getMessage());
                }
            });

            System.out.println("Step 2: Fetching device tokens for userId=" + userId);
            
            // FIXED: Lấy FCM tokens trực tiếp từ repository thay vì lazy loading
            List<DeviceToken> deviceTokenEntities;
            try {
                deviceTokenEntities = deviceTokenRepository.findByUserIdAndIsActiveTrue(userId);
                System.out.println("Successfully fetched device tokens from repository");
            } catch (Exception e) {
                System.err.println("Error fetching device tokens: " + e.getMessage());
                e.printStackTrace();
                return;
            }
            
            System.out.println("📱 Raw device tokens found: " + deviceTokenEntities.size());
            deviceTokenEntities.forEach(dt -> {
                try {
                    System.out.println("  - Token: " + 
                        (dt.getFcmToken() != null ? dt.getFcmToken().substring(0, Math.min(20, dt.getFcmToken().length())) + "..." : "null"));
                } catch (Exception e) {
                    System.err.println("  - ERROR accessing token: " + e.getMessage());
                }
            });
            
            List<String> deviceTokens;
            try {
                deviceTokens = deviceTokenEntities.stream()
                        .map(DeviceToken::getFcmToken)
                        .filter(token -> token != null && !token.trim().isEmpty())
                        .toList();
                System.out.println("Successfully processed device tokens");
            } catch (Exception e) {
                System.err.println("Error processing device tokens: " + e.getMessage());
                e.printStackTrace();
                return;
            }

            System.out.println("Valid device tokens: " + deviceTokens.size());

            if (deviceTokens.isEmpty()) {
                System.out.println("⚠No device tokens found - skipping FCM");
                return;
            }

            System.out.println("Step 3: Sending grouped notification");
            // Gửi thông báo gộp và lưu log
            sendGroupedNotification(medications, deviceTokens, userId, timeSlot);
            System.out.println("Notification process completed");

        } catch (Exception e) {
            System.err.println("Error in SimpleTimeBasedReminderJob: " + e.getMessage());
            System.err.println("Error class: " + e.getClass().getSimpleName());
            e.printStackTrace();
            
            // Check if this is LazyInitializationException
            if (e.getMessage().contains("LazyInitializationException") || e.getMessage().contains("no session")) {
                System.err.println("LAZY INITIALIZATION DETECTED - Check for lazy loading issues!");
            }
        }
    }

    private void sendGroupedNotification(List<MedicationReminder> medications, List<String> deviceTokens, Long userId, String timeSlot) {
        try {
            String title = "Đến giờ uống thuốc";
            String body = buildNotificationBody(medications);

            System.out.println("Sending notification:");
            System.out.println("Title: " + title);
            System.out.println("Body: " + body.substring(0, Math.min(100, body.length())) + "...");

            BatchResponse response = fcmService.sendNotification(deviceTokens, title, body);
            
            System.out.println("FCM Result - Success: " + response.getSuccessCount() +
                             ", Failed: " + response.getFailureCount());
                             
            // Lưu MedicationLog và Notification
            try {
                System.out.println("Saving medication log...");
                
                // Parse timeSlot để tính reminderTime (giờ dự kiến)
                LocalDateTime reminderTime = parseTimeSlot(timeSlot);
                
                // Tạo User object (để cast thành ElderUser)
                User user = userRepository.findById(userId)
                        .orElseThrow(() -> new RuntimeException("User not found"));
                
                if (!(user instanceof ElderUser)) {
                    System.err.println("User is not ElderUser, skipping medication log");
                    return;
                }
                
                ElderUser elderUser = (ElderUser) user;
                
                // Thu thập thông tin từ danh sách medications
                List<String> medicationIdList = medications.stream()
                        .map(med -> String.valueOf(med.getId()))
                        .toList();
                List<String> medicationNameList = medications.stream()
                        .map(MedicationReminder::getName)
                        .toList();
                
                String medicationIds = String.join(",", medicationIdList);
                String medicationNames = String.join(", ", medicationNameList);
                
                // Tạo MedicationLog (tracking adherence)
                MedicationLog medicationLog = new MedicationLog();
                medicationLog.setElderUser(elderUser);
                medicationLog.setReminderTime(reminderTime); // Giờ dự kiến: 09:00
                medicationLog.setActualTakenTime(null); // Chưa confirm
                medicationLog.setMinutesLate(null); // Chưa tính được
                medicationLog.setStatus(EMedicationLogStatus.PENDING);
                medicationLog.setMedicationCount(medications.size());
                medicationLog.setMedicationIds(medicationIds);
                medicationLog.setMedicationNames(medicationNames);
                medicationLog.setNote(null);
                medicationLog.setFcmSent(response.getSuccessCount() > 0);
                medicationLog.setFcmSentTime(LocalDateTime.now());
                
                medicationLog = medicationLogRepository.save(medicationLog);
                System.out.println(" Saved MedicationLog (ID: " + medicationLog.getId() +
                                 ") for " + medications.size() + " medications: " + medicationNames);
                System.out.println("Reminder time (scheduled): " + reminderTime);
                
                //Tạo Notification (general notification system)
                Notifications notification = new Notifications();
                notification.setNotificationType(ENotificationType.MEDICATION_REMINDER);
                notification.setUser(elderUser);
                notification.setRelatedMedicationLog(medicationLog); // Link đến log
                notification.setTitle(title);
                notification.setBody(body);
                notification.setStatus(response.getSuccessCount() > 0 ? 
                    ENotificationStatus.SENT : ENotificationStatus.FAILED);
                notification.setIsRead(false);
                
                notificationService.save(notification);
                System.out.println(" Saved Notification (ID: " + notification.getId() + ") linked to MedicationLog");
                System.out.println("Waiting for user confirmation...");
                
            } catch (Exception e) {
                System.err.println("Failed to save logs: " + e.getMessage());
                e.printStackTrace();
            }

        } catch (FirebaseMessagingException e) {
            System.err.println("FCM Error: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("Notification Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private String buildNotificationBody(List<MedicationReminder> medications) {
        System.out.println("Building notification body for " + medications.size() + " medication(s)");
        
        try {
            if (medications.size() == 1) {
                MedicationReminder med = medications.get(0);
                System.out.println("Single medication: accessing fields for " + med.getClass().getSimpleName());
                
                String name = med.getName();
                String time = med.getReminderTime(); 
                String desc = med.getDescription();
                
                System.out.println("Successfully accessed fields: name=" + name + ", time=" + time);
                
                // Format gọn cho 1 thuốc (1 dòng)
                StringBuilder result = new StringBuilder();
                result.append(String.format("🕐 %s - %s", time, name));
                
                if (desc != null && !desc.trim().isEmpty()) {
                    result.append(String.format("\n%s", desc));
                }
                
                return result.toString();
            }

            System.out.println("Multiple medications: building grouped message");
            StringBuilder body = new StringBuilder();
            
            // FORMAT COMPACT: Gộp thời gian và số lượng vào 1 dòng
            if (!medications.isEmpty()) {
                String time = medications.get(0).getReminderTime();
                body.append(String.format("🕐 %s • %d loại thuốc:\n", time, medications.size()));
            }
            
            // Mỗi thuốc chỉ 1 dòng (gộn tên + ghi chú)
            for (int i = 0; i < medications.size(); i++) {
                MedicationReminder med = medications.get(i);
                System.out.println("Processing medication " + (i+1) + ": " + med.getClass().getSimpleName());
                
                try {
                    String name = med.getName();
                    String desc = med.getDescription();
                    
                    System.out.println("Medication " + (i+1) + " fields accessed successfully");
                    
                    // GỌN: 1 dòng cho 1 thuốc
                    if (desc != null && !desc.trim().isEmpty()) {
                        body.append(String.format("%d. %s - %s\n", (i + 1), name, desc));
                    } else {
                        body.append(String.format("%d. %s\n", (i + 1), name));
                    }
                    
                } catch (Exception e) {
                    System.err.println("Error accessing medication " + (i+1) + " fields: " + e.getMessage());
                    throw e;
                }
            }
            
            String result = body.toString().trim();
            System.out.println("Notification body built successfully (" + result.length() + " chars)");
            return result;
            
        } catch (Exception e) {
            System.err.println("Error in buildNotificationBody: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    /**
     * Parse timeSlot string thành LocalDateTime của hôm nay
     * VD: "09:00" → 2024-12-04 09:00:00
     */
    private LocalDateTime parseTimeSlot(String timeSlot) {
        String[] parts = timeSlot.split(":");
        int hour = Integer.parseInt(parts[0]);
        int minute = Integer.parseInt(parts[1]);
        
        return LocalDateTime.now()
                .withHour(hour)
                .withMinute(minute)
                .withSecond(0)
                .withNano(0);
    }
}