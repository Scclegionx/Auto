package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.MedicationLog;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.enums.EMedicationLogStatus;
import com.example.Auto_BE.repository.MedicationLogRepository;
import com.example.Auto_BE.repository.UserRepository;
import com.example.Auto_BE.service.SupervisorAlertService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/medication-logs")
@RequiredArgsConstructor
@Slf4j
public class MedicationLogController {

    private final MedicationLogRepository medicationLogRepository;
    private final UserRepository userRepository;
    private final SupervisorAlertService supervisorAlertService;

    @PostMapping("/{logId}/confirm")
    public ResponseEntity<BaseResponse<MedicationLog>> confirmMedicationTaken(
            @PathVariable Long logId,
            @RequestParam Long userId,
            @RequestBody(required = false) Map<String, String> body
    ) {
        try {
            User user = userRepository.findById(userId)
                    .orElseThrow(() -> new RuntimeException("User not found"));
            
            if (!(user instanceof ElderUser)) {
                return ResponseEntity.badRequest().body(
                        BaseResponse.<MedicationLog>builder()
                                .status("error")
                                .message("Only Elder can confirm medication")
                                .build()
                );
            }
            
            ElderUser elderUser = (ElderUser) user;
            
            MedicationLog medicationLog = medicationLogRepository.findById(logId)
                    .orElseThrow(() -> new RuntimeException("MedicationLog not found"));
            
            if (!medicationLog.getElderUser().getId().equals(elderUser.getId())) {
                return ResponseEntity.badRequest().body(
                        BaseResponse.<MedicationLog>builder()
                                .status("error")
                                .message("MedicationLog does not belong to user")
                                .build()
                );
            }
            
            LocalDateTime actualTakenTime = LocalDateTime.now();
            medicationLog.setActualTakenTime(actualTakenTime);
            medicationLog.setStatus(EMedicationLogStatus.TAKEN);
            
            long minutesDiff = java.time.Duration.between(
                    medicationLog.getReminderTime(), 
                    actualTakenTime
            ).toMinutes();
            medicationLog.setMinutesLate((int) minutesDiff);
            
            if (body != null && body.containsKey("note")) {
                medicationLog.setNote(body.get("note"));
            }
            
            medicationLogRepository.save(medicationLog);
            
            String message = minutesDiff > 15 
                    ? "Đã ghi nhận (trễ " + minutesDiff + " phút)"
                    : "Đã uống đúng giờ!";
            
            return ResponseEntity.ok(
                    BaseResponse.<MedicationLog>builder()
                            .status("success")
                            .message(message)
                            .data(medicationLog)
                            .build()
            );
            
        } catch (Exception e) {
            log.error("Error confirming medication taken", e);
            return ResponseEntity.internalServerError().body(
                    BaseResponse.<MedicationLog>builder()
                            .status("error")
                            .message("Failed to confirm: " + e.getMessage())
                            .build()
            );
        }
    }
    
    @GetMapping("/my-history")
    public ResponseEntity<BaseResponse<Map<String, Object>>> getMyMedicationHistory(
            @RequestParam Long userId,
            @RequestParam(defaultValue = "7") int days
    ) {
        try {
            User user = userRepository.findById(userId)
                    .orElseThrow(() -> new RuntimeException("User not found"));
            
            if (!(user instanceof ElderUser)) {
                return ResponseEntity.badRequest().body(
                        BaseResponse.<Map<String, Object>>builder()
                                .status("error")
                                .message("Only Elder can view medication history")
                                .build()
                );
            }
            
            ElderUser elderUser = (ElderUser) user;
            LocalDateTime startDate = LocalDateTime.now().minusDays(days);
            LocalDateTime endDate = LocalDateTime.now();
            
            List<MedicationLog> logs = medicationLogRepository.findByElderUserAndReminderTimeBetween(
                    elderUser, startDate, endDate
            );
            
            long totalCount = logs.size();
            long takenCount = logs.stream()
                    .filter(l -> l.getStatus() == EMedicationLogStatus.TAKEN)
                    .count();
            long onTimeCount = logs.stream()
                    .filter(l -> l.getStatus() == EMedicationLogStatus.TAKEN && 
                            Math.abs(l.getMinutesLate() != null ? l.getMinutesLate() : 0) <= 15)
                    .count();
            long missedCount = logs.stream()
                    .filter(l -> l.getStatus() == EMedicationLogStatus.MISSED)
                    .count();
            
            Map<String, Object> statistics = new HashMap<>();
            statistics.put("total", totalCount);
            statistics.put("taken", takenCount);
            statistics.put("missed", missedCount);
            statistics.put("onTime", onTimeCount);
            statistics.put("adherenceRate", totalCount > 0 ? (takenCount * 100.0 / totalCount) : 0);
            statistics.put("onTimeRate", takenCount > 0 ? (onTimeCount * 100.0 / takenCount) : 0);
            
            Map<String, Object> result = new HashMap<>();
            result.put("logs", logs);
            result.put("statistics", statistics);
            
            return ResponseEntity.ok(
                    BaseResponse.<Map<String, Object>>builder()
                            .status("success")
                            .message("History retrieved")
                            .data(result)
                            .build()
            );
            
        } catch (Exception e) {
            log.error("Error getting medication history", e);
            return ResponseEntity.internalServerError().body(
                    BaseResponse.<Map<String, Object>>builder()
                            .status("error")
                            .message("Failed to get history: " + e.getMessage())
                            .build()
            );
        }
    }
    
    @GetMapping("/elder/{elderId}/history")
    public ResponseEntity<BaseResponse<Map<String, Object>>> getElderMedicationHistory(
            @PathVariable Long elderId,
            @RequestParam(defaultValue = "7") int days
    ) {
        try {
            LocalDateTime startDate = LocalDateTime.now().minusDays(days);
            LocalDateTime endDate = LocalDateTime.now();
            
            List<MedicationLog> logs = medicationLogRepository.findByElderUserIdAndReminderTimeBetween(
                    elderId, startDate, endDate
            );
            
            long totalCount = logs.size();
            long takenCount = logs.stream()
                    .filter(l -> l.getStatus() == EMedicationLogStatus.TAKEN)
                    .count();
            long onTimeCount = logs.stream()
                    .filter(l -> l.getStatus() == EMedicationLogStatus.TAKEN && 
                            Math.abs(l.getMinutesLate() != null ? l.getMinutesLate() : 0) <= 15)
                    .count();
            long missedCount = logs.stream()
                    .filter(l -> l.getStatus() == EMedicationLogStatus.MISSED)
                    .count();
            
            Map<String, Object> statistics = new HashMap<>();
            statistics.put("total", totalCount);
            statistics.put("taken", takenCount);
            statistics.put("missed", missedCount);
            statistics.put("onTime", onTimeCount);
            statistics.put("adherenceRate", totalCount > 0 ? (takenCount * 100.0 / totalCount) : 0);
            statistics.put("onTimeRate", takenCount > 0 ? (onTimeCount * 100.0 / takenCount) : 0);
            
            Map<String, Object> result = new HashMap<>();
            result.put("logs", logs);
            result.put("statistics", statistics);
            
            return ResponseEntity.ok(
                    BaseResponse.<Map<String, Object>>builder()
                            .status("success")
                            .message("Elder history retrieved")
                            .data(result)
                            .build()
            );
            
        } catch (Exception e) {
            log.error("Error getting elder medication history", e);
            return ResponseEntity.internalServerError().body(
                    BaseResponse.<Map<String, Object>>builder()
                            .status("error")
                            .message("Failed to get history: " + e.getMessage())
                            .build()
            );
        }
    }
    
    @GetMapping("/{logId}")
    public ResponseEntity<BaseResponse<MedicationLog>> getMedicationLogDetail(
            @PathVariable Long logId
    ) {
        try {
            MedicationLog medicationLog = medicationLogRepository.findById(logId)
                    .orElseThrow(() -> new RuntimeException("MedicationLog not found"));
            
            return ResponseEntity.ok(
                    BaseResponse.<MedicationLog>builder()
                            .status("success")
                            .message("Log retrieved")
                            .data(medicationLog)
                            .build()
            );
            
        } catch (Exception e) {
            log.error("Error getting medication log detail", e);
            return ResponseEntity.internalServerError().body(
                    BaseResponse.<MedicationLog>builder()
                            .status("error")
                            .message("Failed to get log: " + e.getMessage())
                            .build()
            );
        }
    }
}
