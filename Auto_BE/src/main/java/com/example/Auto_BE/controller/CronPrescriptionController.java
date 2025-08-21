package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.dto.response.PrescriptionResponse;
import com.example.Auto_BE.service.CronPrescriptionService;
import com.example.Auto_BE.service.CronSchedulerService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;

@RestController
@RequestMapping("/api/cron-prescriptions")
public class CronPrescriptionController {
    
    private final CronPrescriptionService cronPrescriptionService;
    private final CronSchedulerService cronSchedulerService;

    public CronPrescriptionController(CronPrescriptionService cronPrescriptionService,
                                      CronSchedulerService cronSchedulerService) {
        this.cronPrescriptionService = cronPrescriptionService;
        this.cronSchedulerService = cronSchedulerService;
    }

    @PostMapping("/create")
    public ResponseEntity<BaseResponse<PrescriptionResponse>> createPrescription(
            @RequestBody @Valid PrescriptionCreateRequest prescriptionCreateRequest,
            Authentication authentication) {
        
        BaseResponse<PrescriptionResponse> response = cronPrescriptionService.create(prescriptionCreateRequest, authentication);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/{prescriptionId}")
    public ResponseEntity<BaseResponse<PrescriptionResponse>> getPrescription(
            @PathVariable Long prescriptionId,
            Authentication authentication) {
        
        BaseResponse<PrescriptionResponse> response = cronPrescriptionService.getById(prescriptionId, authentication);
        return ResponseEntity.ok(response);
    }

    @GetMapping
    public ResponseEntity<BaseResponse<List<PrescriptionResponse>>> getAllPrescriptions(
            Authentication authentication) {
        
        BaseResponse<List<PrescriptionResponse>> response = cronPrescriptionService.getAllByUser(authentication);
        return ResponseEntity.ok(response);
    }

    @PutMapping("/{prescriptionId}")
    public ResponseEntity<BaseResponse<PrescriptionResponse>> updatePrescription(
            @PathVariable Long prescriptionId,
            @RequestBody @Valid PrescriptionCreateRequest prescriptionUpdateRequest,
            Authentication authentication) {
        
        BaseResponse<PrescriptionResponse> response = cronPrescriptionService.update(prescriptionId, prescriptionUpdateRequest, authentication);
        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/{prescriptionId}")
    public ResponseEntity<BaseResponse<String>> deletePrescription(
            @PathVariable Long prescriptionId,
            Authentication authentication) {
        
        BaseResponse<String> response = cronPrescriptionService.delete(prescriptionId, authentication);
        return ResponseEntity.ok(response);
    }

    // ===== DEBUGGING ENDPOINTS =====

    @PostMapping("/schedule-all")
    public ResponseEntity<BaseResponse<String>> scheduleAllActiveReminders() {
        try {
            cronSchedulerService.scheduleAllActiveReminders();
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Scheduled all active reminders successfully")
                    .data("Check logs for details")
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Error scheduling reminders: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    @PostMapping("/schedule/{medicationReminderId}")
    public ResponseEntity<BaseResponse<String>> scheduleSpecificReminder(
            @PathVariable Long medicationReminderId) {
        try {
            cronSchedulerService.scheduleWithCron(medicationReminderId);
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Scheduled reminder successfully")
                    .data("Medication Reminder ID: " + medicationReminderId)
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Error scheduling reminder: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    @DeleteMapping("/schedule/{medicationReminderId}")
    public ResponseEntity<BaseResponse<String>> cancelSpecificReminder(
            @PathVariable Long medicationReminderId) {
        try {
            cronSchedulerService.cancelCronSchedule(medicationReminderId);
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Cancelled reminder schedule successfully")
                    .data("Medication Reminder ID: " + medicationReminderId)
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Error cancelling reminder: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/cron-jobs")
    public ResponseEntity<BaseResponse<String>> listAllCronJobs() {
        try {
            cronSchedulerService.listAllCronJobs();
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Listed all cron jobs")
                    .data("Check console logs for details")
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Error listing cron jobs: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

}
