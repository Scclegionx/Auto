package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.CreateMedicationRequest;
import com.example.Auto_BE.dto.request.UpdateMedicationRequest;
import com.example.Auto_BE.dto.response.MedicationResponse;
import com.example.Auto_BE.service.NotificationService;
import com.example.Auto_BE.service.SimpleTimeBasedScheduler;
import com.example.Auto_BE.service.MedicationService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;
import java.util.List;

@RestController
@RequestMapping("/api/medication")
public class MedicationController {

    private final NotificationService notificationService;
    private final SimpleTimeBasedScheduler simpleScheduler;
    private final MedicationService medicationService;

    public MedicationController(NotificationService notificationService, 
                              SimpleTimeBasedScheduler simpleScheduler,
                              MedicationService medicationService) {
        this.notificationService = notificationService;
        this.simpleScheduler = simpleScheduler;
        this.medicationService = medicationService;
    }

    // ===================== MEDICATION CRUD with TIME-BASED SCHEDULING =====================

    /**
     * Tạo medication mới
     * Response trả LIST - mỗi time = 1 medication
     */
    @PostMapping
    public ResponseEntity<BaseResponse<List<MedicationResponse>>> createMedication(
            @RequestBody CreateMedicationRequest request,
            Authentication authentication) {
        
        System.out.println("API: Create medication - " + request.getName());
        
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(request, authentication);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Lấy medication theo ID
     */
    @GetMapping("/{medicationId}")
    public ResponseEntity<BaseResponse<MedicationResponse>> getMedication(
            @PathVariable Long medicationId,
            Authentication authentication) {
        
        System.out.println("📋 API: Get medication - " + medicationId);
        
        BaseResponse<MedicationResponse> response = medicationService.getMedicationById(medicationId, authentication);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Lấy tất cả medications của user
     */
    @GetMapping("/user/{userId}")
    public ResponseEntity<BaseResponse<List<MedicationResponse>>> getUserMedications(
            @PathVariable Long userId) {
        
        System.out.println("📋 API: Get all medications for user - " + userId);
        
        BaseResponse<List<MedicationResponse>> response = medicationService.getAllMedicationsByUser(userId);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Lấy chỉ standalone medications (thuốc ngoài đơn) đã được group theo tên
     */
    @GetMapping("/standalone/user/{userId}")
    public ResponseEntity<BaseResponse<List<MedicationResponse>>> getStandaloneMedications(
            @PathVariable Long userId) {
        
        System.out.println("API: Get standalone medications for user - " + userId);
        
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(userId);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Update medication
     */
    @PutMapping("/{medicationId}")
    public ResponseEntity<BaseResponse<MedicationResponse>> updateMedication(
            @PathVariable Long medicationId,
            @RequestBody UpdateMedicationRequest request,
            Authentication authentication) {
        
        System.out.println("API: Update medication - " + medicationId);
        
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(medicationId, request, authentication);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Xóa medication
     */
    @DeleteMapping("/{medicationId}")
    public ResponseEntity<BaseResponse<String>> deleteMedication(
            @PathVariable Long medicationId,
            Authentication authentication) {
        
        System.out.println("🗑️ API: Delete medication - " + medicationId);
        
        BaseResponse<String> response = medicationService.deleteMedication(medicationId, authentication);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Toggle trạng thái active/inactive của medication
     */
    @PutMapping("/{medicationId}/toggle")
    public ResponseEntity<BaseResponse<MedicationResponse>> toggleMedicationStatus(
            @PathVariable Long medicationId,
            Authentication authentication) {
        
        System.out.println("🔄 API: Toggle medication status - " + medicationId);
        
        BaseResponse<MedicationResponse> response = medicationService.toggleMedicationStatus(medicationId, authentication);
        
        if ("success".equals(response.status)) {
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.badRequest().body(response);
        }
    }

    // ===================== SIMPLE TIME-BASED SCHEDULING =====================

    /**
     * Lên lịch reminders cho user (sau khi CRUD medication)
     */
    @PostMapping("/schedule/{userId}")
    public ResponseEntity<BaseResponse<String>> scheduleUserReminders(@PathVariable Long userId) {
        System.out.println("🔄 API: Schedule reminders for user: " + userId);
        
        try {
            simpleScheduler.scheduleUserReminders(userId);
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("success")
                    .message("Đã lên lịch thông báo cho user")
                    .data("User " + userId + " reminders scheduled successfully")
                    .build();
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("💥 API Error scheduling: " + e.getMessage());
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("error")
                    .message("Lỗi khi lên lịch thông báo")
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Hủy tất cả reminders của user
     */
    @DeleteMapping("/schedule/{userId}")
    public ResponseEntity<BaseResponse<String>> cancelUserReminders(@PathVariable Long userId) {
        System.out.println("🗑️ API: Cancel reminders for user: " + userId);
        
        try {
            simpleScheduler.cancelUserReminders(userId);
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("success")
                    .message("Đã hủy tất cả thông báo của user")
                    .data("User " + userId + " reminders canceled successfully")
                    .build();
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("💥 API Error canceling: " + e.getMessage());
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("error")
                    .message("Lỗi khi hủy thông báo")
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Xem jobs đang active của user
     */
    @GetMapping("/schedule/{userId}")
    public ResponseEntity<BaseResponse<Object>> getUserActiveJobs(@PathVariable Long userId) {
        System.out.println("📋 API: Get active jobs for user: " + userId);
        
        try {
            var activeJobs = simpleScheduler.getUserActiveJobs(userId);
            
            Map<String, Object> responseData = new HashMap<>();
            responseData.put("userId", userId);
            responseData.put("jobCount", activeJobs.size());
            responseData.put("jobs", activeJobs);
            
            BaseResponse<Object> response = BaseResponse.<Object>builder()
                    .status("success")
                    .message("Danh sách jobs active của user")
                    .data(responseData)
                    .build();
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("💥 API Error getting jobs: " + e.getMessage());
            
            BaseResponse<Object> response = BaseResponse.<Object>builder()
                    .status("error")
                    .message("Lỗi khi lấy thông tin jobs")
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Thống kê toàn bộ system
     */
    @GetMapping("/schedule/stats")
    public ResponseEntity<BaseResponse<String>> getSchedulerStats() {
        System.out.println("📊 API: Get scheduler stats");
        
        try {
            simpleScheduler.printJobStats();
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("success")
                    .message("Thống kê scheduler (xem console)")
                    .data("Stats printed to console")
                    .build();
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("💥 API Error getting stats: " + e.getMessage());
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("error")
                    .message("Lỗi khi lấy thống kê")
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }
}