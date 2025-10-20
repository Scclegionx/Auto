package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.dto.response.PrescriptionResponse;
import com.example.Auto_BE.service.CronPrescriptionService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 🎯 OPTIMIZED Prescription Controller với TIME-BASED Scheduling
 * 
 * ✅ Service tự động xử lý TIME-BASED scheduling
 * ✅ Controller chỉ focus vào REST API logic
 * ✅ Loại bỏ manual scheduling endpoints
 */
@RestController
@RequestMapping("/api/cron-prescriptions")
public class CronPrescriptionController {

    private final CronPrescriptionService cronPrescriptionService;

    public CronPrescriptionController(CronPrescriptionService cronPrescriptionService) {
        this.cronPrescriptionService = cronPrescriptionService;
    }

    /**
     * Tạo đơn thuốc mới
     * Service tự động xử lý TIME-BASED scheduling
     */
    @PostMapping("/create")
    public ResponseEntity<BaseResponse<PrescriptionResponse>> createPrescription(
            @RequestBody @Valid PrescriptionCreateRequest prescriptionCreateRequest,
            Authentication authentication) {

        return ResponseEntity.ok(cronPrescriptionService.create(prescriptionCreateRequest, authentication));
    }

    /**
     * Lấy thông tin đơn thuốc theo ID
     */
    @GetMapping("/{prescriptionId}")
    public ResponseEntity<BaseResponse<PrescriptionResponse>> getPrescription(
            @PathVariable Long prescriptionId,
            Authentication authentication) {

        return ResponseEntity.ok(cronPrescriptionService.getById(prescriptionId, authentication));
    }

    /**
     * Lấy tất cả đơn thuốc của user
     */
    @GetMapping
    public ResponseEntity<BaseResponse<List<PrescriptionResponse>>> getAllPrescriptions(
            Authentication authentication) {

        return ResponseEntity.ok(cronPrescriptionService.getAllByUser(authentication));
    }

    /**
     * Cập nhật đơn thuốc
     * Service tự động xử lý TIME-BASED rescheduling
     */
    @PutMapping("/{prescriptionId}")
    public ResponseEntity<BaseResponse<PrescriptionResponse>> updatePrescription(
            @PathVariable Long prescriptionId,
            @RequestBody @Valid PrescriptionCreateRequest prescriptionUpdateRequest,
            Authentication authentication) {

        return ResponseEntity.ok(cronPrescriptionService.update(prescriptionId, prescriptionUpdateRequest, authentication));
    }

    /**
     * Xóa đơn thuốc
     * Service tự động xử lý TIME-BASED rescheduling cho medications còn lại
     */
    @DeleteMapping("/{prescriptionId}")
    public ResponseEntity<BaseResponse<String>> deletePrescription(
            @PathVariable Long prescriptionId,
            Authentication authentication) {

        return ResponseEntity.ok(cronPrescriptionService.delete(prescriptionId, authentication));
    }
}