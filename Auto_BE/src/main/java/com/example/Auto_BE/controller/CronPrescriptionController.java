package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.dto.response.PrescriptionResponse;
import com.example.Auto_BE.service.CloudinaryService;
import com.example.Auto_BE.service.CronPrescriptionService;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

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
@Slf4j
public class CronPrescriptionController {

    private final CronPrescriptionService cronPrescriptionService;
    private final CloudinaryService cloudinaryService;
    private final ObjectMapper objectMapper;

    public CronPrescriptionController(CronPrescriptionService cronPrescriptionService,
                                     CloudinaryService cloudinaryService,
                                     ObjectMapper objectMapper) {
        this.cronPrescriptionService = cronPrescriptionService;
        this.cloudinaryService = cloudinaryService;
        this.objectMapper = objectMapper;
    }

    /**
     * Tạo đơn thuốc mới (không có ảnh)
     * Service tự động xử lý TIME-BASED scheduling
     */
    @PostMapping("/create")
    public ResponseEntity<BaseResponse<PrescriptionResponse>> createPrescription(
            @RequestBody @Valid PrescriptionCreateRequest prescriptionCreateRequest,
            Authentication authentication) {
        System.out.println("POST /create called");
        return ResponseEntity.ok(cronPrescriptionService.create(prescriptionCreateRequest, authentication));
    }

    /**
     * Tạo đơn thuốc mới kèm upload ảnh lên Cloudinary
     * ✅ Flow: Validate → Upload ảnh → Lưu DB → Rollback nếu lỗi
     */
    @PostMapping(value = "/create-with-image", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<BaseResponse<PrescriptionResponse>> createPrescriptionWithImage(
            @RequestParam("data") String prescriptionDataJson,
            @RequestParam("image") MultipartFile image,
            Authentication authentication) {
        System.out.println("POST /create-with-image called");
        String uploadedImageUrl = null;
        
        try {
            // Step 1: Parse JSON
            PrescriptionCreateRequest prescriptionData = objectMapper.readValue(
                    prescriptionDataJson, PrescriptionCreateRequest.class);

            // Step 2: Validate cơ bản
            if (prescriptionData.getName() == null || prescriptionData.getName().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(BaseResponse.<PrescriptionResponse>builder()
                                .status("error")
                                .message("Tên đơn thuốc không được để trống")
                                .build());
            }

            if (prescriptionData.getMedicationReminders() == null || prescriptionData.getMedicationReminders().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(BaseResponse.<PrescriptionResponse>builder()
                                .status("error")
                                .message("Phải có ít nhất 1 loại thuốc")
                                .build());
            }

            if (image == null || image.isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(BaseResponse.<PrescriptionResponse>builder()
                                .status("error")
                                .message("File ảnh không được để trống")
                                .build());
            }

            // Step 3: Upload image to Cloudinary
            log.info("Uploading image to Cloudinary: {} (size: {} bytes)", 
                    image.getOriginalFilename(), image.getSize());
            uploadedImageUrl = cloudinaryService.uploadImage(image);
            prescriptionData.setImageUrl(uploadedImageUrl);

            // Step 4: Create prescription in DB
            BaseResponse<PrescriptionResponse> response = cronPrescriptionService.create(prescriptionData, authentication);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error creating prescription with image", e);
            
            // ✅ ROLLBACK: Xóa ảnh nếu lỗi
            if (uploadedImageUrl != null) {
                try {
                    log.warn("Rollback: Deleting image from Cloudinary");
                    cloudinaryService.deleteImage(uploadedImageUrl);
                } catch (Exception ex) {
                    log.error("Failed to delete image", ex);
                }
            }
            
            return ResponseEntity.internalServerError()
                    .body(BaseResponse.<PrescriptionResponse>builder()
                            .status("error")
                            .message("Lỗi khi tạo đơn thuốc: " + e.getMessage())
                            .build());
        }
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

    /**
     * Toggle trạng thái đơn thuốc (active/inactive)
     * Tự động cập nhật trạng thái tất cả medications và reschedule jobs
     */
    @PatchMapping("/{prescriptionId}/toggle-status")
    public ResponseEntity<BaseResponse<PrescriptionResponse>> togglePrescriptionStatus(
            @PathVariable Long prescriptionId,
            Authentication authentication) {

        return ResponseEntity.ok(cronPrescriptionService.toggleStatus(prescriptionId, authentication));
    }
}