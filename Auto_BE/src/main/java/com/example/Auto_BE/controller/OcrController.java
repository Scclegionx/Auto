package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.example.Auto_BE.service.GeminiService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;

/**
 * Controller xử lý OCR và trích xuất dữ liệu từ ảnh đơn thuốc sử dụng Gemini AI
 */
@RestController
@RequestMapping("/api/ocr")
@RequiredArgsConstructor
@Slf4j
public class OcrController {

    private final GeminiService geminiService;

    /**
     * API nhận ảnh đơn thuốc và trích xuất dữ liệu thành PrescriptionCreateRequest
     * 
     * @param image File ảnh đơn thuốc (JPG, PNG, etc.)
     * @return PrescriptionCreateRequest đã được extract từ ảnh
     */
    @PostMapping(value = "/extract-prescription", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<BaseResponse<PrescriptionCreateRequest>> extractPrescription(
            @RequestParam("image") MultipartFile image) {
        
        try {
            log.info("Received image for OCR: {} (size: {} bytes)", 
                    image.getOriginalFilename(), image.getSize());

            // Validate image
            if (image.isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(BaseResponse.<PrescriptionCreateRequest>builder()
                                .status("error")
                                .message("File ảnh không được để trống")
                                .build());
            }

            // Validate image type
            String contentType = image.getContentType();
            if (contentType == null || !contentType.startsWith("image/")) {
                return ResponseEntity.badRequest()
                        .body(BaseResponse.<PrescriptionCreateRequest>builder()
                                .status("error")
                                .message("File không phải là ảnh hợp lệ")
                                .build());
            }

            // Call Gemini service
            PrescriptionCreateRequest result = geminiService.extractPrescriptionFromImage(image);

            return ResponseEntity.ok(BaseResponse.<PrescriptionCreateRequest>builder()
                    .status(SUCCESS)
                    .message("Trích xuất dữ liệu đơn thuốc thành công")
                    .data(result)
                    .build());

        } catch (Exception e) {
            log.error("Error extracting prescription from image", e);
            return ResponseEntity.internalServerError()
                    .body(BaseResponse.<PrescriptionCreateRequest>builder()
                            .status("error")
                            .message("Lỗi khi xử lý ảnh: " + e.getMessage())
                            .build());
        }
    }
}
