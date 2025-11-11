package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.request.EmergencyContactCreateRequest;
import com.example.Auto_BE.dto.request.EmergencyContactUpdateRequest;
import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.response.EmergencyContactResponse;
import com.example.Auto_BE.service.EmergencyContactService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/emergency-contacts")
@RequiredArgsConstructor
public class EmergencyContactController {
    
    private final EmergencyContactService emergencyContactService;
    
    /**
     * Tạo liên hệ khẩn cấp mới
     * POST /api/emergency-contacts
     */
    @PostMapping
    public ResponseEntity<BaseResponse<EmergencyContactResponse>> create(
            @Valid @RequestBody EmergencyContactCreateRequest request,
            Authentication authentication) {
        BaseResponse<EmergencyContactResponse> response = emergencyContactService.create(request, authentication);
        return ResponseEntity.ok(response);
    }
    
    /**
     * Lấy tất cả liên hệ khẩn cấp của user hiện tại
     * GET /api/emergency-contacts
     */
    @GetMapping
    public ResponseEntity<BaseResponse<List<EmergencyContactResponse>>> getAllByUser(
            Authentication authentication) {
        BaseResponse<List<EmergencyContactResponse>> response = emergencyContactService.getAllByUser(authentication);
        return ResponseEntity.ok(response);
    }
    
    /**
     * Lấy chi tiết liên hệ khẩn cấp theo ID
     * GET /api/emergency-contacts/{id}
     */
    @GetMapping("/{id}")
    public ResponseEntity<BaseResponse<EmergencyContactResponse>> getById(
            @PathVariable Long id,
            Authentication authentication) {
        BaseResponse<EmergencyContactResponse> response = emergencyContactService.getById(id, authentication);
        return ResponseEntity.ok(response);
    }
    
    /**
     * Cập nhật liên hệ khẩn cấp
     * PUT /api/emergency-contacts/{id}
     */
    @PutMapping("/{id}")
    public ResponseEntity<BaseResponse<EmergencyContactResponse>> update(
            @PathVariable Long id,
            @Valid @RequestBody EmergencyContactUpdateRequest request,
            Authentication authentication) {
        BaseResponse<EmergencyContactResponse> response = emergencyContactService.update(id, request, authentication);
        return ResponseEntity.ok(response);
    }
    
    /**
     * Xóa liên hệ khẩn cấp
     * DELETE /api/emergency-contacts/{id}
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<BaseResponse<Void>> delete(
            @PathVariable Long id,
            Authentication authentication) {
        BaseResponse<Void> response = emergencyContactService.delete(id, authentication);
        return ResponseEntity.ok(response);
    }
}
