package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.MedicalDocumentRequest;
import com.example.Auto_BE.dto.response.MedicalDocumentFileResponse;
import com.example.Auto_BE.dto.response.MedicalDocumentResponse;
import com.example.Auto_BE.service.MedicalDocumentService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api/medical-documents")
@RequiredArgsConstructor
public class MedicalDocumentController {

    private final MedicalDocumentService medicalDocumentService;

    /**
     * Tạo mới tài liệu y tế
     * POST /api/medical-documents
     */
    @PostMapping
    public ResponseEntity<BaseResponse<MedicalDocumentResponse>> createDocument(
            Authentication authentication,
            @RequestBody MedicalDocumentRequest request
    ) {
        String userEmail = authentication.getName();
        BaseResponse<MedicalDocumentResponse> response = medicalDocumentService.createDocument(userEmail, request);
        return ResponseEntity.ok(response);
    }

    /**
     * Upload file vào tài liệu y tế
     * POST /api/medical-documents/{documentId}/files
     */
    @PostMapping(value = "/{documentId}/files", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<BaseResponse<MedicalDocumentFileResponse>> uploadFile(
            Authentication authentication,
            @PathVariable Long documentId,
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "note", required = false) String note
    ) {
        String userEmail = authentication.getName();
        BaseResponse<MedicalDocumentFileResponse> response = 
                medicalDocumentService.uploadFile(userEmail, documentId, file, note);
        return ResponseEntity.ok(response);
    }

    /**
     * Lấy danh sách tài liệu y tế
     * GET /api/medical-documents?elderUserId=1 (optional for supervisor)
     */
    @GetMapping
    public ResponseEntity<BaseResponse<List<MedicalDocumentResponse>>> getDocuments(
            Authentication authentication,
            @RequestParam(required = false) Long elderUserId
    ) {
        String userEmail = authentication.getName();
        BaseResponse<List<MedicalDocumentResponse>> response = 
                medicalDocumentService.getDocuments(userEmail, elderUserId);
        return ResponseEntity.ok(response);
    }

    /**
     * Lấy chi tiết tài liệu y tế
     * GET /api/medical-documents/{documentId}
     */
    @GetMapping("/{documentId}")
    public ResponseEntity<BaseResponse<MedicalDocumentResponse>> getDocumentDetail(
            Authentication authentication,
            @PathVariable Long documentId
    ) {
        String userEmail = authentication.getName();
        BaseResponse<MedicalDocumentResponse> response = 
                medicalDocumentService.getDocumentDetail(userEmail, documentId);
        return ResponseEntity.ok(response);
    }

    /**
     * Cập nhật tài liệu y tế
     * PUT /api/medical-documents/{documentId}
     */
    @PutMapping("/{documentId}")
    public ResponseEntity<BaseResponse<MedicalDocumentResponse>> updateDocument(
            Authentication authentication,
            @PathVariable Long documentId,
            @RequestBody MedicalDocumentRequest request
    ) {
        String userEmail = authentication.getName();
        BaseResponse<MedicalDocumentResponse> response = 
                medicalDocumentService.updateDocument(userEmail, documentId, request);
        return ResponseEntity.ok(response);
    }

    /**
     * Xóa tài liệu y tế
     * DELETE /api/medical-documents/{documentId}
     */
    @DeleteMapping("/{documentId}")
    public ResponseEntity<BaseResponse<String>> deleteDocument(
            Authentication authentication,
            @PathVariable Long documentId
    ) {
        String userEmail = authentication.getName();
        BaseResponse<String> response = medicalDocumentService.deleteDocument(userEmail, documentId);
        return ResponseEntity.ok(response);
    }

    /**
     * Xóa file khỏi tài liệu
     * DELETE /api/medical-documents/{documentId}/files/{fileId}
     */
    @DeleteMapping("/{documentId}/files/{fileId}")
    public ResponseEntity<BaseResponse<String>> deleteFile(
            Authentication authentication,
            @PathVariable Long documentId,
            @PathVariable Long fileId
    ) {
        String userEmail = authentication.getName();
        BaseResponse<String> response = medicalDocumentService.deleteFile(userEmail, documentId, fileId);
        return ResponseEntity.ok(response);
    }
}
