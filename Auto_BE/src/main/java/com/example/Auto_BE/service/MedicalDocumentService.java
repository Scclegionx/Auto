package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.MedicalDocumentRequest;
import com.example.Auto_BE.dto.response.MedicalDocumentFileResponse;
import com.example.Auto_BE.dto.response.MedicalDocumentResponse;
import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.MedicalDocument;
import com.example.Auto_BE.entity.MedicalDocumentFile;
import com.example.Auto_BE.entity.SupervisorUser;
import com.example.Auto_BE.repository.ElderSupervisorRepository;
import com.example.Auto_BE.repository.ElderUserRepository;
import com.example.Auto_BE.repository.MedicalDocumentFileRepository;
import com.example.Auto_BE.repository.MedicalDocumentRepository;
import com.example.Auto_BE.repository.SupervisorUserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@Slf4j
@RequiredArgsConstructor
public class MedicalDocumentService {
    
    private static final String SUCCESS = "success";
    private static final String FAILED = "failed";

    private final MedicalDocumentRepository medicalDocumentRepository;
    private final MedicalDocumentFileRepository medicalDocumentFileRepository;
    private final ElderUserRepository elderUserRepository;
    private final SupervisorUserRepository supervisorUserRepository;
    private final ElderSupervisorRepository elderSupervisorRepository;
    private final CloudinaryService cloudinaryService;

    private static final List<String> ALLOWED_IMAGE_TYPES = Arrays.asList(
        "image/jpeg", "image/jpg", "image/png", "image/webp"
    );
    private static final List<String> ALLOWED_DOCUMENT_TYPES = Arrays.asList(
        "application/pdf"
    );
    private static final long MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

    /**
     * Tạo mới tài liệu y tế
     */
    @Transactional
    public BaseResponse<MedicalDocumentResponse> createDocument(
            String userEmail,
            MedicalDocumentRequest request
    ) {
        try {
            // Xác định elder user
            ElderUser elderUser = determineElderUser(userEmail, request.getElderUserId());
            
            // Tạo tài liệu
            MedicalDocument document = new MedicalDocument();
            document.setName(request.getName());
            document.setDescription(request.getDescription());
            document.setElderUser(elderUser);
            
            document = medicalDocumentRepository.save(document);
            log.info("Created medical document ID {} for elder user ID {}", document.getId(), elderUser.getId());
            
            return BaseResponse.<MedicalDocumentResponse>builder()
                    .status(SUCCESS)
                    .message("Tạo tài liệu y tế thành công")
                    .data(convertToResponse(document))
                    .build();
                    
        } catch (RuntimeException e) {
            log.error("Error creating medical document: {}", e.getMessage());
            return BaseResponse.<MedicalDocumentResponse>builder()
                    .status(FAILED)
                    .message(e.getMessage())
                    .build();
        }
    }

    /**
     * Upload file vào tài liệu y tế
     */
    @Transactional
    public BaseResponse<MedicalDocumentFileResponse> uploadFile(
            String userEmail,
            Long documentId,
            MultipartFile file,
            String note
    ) {
        try {
            // Kiểm tra file
            validateFile(file);
            
            // Kiểm tra document và quyền
            MedicalDocument document = medicalDocumentRepository.findById(documentId)
                    .orElseThrow(() -> new RuntimeException("Không tìm thấy tài liệu y tế"));
            
            checkPermission(userEmail, document.getElderUser().getId());
            
            // Upload lên Cloudinary
            Map<String, Object> uploadResult = cloudinaryService.upload(file);
            String fileUrl = (String) uploadResult.get("secure_url");
            
            // Lưu thông tin file
            MedicalDocumentFile documentFile = new MedicalDocumentFile();
            documentFile.setFileName(file.getOriginalFilename());
            documentFile.setFileType(file.getContentType());
            documentFile.setFileUrl(fileUrl);
            documentFile.setFileSize((int) file.getSize());
            documentFile.setNote(note);
            documentFile.setMedicalDocument(document);
            
            documentFile = medicalDocumentFileRepository.save(documentFile);
            log.info("Uploaded file {} to document ID {}", documentFile.getFileName(), documentId);
            
            return BaseResponse.<MedicalDocumentFileResponse>builder()
                    .status(SUCCESS)
                    .message("Upload file thành công")
                    .data(convertToFileResponse(documentFile))
                    .build();
                    
        } catch (IOException e) {
            log.error("Error uploading file: {}", e.getMessage());
            return BaseResponse.<MedicalDocumentFileResponse>builder()
                    .status(FAILED)
                    .message("Lỗi upload file: " + e.getMessage())
                    .build();
        } catch (RuntimeException e) {
            log.error("Error uploading file: {}", e.getMessage());
            return BaseResponse.<MedicalDocumentFileResponse>builder()
                    .status(FAILED)
                    .message(e.getMessage())
                    .build();
        }
    }

    /**
     * Lấy danh sách tài liệu y tế
     */
    public BaseResponse<List<MedicalDocumentResponse>> getDocuments(
            String userEmail,
            Long elderUserId
    ) {
        try {
            ElderUser elderUser = determineElderUser(userEmail, elderUserId);
            
            List<MedicalDocument> documents = medicalDocumentRepository.findByElderUserId(elderUser.getId());
            List<MedicalDocumentResponse> responses = documents.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());
            
            return BaseResponse.<List<MedicalDocumentResponse>>builder()
                    .status(SUCCESS)
                    .message("Lấy danh sách tài liệu y tế thành công")
                    .data(responses)
                    .build();
                    
        } catch (RuntimeException e) {
            log.error("Error getting documents: {}", e.getMessage());
            return BaseResponse.<List<MedicalDocumentResponse>>builder()
                    .status(FAILED)
                    .message(e.getMessage())
                    .build();
        }
    }

    /**
     * Lấy chi tiết tài liệu y tế
     */
    public BaseResponse<MedicalDocumentResponse> getDocumentDetail(
            String userEmail,
            Long documentId
    ) {
        try {
            MedicalDocument document = medicalDocumentRepository.findById(documentId)
                    .orElseThrow(() -> new RuntimeException("Không tìm thấy tài liệu y tế"));
            
            checkPermission(userEmail, document.getElderUser().getId());
            
            return BaseResponse.<MedicalDocumentResponse>builder()
                    .status(SUCCESS)
                    .message("Lấy chi tiết tài liệu y tế thành công")
                    .data(convertToResponse(document))
                    .build();
                    
        } catch (RuntimeException e) {
            log.error("Error getting document detail: {}", e.getMessage());
            return BaseResponse.<MedicalDocumentResponse>builder()
                    .status(FAILED)
                    .message(e.getMessage())
                    .build();
        }
    }

    /**
     * Cập nhật tài liệu y tế
     */
    @Transactional
    public BaseResponse<MedicalDocumentResponse> updateDocument(
            String userEmail,
            Long documentId,
            MedicalDocumentRequest request
    ) {
        try {
            MedicalDocument document = medicalDocumentRepository.findById(documentId)
                    .orElseThrow(() -> new RuntimeException("Không tìm thấy tài liệu y tế"));
            
            checkPermission(userEmail, document.getElderUser().getId());
            
            document.setName(request.getName());
            document.setDescription(request.getDescription());
            document = medicalDocumentRepository.save(document);
            
            log.info("Updated medical document ID {}", documentId);
            
            return BaseResponse.<MedicalDocumentResponse>builder()
                    .status(SUCCESS)
                    .message("Cập nhật tài liệu y tế thành công")
                    .data(convertToResponse(document))
                    .build();
                    
        } catch (RuntimeException e) {
            log.error("Error updating document: {}", e.getMessage());
            return BaseResponse.<MedicalDocumentResponse>builder()
                    .status(FAILED)
                    .message(e.getMessage())
                    .build();
        }
    }

    /**
     * Xóa tài liệu y tế và tất cả file liên quan
     */
    @Transactional
    public BaseResponse<String> deleteDocument(
            String userEmail,
            Long documentId
    ) {
        try {
            MedicalDocument document = medicalDocumentRepository.findById(documentId)
                    .orElseThrow(() -> new RuntimeException("Không tìm thấy tài liệu y tế"));
            
            checkPermission(userEmail, document.getElderUser().getId());
            
            // Xóa tất cả file trên Cloudinary
            List<MedicalDocumentFile> files = medicalDocumentFileRepository.findByMedicalDocumentId(documentId);
            for (MedicalDocumentFile file : files) {
                try {
                    cloudinaryService.deleteImage(file.getFileUrl());
                } catch (Exception e) {
                    log.warn("Failed to delete file from Cloudinary: {}", file.getFileUrl());
                }
            }
            
            // Xóa document (cascade sẽ xóa files trong DB)
            medicalDocumentRepository.delete(document);
            log.info("Deleted medical document ID {} with {} files", documentId, files.size());
            
            return BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Xóa tài liệu y tế thành công")
                    .data("Đã xóa tài liệu và " + files.size() + " file đính kèm")
                    .build();
                    
        } catch (RuntimeException e) {
            log.error("Error deleting document: {}", e.getMessage());
            return BaseResponse.<String>builder()
                    .status(FAILED)
                    .message(e.getMessage())
                    .build();
        }
    }

    /**
     * Xóa một file khỏi tài liệu
     */
    @Transactional
    public BaseResponse<String> deleteFile(
            String userEmail,
            Long documentId,
            Long fileId
    ) {
        try {
            MedicalDocument document = medicalDocumentRepository.findById(documentId)
                    .orElseThrow(() -> new RuntimeException("Không tìm thấy tài liệu y tế"));
            
            checkPermission(userEmail, document.getElderUser().getId());
            
            MedicalDocumentFile file = medicalDocumentFileRepository.findByIdAndDocumentId(fileId, documentId)
                    .orElseThrow(() -> new RuntimeException("Không tìm thấy file"));
            
            // Xóa trên Cloudinary
            try {
                cloudinaryService.deleteImage(file.getFileUrl());
            } catch (Exception e) {
                log.warn("Failed to delete file from Cloudinary: {}", file.getFileUrl());
            }
            
            // Xóa trong DB
            medicalDocumentFileRepository.delete(file);
            log.info("Deleted file ID {} from document ID {}", fileId, documentId);
            
            return BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Xóa file thành công")
                    .build();
                    
        } catch (RuntimeException e) {
            log.error("Error deleting file: {}", e.getMessage());
            return BaseResponse.<String>builder()
                    .status(FAILED)
                    .message(e.getMessage())
                    .build();
        }
    }

    // ============ HELPER METHODS ============

    /**
     * Xác định elder user (cho phép supervisor tạo/xem hộ)
     */
    private ElderUser determineElderUser(String userEmail, Long elderUserId) {
        // Thử tìm elder user trực tiếp
        ElderUser elderUser = elderUserRepository.findByEmail(userEmail).orElse(null);
        
        if (elderUser != null) {
            if (elderUserId != null && !elderUser.getId().equals(elderUserId)) {
                throw new RuntimeException("Không có quyền truy cập tài liệu của người dùng khác");
            }
            return elderUser;
        }
        
        // Nếu không phải elder, kiểm tra supervisor
        SupervisorUser supervisor = supervisorUserRepository.findByEmail(userEmail)
                .orElseThrow(() -> new RuntimeException("Không tìm thấy người dùng"));
        
        if (elderUserId == null) {
            throw new RuntimeException("Supervisor phải chỉ định elderUserId");
        }
        
        // Kiểm tra quyền supervisor
        boolean hasPermission = elderSupervisorRepository
                .findActiveWithViewPermission(supervisor.getId(), elderUserId)
                .isPresent();
        
        if (!hasPermission) {
            throw new RuntimeException("Không có quyền truy cập tài liệu của người dùng này");
        }
        
        return elderUserRepository.findById(elderUserId)
                .orElseThrow(() -> new RuntimeException("Không tìm thấy người cao tuổi"));
    }

    /**
     * Kiểm tra quyền truy cập
     */
    private void checkPermission(String userEmail, Long elderUserId) {
        ElderUser elderUser = elderUserRepository.findByEmail(userEmail).orElse(null);
        
        if (elderUser != null) {
            if (!elderUser.getId().equals(elderUserId)) {
                throw new RuntimeException("Không có quyền truy cập");
            }
            return;
        }
        
        SupervisorUser supervisor = supervisorUserRepository.findByEmail(userEmail)
                .orElseThrow(() -> new RuntimeException("Không tìm thấy người dùng"));
        
        boolean hasPermission = elderSupervisorRepository
                .findActiveWithViewPermission(supervisor.getId(), elderUserId)
                .isPresent();
        
        if (!hasPermission) {
            throw new RuntimeException("Không có quyền truy cập");
        }
    }

    /**
     * Validate file upload
     */
    private void validateFile(MultipartFile file) {
        if (file.isEmpty()) {
            throw new RuntimeException("File không được để trống");
        }
        
        String contentType = file.getContentType();
        boolean isValidType = ALLOWED_IMAGE_TYPES.contains(contentType) || 
                              ALLOWED_DOCUMENT_TYPES.contains(contentType);
        
        if (!isValidType) {
            throw new RuntimeException("Chỉ cho phép file ảnh (JPG, PNG, WebP) hoặc PDF");
        }
        
        if (file.getSize() > MAX_FILE_SIZE) {
            throw new RuntimeException("Kích thước file không được vượt quá 10MB");
        }
    }

    /**
     * Convert entity to response DTO
     */
    private MedicalDocumentResponse convertToResponse(MedicalDocument document) {
        List<MedicalDocumentFile> files = medicalDocumentFileRepository.findByMedicalDocumentId(document.getId());
        
        return MedicalDocumentResponse.builder()
                .id(document.getId())
                .name(document.getName())
                .description(document.getDescription())
                .elderUserId(document.getElderUser().getId())
                .elderUserName(document.getElderUser().getFullName())
                .createdAt(LocalDateTime.ofInstant(document.getCreatedAt(), ZoneId.systemDefault()))
                .updatedAt(LocalDateTime.ofInstant(document.getUpdatedAt(), ZoneId.systemDefault()))
                .files(files.stream().map(this::convertToFileResponse).collect(Collectors.toList()))
                .fileCount(files.size())
                .build();
    }

    private MedicalDocumentFileResponse convertToFileResponse(MedicalDocumentFile file) {
        return MedicalDocumentFileResponse.builder()
                .id(file.getId())
                .fileName(file.getFileName())
                .fileType(file.getFileType())
                .fileUrl(file.getFileUrl())
                .fileSize(file.getFileSize())
                .note(file.getNote())
                .medicalDocumentId(file.getMedicalDocument().getId())
                .createdAt(LocalDateTime.ofInstant(file.getCreatedAt(), ZoneId.systemDefault()))
                .build();
    }
}
