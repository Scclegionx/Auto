package com.example.Auto_BE.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MedicalDocumentFileResponse {
    private Long id;
    private String fileName;
    private String fileType;
    private String fileUrl;
    private Integer fileSize;
    private String note;
    private Long medicalDocumentId;
    private LocalDateTime createdAt;
}
