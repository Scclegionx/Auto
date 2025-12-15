package com.example.Auto_BE.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MedicalDocumentResponse {
    private Long id;
    private String name;
    private String description;
    private Long elderUserId;
    private String elderUserName;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<MedicalDocumentFileResponse> files;
    private Integer fileCount;
}
