package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

@Entity
@Table(name = "medical_document_files")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Accessors(chain = true)
public class MedicalDocumentFile extends BaseEntity {
    
    @Column(name = "file_name", nullable = false, length = 255)
    private String fileName; // Tên của tệp tin

    @Column(name = "file_type", length = 100)
    private String fileType; // Loại tệp tin, ví dụ: "image/jpeg", "application/pdf"
    
    @Column(name = "file_url", nullable = false, length = 500)
    private String fileUrl; // URL file trên cloud storage (Cloudinary, S3, v.v.)
    
    @Column(name = "file_size")
    private Integer fileSize; // Kích thước file tính bằng bytes
    
    @Column(name = "note", columnDefinition = "TEXT")
    private String note; // Ghi chú về file

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "medical_document_id", nullable = false)
    private MedicalDocument medicalDocument; // Liên kết với MedicalDocument

}