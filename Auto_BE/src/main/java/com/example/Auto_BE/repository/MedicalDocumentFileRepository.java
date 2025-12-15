package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.MedicalDocumentFile;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface MedicalDocumentFileRepository extends JpaRepository<MedicalDocumentFile, Long> {
    
    /**
     * Tìm tất cả file của một tài liệu y tế
     */
    @Query("SELECT mdf FROM MedicalDocumentFile mdf WHERE mdf.medicalDocument.id = :documentId ORDER BY mdf.createdAt ASC")
    List<MedicalDocumentFile> findByMedicalDocumentId(@Param("documentId") Long documentId);
    
    /**
     * Tìm file theo ID và document ID (để kiểm tra quyền)
     */
    @Query("SELECT mdf FROM MedicalDocumentFile mdf WHERE mdf.id = :fileId AND mdf.medicalDocument.id = :documentId")
    Optional<MedicalDocumentFile> findByIdAndDocumentId(@Param("fileId") Long fileId, @Param("documentId") Long documentId);
    
    /**
     * Xóa tất cả file của một document
     */
    @Query("DELETE FROM MedicalDocumentFile mdf WHERE mdf.medicalDocument.id = :documentId")
    void deleteByMedicalDocumentId(@Param("documentId") Long documentId);
}
