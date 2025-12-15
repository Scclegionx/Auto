package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.MedicalDocument;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface MedicalDocumentRepository extends JpaRepository<MedicalDocument, Long> {
    
    /**
     * Tìm tất cả tài liệu y tế của elder user
     */
    @Query("SELECT md FROM MedicalDocument md WHERE md.elderUser.id = :elderUserId ORDER BY md.createdAt DESC")
    List<MedicalDocument> findByElderUserId(@Param("elderUserId") Long elderUserId);
    
    /**
     * Tìm tài liệu y tế của elder user với phân trang
     */
    @Query("SELECT md FROM MedicalDocument md WHERE md.elderUser.id = :elderUserId ORDER BY md.createdAt DESC")
    Page<MedicalDocument> findByElderUserId(@Param("elderUserId") Long elderUserId, Pageable pageable);
    
    /**
     * Tìm tài liệu y tế theo ID và elder user (để kiểm tra quyền)
     */
    @Query("SELECT md FROM MedicalDocument md WHERE md.id = :documentId AND md.elderUser.id = :elderUserId")
    Optional<MedicalDocument> findByIdAndElderUserId(@Param("documentId") Long documentId, @Param("elderUserId") Long elderUserId);
    
    /**
     * Tìm kiếm tài liệu y tế theo tên hoặc mô tả
     */
    @Query("SELECT md FROM MedicalDocument md WHERE md.elderUser.id = :elderUserId " +
           "AND (LOWER(md.name) LIKE LOWER(CONCAT('%', :keyword, '%')) " +
           "OR LOWER(md.description) LIKE LOWER(CONCAT('%', :keyword, '%'))) " +
           "ORDER BY md.createdAt DESC")
    List<MedicalDocument> searchByKeyword(@Param("elderUserId") Long elderUserId, @Param("keyword") String keyword);
}
