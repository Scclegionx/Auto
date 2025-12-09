package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.EmergencyContact;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface EmergencyContactRepository extends JpaRepository<EmergencyContact, Long> {
    
    /**
     * Tìm tất cả liên hệ khẩn cấp của một elder user
     */
    List<EmergencyContact> findByElderUserOrderByCreatedAtDesc(ElderUser elderUser);
    
    /**
     * Tìm liên hệ khẩn cấp theo ID và elder user (để đảm bảo user chỉ truy cập được liên hệ của mình)
     */
    Optional<EmergencyContact> findByIdAndElderUser(Long id, ElderUser elderUser);
    
    /**
     * Đếm số lượng liên hệ khẩn cấp của elder user
     */
    long countByElderUser(ElderUser elderUser);
    
    /**
     * Kiểm tra liên hệ có tồn tại theo ID và elder user
     */
    boolean existsByIdAndElderUser(Long id, ElderUser elderUser);
}
