package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.EmergencyContact;
import com.example.Auto_BE.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface EmergencyContactRepository extends JpaRepository<EmergencyContact, Long> {
    
    /**
     * Tìm tất cả liên hệ khẩn cấp của một user
     */
    List<EmergencyContact> findByUserOrderByCreatedAtDesc(User user);
    
    /**
     * Tìm liên hệ khẩn cấp theo ID và user (để đảm bảo user chỉ truy cập được liên hệ của mình)
     */
    Optional<EmergencyContact> findByIdAndUser(Long id, User user);
    
    /**
     * Đếm số lượng liên hệ khẩn cấp của user
     */
    long countByUser(User user);
    
    /**
     * Kiểm tra liên hệ có tồn tại theo ID và user
     */
    boolean existsByIdAndUser(Long id, User user);
}
