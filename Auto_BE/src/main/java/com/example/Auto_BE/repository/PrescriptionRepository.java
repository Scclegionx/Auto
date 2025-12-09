package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Prescriptions;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface PrescriptionRepository extends JpaRepository<Prescriptions, Long> {
    // Lấy tất cả đơn thuốc của user, sắp xếp theo ngày tạo (mới nhất trước)
    List<Prescriptions> findByElderUser_IdOrderByCreatedAtDesc(Long userId);
}