package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Prescriptions;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface PrescriptionRepository extends JpaRepository<Prescriptions, Long> {
    List<Prescriptions> findByUserIdAndIsActiveTrue(Long userId);
}
