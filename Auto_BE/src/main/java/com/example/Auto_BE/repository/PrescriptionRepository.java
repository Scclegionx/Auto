package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Prescriptions;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PrescriptionRepository extends JpaRepository<Prescriptions, Long> {
    // You can define custom query methods here if needed
}
