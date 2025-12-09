package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.ElderUser;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface ElderUserRepository extends JpaRepository<ElderUser, Long> {
    
    Optional<ElderUser> findByEmail(String email);
    
    boolean existsByEmail(String email);
    
    @Query("SELECT e FROM ElderUser e LEFT JOIN FETCH e.emergencyContacts WHERE e.id = :id")
    Optional<ElderUser> findByIdWithEmergencyContacts(@Param("id") Long id);
    
    @Query("SELECT e FROM ElderUser e LEFT JOIN FETCH e.medicalDocuments WHERE e.id = :id")
    Optional<ElderUser> findByIdWithMedicalDocuments(@Param("id") Long id);
}
