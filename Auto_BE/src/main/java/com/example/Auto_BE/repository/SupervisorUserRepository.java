package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.SupervisorUser;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface SupervisorUserRepository extends JpaRepository<SupervisorUser, Long> {
    
    Optional<SupervisorUser> findByEmail(String email);
    
//    boolean existsByEmail(String email);
//
//    Optional<SupervisorUser> findByLicenseNumber(String licenseNumber);
//
//    @Query("SELECT s FROM SupervisorUser s LEFT JOIN FETCH s.elderRelations WHERE s.id = :id")
//    Optional<SupervisorUser> findByIdWithElderRelations(@Param("id") Long id);
}
