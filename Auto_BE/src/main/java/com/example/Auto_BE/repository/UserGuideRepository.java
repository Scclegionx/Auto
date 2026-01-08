package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.UserGuide;
import com.example.Auto_BE.entity.enums.EUserType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserGuideRepository extends JpaRepository<UserGuide, Long> {
    
    List<UserGuide> findByUserTypeOrderByDisplayOrderAsc(EUserType userType);
    
    Optional<UserGuide> findByIdAndUserType(Long id, EUserType userType);
}

