package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.Verification;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface VerificationRepository extends JpaRepository<Verification, Long> {
    Optional<Verification> findByToken(String token);

    Optional<Verification> findByUserAndToken(User user, String token);

    boolean existsByToken(String token);

    void deleteAllByUser(User user);
}