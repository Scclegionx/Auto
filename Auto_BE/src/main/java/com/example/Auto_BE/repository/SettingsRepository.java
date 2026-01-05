package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Settings;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface SettingsRepository extends JpaRepository<Settings, Long> {
    
    /**
     * Tìm setting theo key
     */
    Optional<Settings> findBySettingKey(String settingKey);
    
    /**
     * Lấy tất cả settings đang active
     */
    List<Settings> findByIsActiveTrueOrderBySettingKeyAsc();
    
    /**
     * Kiểm tra setting key đã tồn tại chưa
     */
    boolean existsBySettingKey(String settingKey);
}

