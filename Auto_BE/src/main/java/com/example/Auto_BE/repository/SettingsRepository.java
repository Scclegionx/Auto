package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.Settings;
import com.example.Auto_BE.entity.enums.ESettingType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface SettingsRepository extends JpaRepository<Settings, Long> {
    
    Optional<Settings> findBySettingKey(String settingKey);
    
    List<Settings> findByIsActiveTrueOrderBySettingKeyAsc();
    
    boolean existsBySettingKey(String settingKey);
    
    List<Settings> findBySettingTypeAndIsActiveTrueOrderBySettingKeyAsc(ESettingType settingType);
    
    List<Settings> findBySettingTypeInAndIsActiveTrueOrderBySettingKeyAsc(List<ESettingType> settingTypes);
}

