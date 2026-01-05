package com.example.Auto_BE.repository;

import com.example.Auto_BE.entity.SettingUser;
import com.example.Auto_BE.entity.Settings;
import com.example.Auto_BE.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface SettingUserRepository extends JpaRepository<SettingUser, Long> {
    
    /**
     * Tìm setting của user theo setting và user
     */
    Optional<SettingUser> findBySettingAndUser(Settings setting, User user);
    
    /**
     * Lấy tất cả settings của user
     */
    List<SettingUser> findByUserOrderBySetting_SettingKeyAsc(User user);
    
    /**
     * Lấy setting của user theo setting key và user
     */
    Optional<SettingUser> findBySetting_SettingKeyAndUser(String settingKey, User user);
    
    /**
     * Xóa tất cả settings của user
     */
    void deleteByUser(User user);
    
    /**
     * Kiểm tra user đã có setting này chưa
     */
    boolean existsBySettingAndUser(Settings setting, User user);
}

