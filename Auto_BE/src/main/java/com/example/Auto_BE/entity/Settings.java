package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

@Entity
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Accessors(chain = true)
@Table(name = "settings")
public class Settings extends BaseEntity {
    
    @Column(name = "setting_key", nullable = false, unique = true, length = 100)
    private String settingKey; // Key duy nhất: "theme", "font_size", "voice_support"
    
    @Column(name = "name", nullable = false, length = 255)
    private String name; // Tên hiển thị: "Nền", "Font Size", "Hỗ trợ nói"
    
    @Column(name = "description", length = 500)
    private String description; // Mô tả chi tiết
    
    @Column(name = "default_value", nullable = false, length = 255)
    private String defaultValue; // Giá trị mặc định: "light", "14", "on"
    
    @Column(name = "possible_values", length = 500)
    private String possibleValues; // Các giá trị có thể: "light,dark" hoặc "10,12,14,16,18" hoặc "on,off"
    
    @Column(name = "is_active", nullable = false)
    private Boolean isActive = true; // Trạng thái active/inactive
}

