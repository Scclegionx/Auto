package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.ESettingType;
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
    private String settingKey;
    
    @Column(name = "name", nullable = false, length = 255)
    private String name;
    
    @Column(name = "description", length = 500)
    private String description;
    
    @Column(name = "default_value", nullable = false, length = 255)
    private String defaultValue;
    
    @Column(name = "possible_values", length = 500)
    private String possibleValues;
    
    @Column(name = "is_active", nullable = false)
    private Boolean isActive = true;

    @Column(name = "setting_type", nullable = true)
    @Enumerated(EnumType.STRING)
    private ESettingType settingType;
}

