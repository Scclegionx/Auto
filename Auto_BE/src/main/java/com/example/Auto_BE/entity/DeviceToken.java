package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

@Entity
@Table(name = "device_tokens")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Accessors(chain = true)
public class DeviceToken extends BaseEntity {

    @Column(name = "fcm_token", nullable = false, length = 1000)
    private String fcmToken;

    @Column(name = "device_id", length = 255)
    private String deviceId;

    @Column(name = "device_type", length = 50)
    private String deviceType; // iOS, Android, Web

    @Column(name = "device_name", length = 255)
    private String deviceName;

    @Column(name = "is_active")
    private Boolean isActive = true;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
}
