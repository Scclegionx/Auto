package com.example.Auto_BE.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.time.LocalDateTime;

@Entity
@Table(name = "verifications")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Accessors(chain = true)
public class Verification extends BaseEntity{
    @Column(name = "token", nullable = false, unique = true)
    private String token; // Mã xác thực duy nhất

    @Column(name = "expired_at", nullable = false)
    private LocalDateTime expiredAt; // Thời gian hết hạn của mã xác thực, tính bằng mili giây từ epoch (Unix timestamp)

    @ManyToOne(fetch = jakarta.persistence.FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user; // Người dùng liên kết với mã xác thực
}
