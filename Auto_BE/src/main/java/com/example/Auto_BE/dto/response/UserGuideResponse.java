package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.EUserType;
import lombok.*;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserGuideResponse {

    private Long id;
    private String title; // Tiêu đề hướng dẫn
    private String description; // Mô tả chi tiết
    private String videoUrl; // URL video trên Cloudinary
    private String thumbnailUrl; // URL thumbnail (ảnh đại diện)
    private EUserType userType; // ELDER hoặc SUPERVISOR
    private Integer displayOrder; // Thứ tự hiển thị
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}

