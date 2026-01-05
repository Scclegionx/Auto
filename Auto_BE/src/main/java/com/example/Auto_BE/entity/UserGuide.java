package com.example.Auto_BE.entity;

import com.example.Auto_BE.entity.enums.EUserType;
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
@Table(name = "user_guides")
public class UserGuide extends BaseEntity {
    
    @Column(name = "title", nullable = false, length = 255)
    private String title; // Tiêu đề hướng dẫn
    
    @Column(name = "description", length = 1000)
    private String description; // Mô tả chi tiết
    
    @Column(name = "video_url", nullable = false, length = 500)
    private String videoUrl; // URL video trên Cloudinary
    
    @Column(name = "thumbnail_url", length = 500)
    private String thumbnailUrl; // URL thumbnail (ảnh đại diện) - optional
    
    @Column(name = "user_type", nullable = false, length = 50)
    @Enumerated(EnumType.STRING)
    private EUserType userType; // ELDER hoặc SUPERVISOR
    
    @Column(name = "display_order")
    private Integer displayOrder; // Thứ tự hiển thị (số nhỏ hơn hiển thị trước)
}

