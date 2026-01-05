package com.example.Auto_BE.dto.request;

import com.example.Auto_BE.entity.enums.EUserType;
import jakarta.validation.constraints.Size;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UpdateUserGuideRequest {

    @Size(max = 255, message = "Title too long")
    private String title; // Tiêu đề hướng dẫn

    @Size(max = 1000, message = "Description too long")
    private String description; // Mô tả chi tiết

    private EUserType userType; // ELDER hoặc SUPERVISOR

    @Size(max = 500, message = "Video URL too long")
    private String videoUrl; // URL video (optional - chỉ update khi cần thay đổi video)

    @Size(max = 500, message = "Thumbnail URL too long")
    private String thumbnailUrl; // URL thumbnail (optional)

    private Integer displayOrder; // Thứ tự hiển thị (optional)
}

