package com.example.Auto_BE.dto.request;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class GetUserSettingsRequest {

    private Long userId; // null nếu là GUEST, có giá trị nếu đã đăng nhập
}

