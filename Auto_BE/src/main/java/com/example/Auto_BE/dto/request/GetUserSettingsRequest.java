package com.example.Auto_BE.dto.request;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class GetUserSettingsRequest {

    private Long userId;
}

