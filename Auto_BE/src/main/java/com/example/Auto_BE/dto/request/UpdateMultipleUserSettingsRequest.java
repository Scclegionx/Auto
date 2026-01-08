package com.example.Auto_BE.dto.request;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import lombok.*;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UpdateMultipleUserSettingsRequest {

    private Long userId;
    
    @NotEmpty(message = "Settings list cannot be empty")
    @Valid
    private List<UpdateUserSettingRequest> settings; // Danh sách các setting cần update
}

