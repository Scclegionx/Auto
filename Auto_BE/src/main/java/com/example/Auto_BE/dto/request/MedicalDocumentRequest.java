package com.example.Auto_BE.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MedicalDocumentRequest {
    private String name;
    private String description;
    private Long elderUserId;
}
