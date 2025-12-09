package com.example.Auto_BE.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class RespondToRequestDTO {
    
    private String message; // Lời nhắn khi accept/reject
}
