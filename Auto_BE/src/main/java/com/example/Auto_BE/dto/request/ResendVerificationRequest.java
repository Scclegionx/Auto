package com.example.Auto_BE.dto.request;

import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ResendVerificationRequest {
    private String email; // Email of the user to resend verification
}
