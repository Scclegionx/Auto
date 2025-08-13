package com.example.Auto_BE.event;

import com.example.Auto_BE.service.EmailService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

@Component
public class UserRegistrationListener {
    @Value("${app.url.base}")
    private String baseUrl;

    private final EmailService emailService;

    public UserRegistrationListener(EmailService emailService) {
        this.emailService = emailService;
    }

    @EventListener
    @Async
    public void handleUserRegistrationEvent(UserRegistrationEvent event) {
        String email = event.getVerification().getUser().getEmail();
        String token = event.getVerification().getToken();

        String verificationLink = baseUrl + "/api/auth/verify?token=" + token;

        String subject = "Verify your email address";
        String content = "Welcome! Please verify your email by clicking this link: " + verificationLink;

        emailService.sendEmail(email, subject, content);
    }
}
