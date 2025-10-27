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
        String otp = event.getVerification().getToken();

        // Gửi mã OTP thay vì link
        String subject = "Mã xác thực email của bạn";
        String content = String.format(
            "Xin chào!\n\n" +
            "Mã OTP xác thực email của bạn là: %s\n\n" +
            "Mã này sẽ hết hạn sau 5 phút.\n\n" +
            "Nếu bạn không yêu cầu mã này, vui lòng bỏ qua email này.\n\n" +
            "Trân trọng,\n" +
            "Auto Team",
            otp
        );

        emailService.sendEmail(email, subject, content);
    }
}
