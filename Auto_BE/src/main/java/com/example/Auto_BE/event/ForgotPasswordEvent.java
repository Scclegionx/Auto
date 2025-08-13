package com.example.Auto_BE.event;

import org.springframework.context.ApplicationEvent;

public class ForgotPasswordEvent extends ApplicationEvent {
    private final String email; // Email address of the user requesting a password reset
    private final String newPassword; // New password to be set for the user

    public ForgotPasswordEvent(Object source, String email, String newPassword) {
        super(source);
        this.email = email;
        this.newPassword = newPassword;
    }
    public String getEmail() {
        return email;
    }
    public String getNewPassword() {
        return newPassword;
    }
}
