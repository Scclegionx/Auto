package com.example.Auto_BE.event;

import com.example.Auto_BE.entity.Verification;
import org.springframework.context.ApplicationEvent;

public class UserRegistrationEvent extends ApplicationEvent {
    private final Verification verification;

    public UserRegistrationEvent(Object source, Verification verification) {
        super(source);
        this.verification = verification;
    }
    public Verification getVerification() {
        return verification;
    }
}
