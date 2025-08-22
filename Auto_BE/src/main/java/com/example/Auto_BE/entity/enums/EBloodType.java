package com.example.Auto_BE.entity.enums;

import lombok.Getter;

@Getter
public enum EBloodType {
    A_POSITIVE("A+"),
    A_NEGATIVE("A-"),
    B_POSITIVE("B+"),
    B_NEGATIVE("B-"),
    AB_POSITIVE("AB+"),
    AB_NEGATIVE("AB-"),
    O_POSITIVE("O+"),
    O_NEGATIVE("O-");

    private final String value;

    EBloodType(String value) {
        this.value = value;
    }

}
