package com.example.Auto_BE.constants;

public class ErrorMessages {
    public static final String ERROR = "error";
    public static final String USER_NOT_FOUND = "User not found";
    public static final String UNAUTHENTICATED = "{\"status\":\"error\",\"message\":\"Unauthenticated\",\"data\":null}";
    public static final String UNAUTHORIZED = "Unauthenticated";
    public static final String UNVERIFIED_ACCOUNT = "Unverified account ";
    public static final String EMAIL_ALREADY_EXISTS = "Email already exists";
    public static final String INVALID_INPUT = "Invalid input";
    public static final String PRESCRIPTION_NOT_FOUND = "Prescription not found";
    public static final String CURR_PASSWORD_INCORRECT = "Current password is incorrect";
    public static final String PASSWORD_ERROR = "New password must be different from current password";
    public static final String INTERNAL_SERVER_ERROR = "Internal server error";
    public static final String BAD_REQUEST = "Bad request";
    public static final String ENTITY_NOT_FOUND = "Entity not found";
    public static final String FORBIDDEN = "Forbidden access";
    public static final String USER_ALREADY_VERIFIED = "User already verified";
    public static final String PERMISSION_ERROR = "You don't have permission to update this prescription";
    public static final String MEDICATION_NOT_FOUND = "Medication reminder not found";
    public static final String DAY_ERROR = "daysOfWeek must be 7 characters";
    public static final String NOTIFICATION_NOT_FOUND = "Notification not found";
    public static final String CONFIRM_ERROR = "Cannot confirm taken - notification already marked as missed";
}