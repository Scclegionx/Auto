package com.example.Auto_BE.exception;

public class BaseException extends RuntimeException {
    public BaseException(String message) {
        super(message);
    }

    public static class EntityNotFoundException extends RuntimeException {
        public EntityNotFoundException(String message) {
            super(message);
        }
    }
    public static class BadRequestException extends RuntimeException {
        public BadRequestException(String message) {
            super(message);
        }
    }
    public static class UnauthorizedException extends RuntimeException {
        public UnauthorizedException() {
            super();
        }
    }

    public static class ConflictException extends RuntimeException {
        public ConflictException(String message) {
            super(message);
        }
    }
}