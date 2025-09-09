package com.example.Auto_BE.exception;

import com.example.Auto_BE.dto.BaseResponse;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.HashMap;
import java.util.Map;

import static com.example.Auto_BE.constants.ErrorMessages.ERROR;
import static com.example.Auto_BE.constants.ErrorMessages.INVALID_INPUT;
import static com.example.Auto_BE.constants.ErrorMessages.UNAUTHORIZED;

@RestControllerAdvice
public class GlobalExceptionHandler {
    private ResponseEntity<BaseResponse<Object>> buildErrorResponse(HttpStatus status, String message) {
        return ResponseEntity.status(status)
                .body(BaseResponse.<Object>builder()
                        .status(ERROR)
                        .message(message)
                        .data(null)
                        .build());
    }

    // Handle EntityNotFoundException
    @ExceptionHandler(BaseException.EntityNotFoundException.class)
    public ResponseEntity<BaseResponse<Object>> handleEntityNotFound(BaseException.EntityNotFoundException ex) {
        return buildErrorResponse(HttpStatus.NOT_FOUND, ex.getMessage());
    }

    // Handle BadRequestException
    @ExceptionHandler(BaseException.BadRequestException.class)
    public ResponseEntity<BaseResponse<Object>> handleBadRequest(BaseException.BadRequestException ex) {
        return buildErrorResponse(HttpStatus.BAD_REQUEST, ex.getMessage());
    }

    // Handle ConflictException
    @ExceptionHandler(BaseException.ConflictException.class)
    public ResponseEntity<BaseResponse<Object>> handleConflict(BaseException.ConflictException ex) {
        return buildErrorResponse(HttpStatus.CONFLICT, ex.getMessage());
    }

    // Handle Authentication exceptions (sai email/password)
    @ExceptionHandler(BadCredentialsException.class)
    public ResponseEntity<BaseResponse<Object>> handleBadCredentials(BadCredentialsException ex) {
        return buildErrorResponse(HttpStatus.UNAUTHORIZED, UNAUTHORIZED);
    }

    // Handle general Authentication exceptions
    @ExceptionHandler(AuthenticationException.class)
    public ResponseEntity<BaseResponse<Object>> handleAuthentication(AuthenticationException ex) {
        return buildErrorResponse(HttpStatus.UNAUTHORIZED, ex.getMessage());
    }


    // Xử lý lỗi validation từ @Valid
    @ExceptionHandler({MethodArgumentNotValidException.class,
            HttpMessageNotReadableException.class,
            DataIntegrityViolationException.class})
    public ResponseEntity<BaseResponse<Object>> handleValidationExceptions(Exception ex) {
        return buildErrorResponse(HttpStatus.BAD_REQUEST, INVALID_INPUT + " - " + ex.getMessage());
    }


}