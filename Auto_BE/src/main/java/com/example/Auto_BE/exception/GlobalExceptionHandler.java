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


    // Xử lý lỗi validation từ @Valid - trả về chi tiết lỗi từng field
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<BaseResponse<Map<String, String>>> handleMethodArgumentNotValid(MethodArgumentNotValidException ex) {
        Map<String, String> errors = new HashMap<>();
        StringBuilder errorMessage = new StringBuilder("Dữ liệu không hợp lệ: ");
        
        ex.getBindingResult().getAllErrors().forEach((error) -> {
            String fieldName = ((FieldError) error).getField();
            String errorMsg = error.getDefaultMessage();
            errors.put(fieldName, errorMsg);
            errorMessage.append(fieldName).append(" - ").append(errorMsg).append("; ");
        });
        
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body(BaseResponse.<Map<String, String>>builder()
                        .status(ERROR)
                        .message(errorMessage.toString())
                        .data(errors)
                        .build());
    }

    // Xử lý lỗi parse JSON
    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<BaseResponse<Object>> handleHttpMessageNotReadable(HttpMessageNotReadableException ex) {
        String message = "Dữ liệu không hợp lệ. Vui lòng kiểm tra format JSON.";
        
        // Chi tiết hóa message cho các lỗi phổ biến
        String exMessage = ex.getMessage();
        if (exMessage != null) {
            // Extract field name từ error message nếu có
            if (exMessage.contains("LocalDate")) {
                message = "Định dạng ngày tháng không hợp lệ. Vui lòng sử dụng format: yyyy-MM-dd";
            } else if (exMessage.contains("LocalDateTime")) {
                message = "Định dạng ngày giờ không hợp lệ. Vui lòng sử dụng format: yyyy-MM-ddTHH:mm:ss";
            } else if (exMessage.contains("Cannot deserialize value of type")) {
                // Cố gắng extract field name và value
                if (exMessage.contains("ETypeMedication")) {
                    message = "Loại thuốc không hợp lệ. Các giá trị hợp lệ: OVER_THE_COUNTER, PRESCRIPTION";
                } else if (exMessage.contains("EGender")) {
                    message = "Giới tính không hợp lệ. Các giá trị hợp lệ: MALE, FEMALE, OTHER";
                } else if (exMessage.contains("EBloodType")) {
                    message = "Nhóm máu không hợp lệ. Các giá trị hợp lệ: A_POSITIVE, A_NEGATIVE, B_POSITIVE, B_NEGATIVE, AB_POSITIVE, AB_NEGATIVE, O_POSITIVE, O_NEGATIVE";
                } else {
                    message = "Kiểu dữ liệu không đúng. Vui lòng kiểm tra lại các trường số, ngày tháng, enum.";
                }
            } else if (exMessage.contains("JSON parse error")) {
                message = "Lỗi cú pháp JSON. Vui lòng kiểm tra dấu ngoặc, dấu phẩy.";
            }
            
            // Log full error để debug
            System.err.println("HttpMessageNotReadableException: " + exMessage);
        }
        
        return buildErrorResponse(HttpStatus.BAD_REQUEST, message);
    }

    // Xử lý lỗi database constraint
    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<BaseResponse<Object>> handleDataIntegrityViolation(DataIntegrityViolationException ex) {
        String message = "Dữ liệu vi phạm ràng buộc database";
        if (ex.getMessage().contains("duplicate") || ex.getMessage().contains("Duplicate")) {
            message = "Dữ liệu đã tồn tại trong hệ thống";
        }
        return buildErrorResponse(HttpStatus.CONFLICT, message);
    }


}