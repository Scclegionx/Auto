package com.example.Auto_BE.dto.request;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import lombok.*;

import java.util.List;

@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class PrescriptionCreateRequest {
    @NotBlank(message = "Tên đơn thuốc không được để trống")
    @Size(min = 3, max = 200, message = "Tên đơn thuốc phải từ 3 đến 200 ký tự")
    private String name;

    @NotBlank(message = "Mô tả không được để trống")
    @Size(min = 10, max = 1000, message = "Mô tả phải từ 10 đến 1000 ký tự")
    private String description;

    @NotBlank(message = "URL hình ảnh không được để trống")
    private String imageUrl;

    @Valid
    @NotEmpty(message = "Danh sách thuốc không được để trống")
    private List<MedicationReminderCreateRequest> medicationReminders;
}