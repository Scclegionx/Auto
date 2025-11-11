package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 💊 UNIFIED Response cho Medication
 * 
 * ✅ Đồng bộ với CreateMedicationRequest
 * ✅ Hỗ trợ multiple reminder times
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@JsonIgnoreProperties({"reminderTime"}) // Ignore deprecated field
public class MedicationResponse {

    private Long id;
    private Long userId;
    private String userName;
    private Long prescriptionId;
    
    // ✅ UNIFIED FIELDS
    private String medicationName;
    private ETypeMedication type;                  // ✅ Loại thuốc
    private List<String> reminderTimes;            // ✅ Array: ["08:00", "14:00", "20:00"]
    private String daysOfWeek;                     // ✅ "1111111"
    private String description;                    // ✅ Mô tả/ghi chú thuốc
    
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}